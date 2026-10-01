"""v2 evaluation on a chosen fold (DEV by default; CONF only for the single final confirmation).
usage: python eval_v2.py <model_run> [fold=DEV] [split=v2train] [--ceiling] [--budget]"""
import json
import sys

import numpy as np
import polars as pl

from runlib import RUNS, cache_path, mem, note
from scorer import per_entity

GRID = [round(x, 3) for x in np.arange(0.40, 0.951, 0.05)]


class EvalCtx:
    def __init__(self, fold="DEV", split="v2train", drop_s1=None):
        sp = pl.read_parquet(RUNS / "v2_data" / "split_s1.parquet")
        truth = pl.read_parquet(RUNS / "v2_data" / "truth_pairs.parquet").select(pl.col("true_s1").alias("s1_idx"), "q_idx")
        n_s2 = pl.scan_parquet(cache_path("train_source2.parquet")).select(pl.len()).collect().item()
        self.n_s2 = n_s2  # records with q_idx >= n_s2 are Source 3
        ev = sp.filter(pl.col("fold") == fold).select("s1_idx", "country")
        if drop_s1 is not None:  # stress suites: removed S1s are no longer evaluated; their records become distractors
            ev = ev.join(drop_s1, on="s1_idx", how="anti")
            truth = truth.join(drop_s1, on="s1_idx", how="anti")
        n_true = truth.group_by("s1_idx").agg(pl.len().alias("n_true"))
        dup = (pl.read_parquet(cache_path(f"{split}_source1_norm.parquet"), columns=["country", "name_core"]).with_row_index("s1_idx")
               .with_columns(pl.len().over("country", "name_core").alias("name_dup")).select("s1_idx", "name_dup"))
        self.eval_s1 = (ev.join(n_true, on="s1_idx", how="left").join(dup, on="s1_idx", how="left")
                        .with_columns(pl.col("n_true").fill_null(0)))
        self.truth = truth.join(self.eval_s1.select("s1_idx"), on="s1_idx", how="semi")
        self.fold, self.split = fold, split

    def score(self, pred):
        d = per_entity(pred.select("s1_idx", "q_idx"), self.truth, self.eval_s1)
        return d

    def summary(self, d):
        out = {"F": d["F"].mean(), "n": d.height}
        for c in d["country"].unique().sort().to_list():
            out[f"F_{c}"] = d.filter(pl.col("country") == c)["F"].mean()
        bins = d.with_columns(pl.col("n_true").cut([0, 1, 3], labels=["0", "1", "2-3", "4+"]).alias("b"))
        for b in ["0", "1", "2-3", "4+"]:
            out[f"F_ntrue_{b}"] = bins.filter(pl.col("b") == b)["F"].mean()
        out["F_ambiguous_name"] = d.filter(pl.col("name_dup") > 1)["F"].mean()
        return out


def load_top3(model_run, split):
    return pl.read_parquet(RUNS / model_run / f"top3_{split}.parquet")


def ceiling(ctx: EvalCtx, split="v2train"):
    """exact candidate-restricted oracle: perfect decisions on retrieved candidates."""
    hits = [ctx.truth.join(pl.read_parquet(f, columns=["q_idx", "s1_idx"], memory_map=False), on=["q_idx", "s1_idx"], how="semi")
            for f in sorted(cache_path(f"{split}_cands").glob("*.parquet"))]  # per file: bounded memory
    hit = pl.concat(hits).unique()
    return per_entity(hit, ctx.truth, ctx.eval_s1)["F"].mean()


def sweep(ctx, top, col="p1"):
    rows = []
    for t in GRID:
        d = ctx.score(top.filter(pl.col(col) >= t))
        rows.append({"t": t, **ctx.summary(d), "accept_rate_all_records": float((top[col] >= t).mean())})
    return pl.DataFrame(rows)


if __name__ == "__main__":
    model_run = sys.argv[1]
    fold = sys.argv[2] if len(sys.argv) > 2 else "DEV"
    split = sys.argv[3] if len(sys.argv) > 3 else "v2train"
    if fold == "CONF":
        raise SystemExit("CONF is sealed: use confirm_v2.py for the single final confirmation")
    ctx = EvalCtx(fold, split)
    top = load_top3(model_run, split)
    if "--ceiling" in sys.argv:
        note(f"candidate-restricted oracle macro F0.5 on {fold}: {ceiling(ctx, split):.5f}")
    s = sweep(ctx, top)
    print(s.select("t", "F", "F_India", "F_US", "F_ntrue_0", "F_ntrue_1", "F_ambiguous_name", "accept_rate_all_records"))
    best = s.sort("F", descending=True).row(0, named=True)
    (RUNS / model_run / f"eval_{fold}_{split}.json").write_text(json.dumps({"best": best, "sweep": s.to_dicts()}, indent=1))
    note(f"{model_run} {fold}/{split}: best F={best['F']:.5f} at t={best['t']} (India {best['F_India']:.5f}, US {best['F_US']:.5f}, "
         f"singletons {best['F_ntrue_0']:.5f}, accept {best['accept_rate_all_records']:.4f})")
