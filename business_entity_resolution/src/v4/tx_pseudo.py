"""Self-training domain adaptation, validated on the US->India proxy before any use on France.
1. Source model (fit on US only) scores all India records (top-1 per record already in tx run dir).
2. Pseudo-labels from India records OUTSIDE the DEV fold only:
     record top-1 p >= HI  -> (record, top-1 S1) = 1, its other candidates = 0 (one S1 per record)
     record top-1 p <= LO  -> all its candidates = 0
3. Refit on US TRAIN rows + pseudo-labelled India rows; evaluate India DEV (true labels, never used above).
usage: python tx_pseudo.py <name> <source_tx_run> [HI] [LO] [pseudo_frac]"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v2"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from configs_v2 import CONFIGS, XCOLS  # noqa: E402
from eval_v2 import EvalCtx  # noqa: E402
from runlib import RUNS, Run, cache_path, mem, note  # noqa: E402

name, src = sys.argv[1], sys.argv[2]
HI = float(sys.argv[3]) if len(sys.argv) > 3 else 0.98
LO = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
PFRAC = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
SRC_C = sys.argv[6] if len(sys.argv) > 6 else "US"      # labelled source country
TGT_C = sys.argv[7] if len(sys.argv) > 7 else "India"   # pseudo-labelled / evaluated target country
FEATS = CONFIGS["C3_direct_x_more_data"]["feats"]
SEED = 42
run = Run(f"txp_{name}", {"src": src, "hi": HI, "lo": LO, "pfrac": PFRAC, "src_c": SRC_C, "tgt_c": TGT_C}, code_files=[__file__])


def batches(prefix):
    base_cols = ["q_idx", "s1_idx"] + [c for c in FEATS if c not in XCOLS]
    for f in sorted(cache_path("v2train_feats").glob(f"{prefix}_*.parquet")):
        n, lo, hi = pl.scan_parquet(f).select(pl.len().alias("n"), pl.col("q_idx").min().alias("lo"), pl.col("q_idx").max().alias("hi")).collect().row(0)
        k = max(1, -(-n // 1_500_000))
        edges = [lo + (hi + 1 - lo) * i // k for i in range(k + 1)]
        for a, b in zip(edges[:-1], edges[1:]):
            rng = (pl.col("q_idx") >= a) & (pl.col("q_idx") < b)
            d = pl.scan_parquet(f).select(base_cols).filter(rng).collect()
            yield d.join(pl.scan_parquet(cache_path("v2train_featx") / f.name).filter(rng).collect(), on=["q_idx", "s1_idx"], how="left")


qf = pl.read_parquet(RUNS / "v2_data" / "qf.parquet")
top = pl.read_parquet(RUNS / src / "top1.parquet").rename({"s1_idx": "top_s1", "p1": "top_p"})
if not run.done("fit"):
    keep_us = (pl.col("fold") == "TRAIN") & ((pl.col("q_idx").hash(SEED) % 1000) < 150)
    es = (pl.col("fold") == "TRAIN") & ~keep_us & ((pl.col("q_idx").hash(SEED + 1) % 1000) < 15)
    pick = (pl.col("fold") != "DEV") & ((pl.col("q_idx").hash(SEED + 3) % 1000) < int(PFRAC * 1000))

    def us_frames():
        for d in batches(SRC_C):  # labelled source rows, exactly as the source model
            d = d.join(qf, on="q_idx", how="left").with_columns((pl.col("s1_idx") == pl.col("true_s1")).fill_null(False).alias("y"))
            yield d.filter(keep_us), d.filter(es)

    def in_frames():
        for d in batches(TGT_C):  # pseudo-labelled target rows; true labels are NOT read here
            d = d.join(qf.select("q_idx", "fold"), on="q_idx", how="left").filter(pick).join(top, on="q_idx", how="inner")
            yield d.filter((pl.col("top_p") >= HI) | (pl.col("top_p") <= LO)).with_columns(
                ((pl.col("top_p") >= HI) & (pl.col("s1_idx") == pl.col("top_s1"))).alias("y"))

    n_tr = n_es = n_ps = 0
    for a_, b_ in us_frames():
        n_tr += a_.height; n_es += b_.height
    for d in in_frames():
        n_ps += d.height
    Xa = np.empty((n_tr + n_ps, len(FEATS)), np.float32); ya = np.empty(n_tr + n_ps, np.float32)
    Xe = np.empty((n_es, len(FEATS)), np.float32); ye = np.empty(n_es, np.float32)
    ia = ie = 0
    for a_, b_ in us_frames():
        Xa[ia:ia + a_.height] = a_.select(FEATS).to_numpy(); ya[ia:ia + a_.height] = a_["y"].to_numpy(); ia += a_.height
        Xe[ie:ie + b_.height] = b_.select(FEATS).to_numpy(); ye[ie:ie + b_.height] = b_["y"].to_numpy(); ie += b_.height
    n_pseudo = [0, 0]
    for d in in_frames():
        Xa[ia:ia + d.height] = d.select(FEATS).to_numpy(); ya[ia:ia + d.height] = d["y"].to_numpy(); ia += d.height
        n_pseudo[0] += d.height; n_pseudo[1] += int(d["y"].sum())
    print(f"fit rows {len(ya):,} (pseudo {n_pseudo[0]:,}, pseudo positives {n_pseudo[1]:,})", mem(), flush=True)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, num_threads=12, seed=SEED, verbose=-1)
    dtr = lgb.Dataset(Xa, ya, feature_name=FEATS, free_raw_data=True).construct()
    dva = lgb.Dataset(Xe, ye, reference=dtr).construct()  # early stopping on labelled SOURCE rows only
    del Xa, ya, Xe, ye
    m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(250)])
    m.save_model(str(run.path("model.txt")))
    run.complete("fit", files=["model.txt"], trees=m.best_iteration, pseudo=n_pseudo)

if not run.done("score"):
    m = lgb.Booster(model_file=str(run.path("model.txt")))
    parts = []
    for d in batches(TGT_C):
        p = m.predict(d.select(FEATS).to_numpy(), num_threads=12).astype(np.float32)
        s = d.select("q_idx", "s1_idx").with_columns(pl.Series("p", p))
        parts.append(s.sort(["q_idx", "p"], descending=[False, True]).group_by("q_idx", maintain_order=True).agg(
            pl.col("s1_idx").first(), pl.col("p").first().alias("p1")))
    run.write_parquet(pl.concat(parts), "top1.parquet")
    run.complete("score", files=["top1.parquet"])

ctx = EvalCtx("DEV", "v2train")
ctx.eval_s1 = ctx.eval_s1.filter(pl.col("country") == TGT_C)
ctx.truth = ctx.truth.join(ctx.eval_s1.select("s1_idx"), on="s1_idx", how="semi")
t1 = pl.read_parquet(run.path("top1.parquet"))
res = {t: ctx.score(t1.filter(pl.col("p1") >= t))["F"].mean() for t in (0.5, 0.6, 0.7, 0.8, 0.9)}
run.path("eval.json").write_text(json.dumps(res, indent=1))
note(f"PSEUDO {name} (src {src}, hi {HI}, lo {LO}): {SRC_C}+{TGT_C}-pseudo -> {TGT_C} DEV " + ", ".join(f"t{t}={v:.5f}" for t, v in res.items())
     + " | compare source-only best 0.92951")
run.release()
