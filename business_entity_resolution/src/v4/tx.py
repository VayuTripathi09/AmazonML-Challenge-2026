"""Cross-country transfer proxy for France (no France labels exist).
Fit a first-stage direct model on ONE country's TRAIN records, score the OTHER country's records, and measure
macro F0.5 on the other country's DEV S1 entities. Compares the C3 recipe with a monotone + regularized recipe.
usage: python tx.py <name> <train_country> <eval_country> [mono] [leaves] [min_leaf] [frac]"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v2"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from configs_v2 import CONFIGS, XCOLS  # noqa: E402
from eval_v2 import EvalCtx  # noqa: E402
from runlib import RUNS, Run, cache_path, mem, note  # noqa: E402

name, tr_c, ev_c = sys.argv[1], sys.argv[2], sys.argv[3]
VAR = sys.argv[4] if len(sys.argv) > 4 else "base"
MONO = VAR == "mono"
QNORM = "qnorm" in VAR
NOSCALE = "noscale" in VAR
LEAVES = int(sys.argv[5]) if len(sys.argv) > 5 else 255
MINLEAF = int(sys.argv[6]) if len(sys.argv) > 6 else 200
FRAC = float(sys.argv[7]) if len(sys.argv) > 7 else 0.25
FEATS = CONFIGS["C3_direct_x_more_data"]["feats"]
SCALE = {"len_q", "len_s", "ad_len_q", "ad_len_s", "num_q", "num_s", "n_cands", "comb_max", "tok_inter", "num_inter",
         "nm_q_only", "nm_s_only", "ad_q_only", "num_q_only", "num_s_only", "nm_shared_idf", "ad_shared_idf",
         "nm_qonly_idf_max", "nm_sonly_idf_max", "ad_qonly_idf_max", "ad_qonly_n_rare", "s1_name_dup", "n_name_ties"}
if NOSCALE:
    FEATS = [f for f in FEATS if f not in SCALE]
INC = {"cos_name", "cos_addr", "cos_comb", "comb_gap", "comb_gap_next", "name_gap", "addr_gap", "nm_ratio", "nm_tset", "nm_tsort",
       "nm_partial", "key_ratio", "key_partial", "key_jw", "full_ratio", "nm_exact", "key_exact", "tok_inter", "tok_jacc", "tok_q_cov",
       "tok_s_cov", "first_tok_eq", "ad_ratio", "ad_tset", "ad_tsort", "ad_partial", "ad_jacc", "ad_q_cov", "num_inter", "num_q_cov",
       "num_first_eq", "nm_diff_ratio", "ad_qonly_in_s", "num_set_eq", "m_cos_name", "m_cos_addr", "m_nm_tset", "m_full_ratio",
       "m_ad_tset", "m_num_q_cov", "nm_wcov_q", "nm_wcov_s", "nm_shared_idf", "ad_wcov_q", "ad_shared_idf", "prem_eq", "prem_in_s",
       "s_prem_in_q", "q_nums_subset", "raw_name_eq", "raw_addr_eq"}
DEC = {"rank", "num_conflict", "nm_q_only", "nm_s_only", "ad_q_only", "num_q_only", "num_s_only", "n_name_ties",
       "nm_qonly_idf_max", "nm_sonly_idf_max", "ad_qonly_idf_max", "ad_qonly_n_rare", "s1_name_dup"}
MONO_VEC = [1 if f in INC else -1 if f in DEC else 0 for f in FEATS]
SEED = 42
run = Run(f"tx_{name}", {"train": tr_c, "eval": ev_c, "var": VAR, "mono": MONO, "leaves": LEAVES, "min_leaf": MINLEAF, "frac": FRAC}, code_files=[__file__])


QDIR = RUNS / "v4_qmaps"
QDIR.mkdir(exist_ok=True)


def qmap(prefix, split="v2train"):
    """per-country, per-feature quantile grid (1001 points) from a 2M-pair sample of unlabeled candidates"""
    path = QDIR / f"{split}_{prefix}.npy"
    if path.exists():
        return np.load(path)
    rows = []
    for d in batches(prefix, split):
        rows.append(d.select(FEATS_ALL).sample(fraction=0.05, seed=1).to_numpy().astype(np.float32))
    X = np.concatenate(rows)
    Q = np.nanquantile(X, np.linspace(0, 1, 1001), axis=0).astype(np.float32)
    np.save(path, Q)
    return Q


def apply_q(X, Q):
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = np.searchsorted(Q[:, j], X[:, j], side="right") / 1001.0
    return out


FEATS_ALL = CONFIGS["C3_direct_x_more_data"]["feats"]


def batches(prefix, split="v2train"):
    feat_dir, x_dir = cache_path(f"{split}_feats"), cache_path(f"{split}_featx")
    base_cols = ["q_idx", "s1_idx"] + [c for c in FEATS_ALL if c not in XCOLS]
    for f in sorted(feat_dir.glob(f"{prefix}_*.parquet")):
        n, lo, hi = pl.scan_parquet(f).select(pl.len().alias("n"), pl.col("q_idx").min().alias("lo"), pl.col("q_idx").max().alias("hi")).collect().row(0)
        k = max(1, -(-n // 1_500_000))
        edges = [lo + (hi + 1 - lo) * i // k for i in range(k + 1)]
        for a, b in zip(edges[:-1], edges[1:]):
            rng = (pl.col("q_idx") >= a) & (pl.col("q_idx") < b)
            d = pl.scan_parquet(f).select(base_cols).filter(rng).collect()
            yield d.join(pl.scan_parquet(x_dir / f.name).filter(rng).collect(), on=["q_idx", "s1_idx"], how="left")


IDX = [FEATS_ALL.index(f) for f in FEATS]
if QNORM:
    QTR = qmap(tr_c)[:, IDX]
    QEV = qmap(ev_c)[:, IDX]
if not run.done("fit"):
    qf = pl.read_parquet(RUNS / "v2_data" / "qf.parquet")
    keep = (pl.col("fold") == "TRAIN") & ((pl.col("q_idx").hash(SEED) % 1000) < int(FRAC * 1000))
    es = (pl.col("fold") == "TRAIN") & ~keep & ((pl.col("q_idx").hash(SEED + 1) % 1000) < 15)
    Xa, ya, Xe, ye = [], [], [], []
    for d in batches(tr_c):
        d = d.join(qf, on="q_idx", how="left").with_columns((pl.col("s1_idx") == pl.col("true_s1")).fill_null(False).alias("y"))
        a, b = d.filter(keep), d.filter(es)
        xa, xe = a.select(FEATS).to_numpy().astype(np.float32), b.select(FEATS).to_numpy().astype(np.float32)
        if QNORM:
            xa, xe = apply_q(xa, QTR), apply_q(xe, QTR)
        Xa.append(xa); ya.append(a["y"].to_numpy())
        Xe.append(xe); ye.append(b["y"].to_numpy())
    Xa, ya, Xe, ye = map(np.concatenate, (Xa, ya, Xe, ye))
    print(f"fit rows {len(ya):,} es {len(ye):,}", mem(), flush=True)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=LEAVES, min_data_in_leaf=MINLEAF, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0 if not MONO else 10.0, max_bin=255, num_threads=12, seed=SEED, verbose=-1)
    if MONO:
        params.update(monotone_constraints=MONO_VEC, monotone_constraints_method="advanced")
    dtr = lgb.Dataset(Xa, ya, feature_name=FEATS, free_raw_data=True).construct()
    dva = lgb.Dataset(Xe, ye, reference=dtr).construct()
    del Xa, ya, Xe, ye
    t0 = time.time()
    m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(250)])
    m.save_model(str(run.path("model.txt")))
    run.complete("fit", files=["model.txt"], trees=m.best_iteration, sec=round(time.time() - t0))
    print("fit done", m.best_iteration, mem(), flush=True)

if not run.done("score"):
    m = lgb.Booster(model_file=str(run.path("model.txt")))
    parts = []
    for d in batches(ev_c):
        xs = d.select(FEATS).to_numpy().astype(np.float32)
        if QNORM:
            xs = apply_q(xs, QEV)
        p = m.predict(xs, num_threads=12).astype(np.float32)
        s = d.select("q_idx", "s1_idx").with_columns(pl.Series("p", p))
        parts.append(s.sort(["q_idx", "p"], descending=[False, True]).group_by("q_idx", maintain_order=True).agg(
            pl.col("s1_idx").first(), pl.col("p").first().alias("p1")))
    run.write_parquet(pl.concat(parts), "top1.parquet")
    run.complete("score", files=["top1.parquet"])

ctx = EvalCtx("DEV", "v2train")
ctx.eval_s1 = ctx.eval_s1.filter(pl.col("country") == ev_c)
ctx.truth = ctx.truth.join(ctx.eval_s1.select("s1_idx"), on="s1_idx", how="semi")
top = pl.read_parquet(run.path("top1.parquet"))
res = {t: ctx.score(top.filter(pl.col("p1") >= t))["F"].mean() for t in (0.5, 0.6, 0.7, 0.8, 0.9)}
best_t = max(res, key=res.get)
(run.path("eval.json")).write_text(json.dumps({"by_threshold": res, "best_t": best_t}, indent=1))
note(f"TRANSFER {name}: train {tr_c} -> {ev_c} DEV macro F0.5 " + ", ".join(f"t{t}={v:.5f}" for t, v in res.items())
     + f" | best {res[best_t]:.5f}@{best_t} (in-country C3 {ev_c}: see v2 log)")
run.release()
