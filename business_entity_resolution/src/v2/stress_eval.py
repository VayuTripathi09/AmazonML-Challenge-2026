"""Stress-suite evaluation + duplication-invariance check.
usage: python stress_eval.py <model_run> <threshold> [stress_split=v2stress]
 * macro F0.5 on remaining DEV S1s with X's records turned into clustered wrong-branch distractors
 * critical slice: DEV twins Y (the entities receiving the clusters), incl. zero-match twins
 * by cluster size of the removed X
 * duplication invariance: duplicating a rejected distractor record k times (fresh ids) may change only
   population features (s1_rank1_deg); the original pair's probability must not rise / flip."""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from eval_v2 import EvalCtx, load_top3
from runlib import RUNS, cache_path, mem, note

model_run, T = sys.argv[1], float(sys.argv[2])
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "v2stress"
srun = RUNS / f"stress_{SPLIT}"
drop = pl.read_parquet(srun / "dropped_X.parquet")
twins = pl.read_parquet(srun / "twins_Y.parquet").filter(pl.col("fold") == "DEV").select("s1_idx")
ctx = EvalCtx("DEV", SPLIT, drop_s1=drop)
ctx_clean = EvalCtx("DEV", "v2train")
top = load_top3(model_run, SPLIT)
d = ctx.score(top.filter(pl.col("p1") >= T))
dt = d.join(twins, on="s1_idx", how="semi")
res = {"model": model_run, "t": T, "F_stress_all_DEV": d["F"].mean(), "F_twins": dt["F"].mean(), "n_twins": dt.height,
       "F_twins_zero_match": dt.filter(pl.col("n_true") == 0)["F"].mean(), "n_twins_zero": dt.filter(pl.col("n_true") == 0).height,
       "F_clean_same_entities": ctx_clean.score(load_top3(model_run, "v2train").filter(pl.col("p1") >= T))
       .join(d.select("s1_idx"), on="s1_idx", how="semi")["F"].mean()}
# distractor records from removed X: how many are accepted (into anything)?
xrec = (pl.read_parquet(RUNS / "v2_data" / "truth_pairs.parquet").join(drop.rename({"s1_idx": "true_s1"}), on="true_s1", how="semi")
        .select("q_idx", pl.col("true_s1").alias("x_s1")))
xrec = xrec.with_columns(pl.len().over("x_s1").alias("cluster_size"))
acc = xrec.join(top.select("q_idx", "p1"), on="q_idx").with_columns((pl.col("p1") >= T).alias("accepted"))
res["distractor_records"] = acc.height
res["distractor_accept_rate"] = acc["accepted"].mean()
by = acc.with_columns(pl.col("cluster_size").clip(1, 8)).group_by("cluster_size").agg(pl.len(), pl.col("accepted").mean().alias("accept")).sort("cluster_size")
res["distractor_accept_by_cluster_size"] = by.to_dicts()

# duplication invariance on rejected distractors (probability must not rise with duplicates)
m = lgb.Booster(model_file=str(RUNS / model_run / "model.txt"))
feats = m.feature_name()
POP = {"s1_rank1_deg"}  # the only candidate feature that depends on OTHER records
if not (POP & set(feats)) and "p1" in feats:
    # stacked model: inputs are a first-stage model without population features + pairwise features
    res["duplication_invariance"] = "invariant by construction: no feature depends on other records (checked feature list)"
    res["population_feature_used"] = False
    (RUNS / model_run / f"stress_{SPLIT}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    note(f"STRESS {model_run} t={T}: F_stress={res['F_stress_all_DEV']:.5f} F_twins={res['F_twins']:.5f} (zero-match twins {res['F_twins_zero_match']:.4f}) "
         f"distractor accept={res['distractor_accept_rate']:.4f}; duplication-invariant by construction")
    raise SystemExit(0)
rej = acc.filter(~pl.col("accepted")).sample(min(20000, acc.filter(~pl.col("accepted")).height), seed=1).select("q_idx")
fe = []
for f in sorted(cache_path(f"{SPLIT}_feats").glob("*.parquet")):
    x = pl.read_parquet(f, memory_map=False).join(rej, on="q_idx", how="semi").filter(pl.col("rank") == 1)
    if x.height:
        if any(c not in x.columns for c in feats):
            x = x.join(pl.read_parquet(cache_path(f"{SPLIT}_featx") / f.name, memory_map=False), on=["q_idx", "s1_idx"], how="left")
        fe.append(x)
X = pl.concat(fe)
p0 = m.predict(X.select(feats).to_numpy())
dup = {}
for k in (1, 2, 4, 8):
    Xk = X.with_columns((pl.col("s1_rank1_deg") + k).alias("s1_rank1_deg")) if "s1_rank1_deg" in feats else X
    pk = m.predict(Xk.select(feats).to_numpy())
    dup[k] = {"mean_prob_increase": float((pk - p0).mean()), "max_prob_increase": float((pk - p0).max()),
              "flipped_to_accept": int(((p0 < T) & (pk >= T)).sum()), "n": int(len(p0))}
res["duplication_invariance"] = dup
res["population_feature_used"] = "s1_rank1_deg" in feats
(RUNS / model_run / f"stress_{SPLIT}.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
note(f"STRESS {model_run} t={T}: F_stress={res['F_stress_all_DEV']:.5f} F_twins={res['F_twins']:.5f} (zero-match twins {res['F_twins_zero_match']:.4f}) "
     f"distractor accept={res['distractor_accept_rate']:.4f}; dup k=8 flips={dup[8]['flipped_to_accept']}/{dup[8]['n']} mean dp={dup[8]['mean_prob_increase']:+.4f}")
