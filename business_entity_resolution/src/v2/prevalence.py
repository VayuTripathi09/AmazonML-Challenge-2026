"""Clustered-distractor prevalence sensitivity: duplicate every removed-X record's prediction k times (fresh ids).
Exact for duplication-invariant models (C3, C4); approximate for B0 (its population feature would also move)."""
import polars as pl
from eval_v2 import EvalCtx, load_top3
from runlib import RUNS, note
drop = pl.read_parquet(RUNS / "stress_v2stress" / "dropped_X.parquet")
ctx = EvalCtx("DEV", "v2stress", drop_s1=drop)
xrec = pl.read_parquet(RUNS / "v2_data" / "truth_pairs.parquet").join(drop.rename({"s1_idx": "true_s1"}), on="true_s1", how="semi").select("q_idx")
for run, t in (("model_B0_baseline_refit", 0.65), ("model_C3_direct_x_more_data", 0.7), ("model_C4_stack_on_C3_direct_x_more_data", 0.7)):
    top = load_top3(run, "v2stress").filter(pl.col("p1") >= t).select("s1_idx", "q_idx")
    xacc = top.join(xrec, on="q_idx", how="semi")
    row = []
    for k in (0, 1, 3, 7):
        extra = [xacc.with_columns((pl.col("q_idx").cast(pl.Int64) + (i + 1) * 20_000_000).alias("q_idx")) for i in range(k)]
        pred = pl.concat([top.with_columns(pl.col("q_idx").cast(pl.Int64))] + extra) if k else top
        row.append(f"x{k+1}: {ctx.score(pred)['F'].mean():.5f}")
    note(f"PREVALENCE {run} (cluster prevalence x1/x2/x4/x8{' - approximate for B0' if 'B0' in run else ' - exact'}): " + ", ".join(row))
