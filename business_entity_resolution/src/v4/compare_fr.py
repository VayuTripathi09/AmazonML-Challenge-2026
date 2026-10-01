"""Compare C4 decisions on v2test vs v4test (France-locale). US/India must be identical; France: distribution + flips."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "v2"))
import polars as pl
from runlib import RUNS, cache_path, note
T = 0.7
C4 = RUNS / "model_C4_stack_on_C3_direct_x_more_data"
a = pl.read_parquet(C4 / "top3_v2test.parquet").select("q_idx", pl.col("s1_idx").alias("s_old"), pl.col("p1").alias("p_old"))
b = pl.read_parquet(C4 / "top3_v4test.parquet").select("q_idx", pl.col("s1_idx").alias("s_new"), pl.col("p1").alias("p_new"))
s1 = pl.read_parquet(cache_path("test_source1.parquet"), columns=["country"]).with_row_index("s1_idx")
d = a.join(b, on="q_idx", how="full", coalesce=True).join(s1.rename({"s1_idx": "s_old"}), on="s_old", how="left")
d = d.with_columns(pl.coalesce("country", pl.lit("France")).alias("country"))
for c in ("US", "India"):
    x = d.filter(pl.col("country") == c)
    same = ((x["s_old"] == x["s_new"]) & ((x["p_old"] - x["p_new"]).abs() < 1e-6)).mean()
    print(f"{c}: records {x.height:,}; identical best S1 and prob: {same:.6f}")
f = d.filter(pl.col("country") == "France")
for tag, col in (("old", "p_old"), ("new", "p_new")):
    p = f[col].fill_null(0)
    print(f"France {tag}: accept {(p >= T).mean():.4f}  p>=.97 {(p >= .97).mean():.4f}  uncertain[.3,.97) {((p >= .3) & (p < .97)).mean():.4f}  p<.03 {(p < .03).mean():.4f}")
acc_old, acc_new = f["p_old"].fill_null(0) >= T, f["p_new"].fill_null(0) >= T
print("France flips: newly accepted", int((~acc_old & acc_new).sum()), " newly rejected", int((acc_old & ~acc_new).sum()),
      " accepted both but different S1", int((acc_old & acc_new & (f["s_old"] != f["s_new"])).sum()))
f.write_parquet(RUNS / "v4_data" / "fr_compare.parquet")
note(f"France-locale C4 comparison written; newly accepted {int((~acc_old & acc_new).sum())}, newly rejected {int((acc_old & ~acc_new).sum())}")
