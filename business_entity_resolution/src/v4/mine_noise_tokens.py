"""Unsupervised: name/address tokens that differ between records and their S1 in HIGH-CONFIDENCE matches
(C4 p>=0.97, ~99% precise on DEV), per country. Reveals each country's benign noise vocabulary."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "v2"))
import polars as pl
from runlib import RUNS, cache_path, mem
C4 = RUNS / "model_C4_stack_on_C3_direct_x_more_data"
t = pl.read_parquet(C4 / "top3_v2test.parquet").filter(pl.col("p1") >= 0.97).select("q_idx", "s1_idx")
s1 = pl.read_parquet(cache_path("v2test_source1_norm.parquet"), columns=["country", "name_norm", "addr_norm"]).with_row_index("s1_idx")
q = pl.concat([pl.read_parquet(cache_path(f"v2test_{s}_norm.parquet"), columns=["name_norm", "addr_norm"]) for s in ("source2", "source3")]).with_row_index("q_idx")
d = t.join(s1, on="s1_idx").join(q, on="q_idx", suffix="_q").sample(fraction=0.25, seed=1)
for field in ("name_norm", "addr_norm"):
    x = d.select("country", pl.col(field + "_q").str.split(" ").alias("a"), pl.col(field).str.split(" ").alias("b"))
    x = x.select("country", pl.col("a").list.set_difference("b").alias("q_only"), pl.col("b").list.set_difference("a").alias("s1_only"))
    for side in ("q_only", "s1_only"):
        c = x.select("country", side).explode(side).drop_nulls().filter(pl.col(side) != "").group_by("country", side).len()
        tot = d.group_by("country").len().rename({"len": "n"})
        c = c.join(tot, on="country").with_columns((pl.col("len") / pl.col("n")).alias("rate"))
        for ctry in ("France", "US", "India"):
            top = c.filter(pl.col("country") == ctry).sort("len", descending=True).head(28)
            print(f"{field} {side} {ctry}: " + ", ".join(f"{r[side]}({r['rate']:.3f})" for r in top.iter_rows(named=True)))
print(mem())
