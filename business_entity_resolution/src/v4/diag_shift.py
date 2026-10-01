"""Train(DEV) vs test distribution diagnostics by country for the C4/C3 pipeline (no labels on test)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "v2"))
import polars as pl
from runlib import RUNS, cache_path, mem
pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
C3 = RUNS / "model_C3_direct_x_more_data"; C4 = RUNS / "model_C4_stack_on_C3_direct_x_more_data"
def load(split, s1split):
    t3 = pl.read_parquet(C3 / f"top3_{split}.parquet").select("q_idx", "s1_idx", pl.col("p1").alias("c3p1"), pl.col("p2").alias("c3p2"))
    t4 = pl.read_parquet(C4 / f"top3_{split}.parquet").select("q_idx", pl.col("p1").alias("c4p"))
    s1 = pl.read_parquet(cache_path(f"{s1split}_source1_norm.parquet"), columns=["country", "name_core"]).with_row_index("s1_idx")
    s1 = s1.with_columns(pl.len().over("country", "name_core").alias("name_dup"))
    q = pl.concat([pl.read_parquet(cache_path(f"{s1split}_{s}_norm.parquet"), columns=["addr_missing", "is_domain", "has_indic"]) for s in ("source2", "source3")]).with_row_index("q_idx")
    return t3.join(t4, on="q_idx").join(s1.select("s1_idx", "country", "name_dup"), on="s1_idx").join(q, on="q_idx")
dev = load("v2train", "v2train")
sp = pl.read_parquet(RUNS / "v2_data" / "qf.parquet").select("q_idx", "fold")
dev = dev.join(sp, on="q_idx").filter(pl.col("fold") == "DEV").with_columns(pl.col("country") + pl.lit(" (DEV)"))
test = load("v2test", "v2test").with_columns(pl.col("country") + pl.lit(" (TEST)"))
both = pl.concat([dev.drop("fold"), test])
print(both.group_by("country").agg(
    pl.len().alias("records"),
    (pl.col("c4p") >= 0.7).mean().alias("accept"),
    (pl.col("c4p") >= 0.97).mean().alias("p>=.97"),
    ((pl.col("c4p") >= 0.3) & (pl.col("c4p") < 0.97)).mean().alias("uncertain"),
    (pl.col("c4p") < 0.03).mean().alias("p<.03"),
    (pl.col("c3p1") - pl.col("c3p2")).median().alias("med_margin"),
    pl.col("addr_missing").mean().alias("no_addr"),
    pl.col("is_domain").mean().alias("domain"),
    (pl.col("name_dup") > 1).mean().alias("best_s1_name_dup"),
).sort("country"))
print(mem())
