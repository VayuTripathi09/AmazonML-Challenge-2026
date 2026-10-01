"""Where does US->India transfer lose? Compare tx model vs in-country C3 on India DEV, by slice."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "v2"))
import polars as pl
from runlib import RUNS, cache_path
T_TX, T_C3 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8, 0.7
tx = pl.read_parquet(RUNS / f"tx_{sys.argv[1]}" / "top1.parquet").rename({"s1_idx": "s_tx", "p1": "p_tx"})
c3 = pl.read_parquet(RUNS / "model_C3_direct_x_more_data" / "top3_v2train.parquet").select("q_idx", pl.col("s1_idx").alias("s_c3"), pl.col("p1").alias("p_c3"))
qf = pl.read_parquet(RUNS / "v2_data" / "qf.parquet")
sp = pl.read_parquet(RUNS / "v2_data" / "split_s1.parquet").select("s1_idx", "country", "fold")
flags = pl.concat([pl.read_parquet(cache_path(f"v2train_{s}_norm.parquet"), columns=["has_indic", "addr_missing", "is_domain"]) for s in ("source2", "source3")]).with_row_index("q_idx")
d = tx.join(c3, on="q_idx").join(qf, on="q_idx").join(flags, on="q_idx").join(sp.rename({"s1_idx": "s_tx"}), on="s_tx")
d = d.filter((pl.col("country") == "India") & (pl.col("fold") == "DEV"))
def cls(s, p, t):
    return (pl.when((pl.col(p) >= t) & (pl.col(s) == pl.col("true_s1"))).then(pl.lit("TP"))
            .when((pl.col(p) >= t) & pl.col("true_s1").is_null()).then(pl.lit("FP_unmatched"))
            .when(pl.col(p) >= t).then(pl.lit("FP_wrongS1"))
            .when(pl.col("true_s1").is_not_null() & (pl.col(s) == pl.col("true_s1"))).then(pl.lit("FN_below_t"))
            .when(pl.col("true_s1").is_not_null()).then(pl.lit("FN_best_wrong")).otherwise(pl.lit("TN")))
d = d.with_columns(cls("s_tx", "p_tx", T_TX).alias("tx"), cls("s_c3", "p_c3", T_C3).alias("c3"))
print(d.group_by("tx").len().join(d.group_by("c3").len().rename({"c3": "tx", "len": "len_c3"}), on="tx", how="full", coalesce=True).sort("tx"))
for col in ("has_indic", "addr_missing", "is_domain"):
    print(col, d.group_by(col).agg(pl.len(), (pl.col("tx").str.starts_with("F")).mean().alias("err_tx"), (pl.col("c3").str.starts_with("F")).mean().alias("err_c3")).sort(col).rows())
