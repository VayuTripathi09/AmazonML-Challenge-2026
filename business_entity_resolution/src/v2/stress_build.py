"""Clustered wrong-branch stress suite built from REAL labeled records.

X = DEV S1 entities that have a same-name twin Y (same country, same core-name key, different address).
Removing X from the catalog turns X's true S2/S3 records into a coherent same-name / wrong-address group
that retrieves the twin Y: exactly the test-set failure pattern, with defensible ground truth (the records
belong to X, not to Y). Cluster size = X's number of true records (1..11).
usage: python stress_build.py [share=1.0]   (share of eligible X removed; default all)"""
import shutil
import sys

import polars as pl

from runlib import RUNS, Run, cache_path, mem, note

share = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
SPLIT = "v2stress" if share == 1.0 else f"v2stress{int(share*100)}"
run = Run(f"stress_{SPLIT}", {"share": share, "rule": "one DEV X per same-name DEV group (>=2 addresses); others are twins Y"}, code_files=[__file__])
sp = pl.read_parquet(RUNS / "v2_data" / "split_s1.parquet")
s1 = pl.read_parquet(cache_path("v2train_source1_norm.parquet"), columns=["country", "name_core", "addr_norm"]).with_row_index("s1_idx")
s1 = s1.join(sp.select("s1_idx", "fold"), on="s1_idx").with_columns(pl.col("name_core").str.replace_all(" ", "").alias("key"))
# twin groups: >=2 DEV members sharing (country, core-name key) at >=2 distinct addresses
dev = s1.filter((pl.col("fold") == "DEV") & (pl.col("key") != ""))
g = dev.group_by("country", "key").agg(pl.len().alias("n_dev"), pl.col("addr_norm").n_unique().alias("n_addr"))
g = g.filter((pl.col("n_dev") >= 2) & (pl.col("n_addr") >= 2))
if share < 1.0:
    g = g.filter((pl.struct("country", "key").hash(99) % 1000) < int(share * 1000))
members = dev.join(g.select("country", "key"), on=["country", "key"]).with_columns(pl.col("s1_idx").hash(7).alias("hh"))
members = members.with_columns(pl.col("hh").rank("ordinal").over("country", "key").alias("r"))
drop = members.filter(pl.col("r") == 1).select("s1_idx")          # one X per group
twins = members.filter(pl.col("r") > 1).select("s1_idx", "fold")  # remaining DEV members = evaluated twins Y
run.write_parquet(drop, "dropped_X.parquet")
run.write_parquet(twins, "twins_Y.parquet")
note(f"stress {SPLIT}: removed {drop.height:,} DEV S1 with same-name twins; twins Y remaining: {twins.height:,} "
     f"({twins.filter(pl.col('fold') == 'DEV').height:,} in DEV)")

if not run.done("inputs"):
    for s in ("source1", "source2", "source3"):
        shutil.copyfile(cache_path(f"v2train_{s}_norm.parquet"), cache_path(f"{SPLIT}_{s}_norm.parquet"))
        shutil.copyfile(cache_path(f"v2train_{s}.parquet"), cache_path(f"{SPLIT}_{s}.parquet"))
    out = cache_path(f"{SPLIT}_cands"); out.mkdir(exist_ok=True)
    xo = cache_path(f"{SPLIT}_featx"); xo.mkdir(exist_ok=True)
    n = 0
    for f in sorted(cache_path("v2train_cands").glob("*.parquet")):
        c = pl.read_parquet(f, memory_map=False).join(drop, on="s1_idx", how="anti")
        c = c.with_columns(pl.col("cos_comb").rank("ordinal", descending=True).over("q_idx").cast(pl.Int16).alias("rank"))
        c.write_parquet(out / f.name); n += c.height
        # direct features depend only on the (record, S1) pair -> reuse by semi-join
        x = pl.read_parquet(cache_path("v2train_featx") / f.name, memory_map=False).join(c.select("q_idx", "s1_idx"), on=["q_idx", "s1_idx"], how="semi")
        assert x.height == c.height
        x.write_parquet(xo / f.name)
    note(f"stress {SPLIT}: {n:,} candidate pairs (direct features reused)")
    run.complete("inputs", pairs=n)
run.release()
print("stress inputs ready", mem())
