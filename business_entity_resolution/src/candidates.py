"""
Step 2: candidate generation for a split (train or test).

Writes cache/<split>_cands/<country>_<chunk>.parquet with columns
  q_idx   : row in the concatenated query table (Source 2 rows, then Source 3 rows)
  s1_idx  : row in the Source 1 table
  cos_name, cos_addr, cos_comb, rank (rank of this S1 among the query's candidates)
"""
import sys
import time

import numpy as np
import polars as pl
import torch

from config import cache_path
from retrieval import CountryIndex

K_APPROX = 40   # GPU approximate neighbours per query
K_KEEP = 10     # exact re-scored neighbours kept per query
CHUNK = 200_000


def scan_tables(split: str):
    """Lazy frames; materialised one country at a time to bound memory."""
    cols = ["country", "name_core", "addr_norm"]
    s1 = pl.scan_parquet(cache_path(f"{split}_source1_norm.parquet")).select(["entity_id"] + cols).with_row_index("s1_idx")
    q = pl.concat([pl.scan_parquet(cache_path(f"{split}_source2_norm.parquet")).select(cols),
                   pl.scan_parquet(cache_path(f"{split}_source3_norm.parquet")).select(cols)]).with_row_index("q_idx")
    return s1, q


def generate(split: str):
    out_dir = cache_path(f"{split}_cands")
    out_dir.mkdir(parents=True, exist_ok=True)  # resumable: finished chunks are skipped
    s1, q = scan_tables(split)
    countries = sorted(s1.select(pl.col("country").unique()).collect()["country"].to_list())
    q_countries = set(q.select(pl.col("country").unique()).collect()["country"].to_list())
    orphan = q_countries - set(countries)
    if orphan:
        print(f"note: query countries without any Source 1 record (no candidates possible): {orphan}")
    total = 0
    for country in countries:
        t0 = time.time()
        s1c = s1.filter(pl.col("country") == country).collect()
        qc = q.filter(pl.col("country") == country).collect()
        if qc.height == 0:
            continue
        done_marker = out_dir / f"{country}.done"
        if done_marker.exists():
            print(f"[{split}/{country}] done already, skipping")
            continue
        idx = CountryIndex(s1c)
        s1_map = s1c["s1_idx"].to_numpy()
        print(f"[{split}/{country}] index {s1c.height:,} S1 built in {time.time()-t0:.0f}s; {qc.height:,} queries", flush=True)
        for ci, s in enumerate(range(0, qc.height, CHUNK)):
            out_file = out_dir / f"{country}_{s:09d}.parquet"  # named by query offset -> resumable
            if out_file.exists():
                continue
            part = qc.slice(s, CHUNK)
            Qn, Qa = idx.encode(part)
            cand = idx.candidates(Qn, Qa, K=K_APPROX, k=K_KEEP)
            q_map = part["q_idx"].to_numpy()
            cand = cand.with_columns(
                pl.Series("q_idx", q_map[cand["q_row"].to_numpy()], dtype=pl.UInt32),
                pl.Series("s1_idx", s1_map[cand["s1_row"].to_numpy()], dtype=pl.UInt32),
            ).select("q_idx", "s1_idx", "cos_name", "cos_addr", "cos_comb", "rank")
            cand.write_parquet(out_file.with_suffix(".tmp"))
            out_file.with_suffix(".tmp").replace(out_file)
            total += cand.height
            print(f"  chunk {ci}: {part.height:,} queries -> {cand.height:,} pairs ({time.time()-t0:.0f}s)", flush=True)
        done_marker.touch()
        del idx
        torch.cuda.empty_cache()
    print(f"[{split}] total candidate pairs: {total:,}")


def patch_queries(split: str, flag: str = "has_indic"):
    """Re-generate the candidates of the queries whose `flag` is set (e.g. after their names were
    re-normalized) and splice them into the existing chunk files. The S1 index is unchanged, so this
    is equivalent to a full regeneration."""
    out_dir = cache_path(f"{split}_cands")
    marker = out_dir / f"patched_{flag}.done"
    if marker.exists():
        print(f"[{split}] {flag} patch already applied")
        return
    s1, q = scan_tables(split)
    flags = pl.concat([pl.scan_parquet(cache_path(f"{split}_{s}_norm.parquet")).select(flag)
                       for s in ("source2", "source3")]).with_row_index("q_idx")
    q = q.join(flags, on="q_idx").filter(pl.col(flag))
    countries = q.select(pl.col("country").unique()).collect()["country"].to_list()
    for country in countries:
        t0 = time.time()
        qc = q.filter(pl.col("country") == country).collect()
        s1c = s1.filter(pl.col("country") == country).collect()
        if s1c.height == 0:
            continue
        idx = CountryIndex(s1c)
        s1_map = s1c["s1_idx"].to_numpy()
        parts = []
        for s in range(0, qc.height, CHUNK):
            part = qc.slice(s, CHUNK)
            Qn, Qa = idx.encode(part)
            cand = idx.candidates(Qn, Qa, K=K_APPROX, k=K_KEEP)
            parts.append(cand.with_columns(
                pl.Series("q_idx", part["q_idx"].to_numpy()[cand["q_row"].to_numpy()], dtype=pl.UInt32),
                pl.Series("s1_idx", s1_map[cand["s1_row"].to_numpy()], dtype=pl.UInt32),
            ).select("q_idx", "s1_idx", "cos_name", "cos_addr", "cos_comb", "rank"))
        new = pl.concat(parts)
        patched = qc.select(pl.col("q_idx").cast(pl.UInt32))
        for f in sorted(out_dir.glob(f"{country}_*.parquet")):
            if f.name == f"{country}_patch_{flag}.parquet":
                continue  # previous patch output is replaced below
            old = pl.read_parquet(f, memory_map=False)  # no mmap: file may be rewritten in place (Windows)
            keep = old.join(patched, on="q_idx", how="anti")
            if keep.height != old.height:
                keep.write_parquet(f)
        new.write_parquet(out_dir / f"{country}_patch_{flag}.parquet")
        print(f"[{split}/{country}] re-generated candidates for {qc.height:,} {flag} queries ({time.time()-t0:.0f}s)", flush=True)
        del idx
        torch.cuda.empty_cache()
    marker.touch()


if __name__ == "__main__":
    for sp_ in sys.argv[1:] or ["train", "test"]:
        generate(sp_)
