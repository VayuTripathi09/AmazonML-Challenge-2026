"""Step 1: normalize names/addresses of every record (multiprocess) and cache as parquet."""
import os
import sys
import time
from multiprocessing import Pool

import polars as pl

from config import cache_path
from normalization import normalize_address, normalize_name

N_PROC = max(1, (os.cpu_count() or 2) - 1)


def _norm_batch(batch):
    names, addrs = batch
    out = []
    for n, a in zip(names, addrs):
        nn, nc, dom, ind = normalize_name(n)
        an, nums = normalize_address(a)
        out.append((nn, nc, dom, ind, an, nums))
    return out


def normalize_frame(df: pl.DataFrame, pool, batch=20000) -> pl.DataFrame:
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    batches = [(names[i:i + batch], addrs[i:i + batch]) for i in range(0, len(names), batch)]
    rows = [r for part in pool.imap(_norm_batch, batches) for r in part]
    cols = list(zip(*rows)) if rows else [[]] * 6
    return df.select("entity_id", "country").with_columns(
        pl.Series("name_norm", cols[0], dtype=pl.Utf8),
        pl.Series("name_core", cols[1], dtype=pl.Utf8),
        pl.Series("is_domain", cols[2], dtype=pl.Boolean),
        pl.Series("has_indic", cols[3], dtype=pl.Boolean),
        pl.Series("addr_norm", cols[4], dtype=pl.Utf8),
        pl.Series("addr_nums", cols[5], dtype=pl.Utf8),
        pl.Series("addr_missing", [len(a.strip()) == 0 for a in addrs], dtype=pl.Boolean),
    )


def main(splits=("train", "test")):
    with Pool(N_PROC) as pool:
        for split in splits:
            for src in ("source1", "source2", "source3"):
                out = cache_path(f"{split}_{src}_norm.parquet")
                if out.exists():
                    print(f"skip {out.name} (exists)")
                    continue
                t = time.time()
                df = pl.read_parquet(cache_path(f"{split}_{src}.parquet"))
                normalize_frame(df, pool).write_parquet(out)
                print(f"{split}_{src}: {df.height:,} rows normalized in {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("train", "test"))
