"""
Data Loader Module for Business Entity Resolution Challenge.
Provides memory-efficient streaming and country-partitioned data loading using Polars.
Ensures tab separation (sep='\\t') is strictly enforced on all dataset files.
"""

import os
import polars as pl
from typing import List, Dict, Generator, Optional

def discover_countries(source1_path: str) -> List[str]:
    """Dynamically discovers all unique country labels from Source 1."""
    df = pl.read_csv(source1_path, separator='\t', columns=['country'])
    countries = df['country'].unique().drop_nulls().to_list()
    return sorted(countries)

def load_source_partition(
    file_path: str,
    country: Optional[str] = None,
    columns: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Loads a specific source file, optionally filtering by country.
    Uses lazy scanning to push down predicates for low memory usage.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    lazy_df = pl.scan_csv(
        file_path,
        separator='\t',
        has_header=True,
        truncate_ragged_lines=True
    )
    
    if country is not None:
        lazy_df = lazy_df.filter(pl.col('country') == country)
        
    if columns is not None:
        lazy_df = lazy_df.select(columns)
        
    return lazy_df.collect()

def load_ground_truth(gt_path: str) -> Dict[str, List[str]]:
    """
    Loads the ground truth file into a mapping: {source1_id: [matched_ids]}.
    Preserves empty lists for singletons.
    """
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
    df = pl.read_csv(gt_path, separator='\t', has_header=True, truncate_ragged_lines=True)
    mapping = {}
    for row in df.iter_rows(named=True):
        s1 = row['source1_entity_id']
        m_str = (row['matched_entity_ids'] or "").strip()
        mapping[s1] = m_str.split(',') if m_str else []
    return mapping

def stream_source_chunks(
    file_path: str,
    country: Optional[str] = None,
    chunk_size: int = 100000
) -> Generator[pl.DataFrame, None, None]:
    """
    Streams through a source file in chunks to ensure bounded memory.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    reader = pl.read_csv_batched(
        file_path,
        separator='\t',
        has_header=True,
        batch_size=chunk_size,
        truncate_ragged_lines=True
    )
    
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        batch = batches[0]
        if country is not None:
            batch = batch.filter(pl.col('country') == country)
        if batch.height > 0:
            yield batch
