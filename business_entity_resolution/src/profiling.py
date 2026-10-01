"""
Dataset Profiling and EDA Module.
Analyzes row counts, columns, missingness, unique IDs, country distributions, string lengths, and ground truth statistics.
"""

import os
import json
import time
import polars as pl
from config import Config

def profile_source_file(file_path: str):
    name = os.path.basename(file_path)
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"Profiling {name} ({file_size_mb:.2f} MB)...")
    t0 = time.time()
    
    df = pl.read_csv(file_path, separator='\t', has_header=True, infer_schema_length=10000, truncate_ragged_lines=True)
    n_rows = df.height
    n_cols = df.width
    cols = df.columns
    dtypes = {col: str(df.schema[col]) for col in cols}
    
    null_counts = {}
    for col in cols:
        n_null = int(df[col].is_null().sum())
        if df.schema[col] == pl.String:
            n_null += int((df[col] == "").sum())
        null_counts[col] = n_null
        
    missing_rates = {col: round(null_counts[col] / max(1, n_rows), 6) for col in cols}
    unique_ids = df['entity_id'].n_unique()
    dup_ids = n_rows - unique_ids
    dup_rate = round(dup_ids / max(1, n_rows), 6)
    
    country_dist = {
        row['country']: int(row['count'])
        for row in df['country'].value_counts().iter_rows(named=True)
    }
    
    str_stats = {}
    for text_col in ['business_name', 'business_address']:
        if text_col in df.columns:
            lens = df[text_col].fill_null("").str.len_chars()
            str_stats[text_col] = {
                'min': int(lens.min()) if n_rows > 0 else 0,
                'max': int(lens.max()) if n_rows > 0 else 0,
                'avg': round(float(lens.mean()), 2) if n_rows > 0 else 0.0
            }
            
    ram_mb = df.estimated_size() / (1024 * 1024)
    elapsed = time.time() - t0
    
    return {
        'file_name': name,
        'file_size_mb': round(file_size_mb, 2),
        'in_memory_mb': round(ram_mb, 2),
        'total_rows': n_rows,
        'num_columns': n_cols,
        'column_names': cols,
        'data_types': dtypes,
        'unique_ids': unique_ids,
        'duplicate_ids': dup_ids,
        'duplicate_rate': dup_rate,
        'missing_counts': null_counts,
        'missing_rates': missing_rates,
        'country_distribution': country_dist,
        'string_lengths': str_stats,
        'elapsed_seconds': round(elapsed, 2)
    }

def profile_ground_truth(file_path: str):
    name = os.path.basename(file_path)
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"Profiling {name} ({file_size_mb:.2f} MB)...")
    t0 = time.time()
    
    df = pl.read_csv(file_path, separator='\t', has_header=True, truncate_ragged_lines=True)
    n_rows = df.height
    n_cols = df.width
    cols = df.columns
    dtypes = {col: str(df.schema[col]) for col in cols}
    
    unique_s1 = df['source1_entity_id'].n_unique()
    dup_s1 = n_rows - unique_s1
    
    match_lens_series = df.select(
        pl.when(pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == ""))
        .then(0)
        .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1)
        .alias("m_len")
    )["m_len"]
    
    total_matches = int(match_lens_series.sum())
    singleton_count = int((match_lens_series == 0).sum())
    non_singleton_count = n_rows - singleton_count
    
    match_dist = match_lens_series.value_counts().sort("m_len")
    match_dist_dict = {
        str(row['m_len']): int(row['count'])
        for row in match_dist.iter_rows(named=True) if row['m_len'] <= 15
    }
    
    s2_est_count = int(df.select(pl.col("matched_entity_ids").str.count_matches("S2-").sum()).item())
    s3_est_count = int(df.select(pl.col("matched_entity_ids").str.count_matches("S3-").sum()).item())
    
    ram_mb = df.estimated_size() / (1024 * 1024)
    elapsed = time.time() - t0
    
    return {
        'file_name': name,
        'file_size_mb': round(file_size_mb, 2),
        'in_memory_mb': round(ram_mb, 2),
        'total_rows': n_rows,
        'num_columns': n_cols,
        'column_names': cols,
        'data_types': dtypes,
        'unique_s1_ids': unique_s1,
        'duplicate_s1_ids': dup_s1,
        'singleton_count': singleton_count,
        'singleton_rate': round(singleton_count / max(1, n_rows), 4),
        'total_ground_truth_matches': total_matches,
        's2_matches': s2_est_count,
        's3_matches': s3_est_count,
        'avg_matches_per_s1': round(total_matches / max(1, n_rows), 3),
        'avg_matches_per_non_singleton': round(total_matches / max(1, non_singleton_count), 3),
        'match_count_distribution': match_dist_dict,
        'elapsed_seconds': round(elapsed, 2)
    }

def run_profiling(config: Config):
    print("=" * 80)
    print("DATASET PROFILING STARTED")
    print("=" * 80)
    
    profile = {
        'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        'train_sources': {},
        'test_sources': {},
        'ground_truth': None
    }
    
    gt_file = os.path.join(config.train_dir, "train_ground_truth.tsv")
    profile['ground_truth'] = profile_ground_truth(gt_file)
    
    for fn in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
        fp = os.path.join(config.train_dir, fn)
        profile['train_sources'][fn] = profile_source_file(fp)
        
    for fn in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
        fp = os.path.join(config.test_dir, fn)
        profile['test_sources'][fn] = profile_source_file(fp)
        
    with open(config.profile_report_path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2)
        
    print(f"\nProfile report written to: {config.profile_report_path}")
    return profile
