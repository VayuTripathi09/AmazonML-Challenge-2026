"""
Production Test Inference Pipeline Module (v2).
Executes high-recall, precision-calibrated entity resolution over the complete 1.73M test dataset:
- Country-by-country partition processing (France, India, US)
- Safe Unicode accent removal & Indic transliteration (Learned dictionary + Unidecode)
- Prioritized 2-tier multi-strategy candidate blocker
- Fair S2 / S3 quota allocation (up to 60 S2, up to 60 S3 candidates per S1)
- Vectorized 51-feature batch computation directly in NumPy float32
- Scored with production LightGBM model trained on 521k hard-negative pairs
- Dual source thresholds (S2=0.85, S3=0.80) with fatal conflict pruning
- Global 1-to-1 Target Exclusivity across all target records
- Expected F0.5 entity decoding & singleton handling
- Streams official validated matching_results.tsv and candidate_pairs.tsv directly to disk
"""

import os
import gc
import time
from collections import defaultdict
from typing import Dict, Set, List, Tuple
import polars as pl
import numpy as np

from config import Config
from normalization import CompactRecord
from features import compute_features_batch, N_FEATURES
from blocking import CandidateBlocker
from model import EntityResolutionModel

def run_test_inference(config: Config):
    print("=" * 80)
    print("STARTING PRODUCTION TEST INFERENCE PIPELINE (v2)")
    print("=" * 80)
    t_start = time.time()
    
    test_s1_path = os.path.join(config.test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(config.test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(config.test_dir, "test_source3.tsv")
    
    # 1. Discover countries from test S1
    print(f"Reading test S1 metadata from {test_s1_path}...")
    meta_df = pl.read_csv(test_s1_path, separator='\t', columns=['entity_id', 'country'])
    total_test_s1 = meta_df.height
    discovered_countries = sorted(meta_df['country'].unique().to_list())
    print(f"Total Test S1 Entities: {total_test_s1:,}")
    print(f"Discovered Countries:   {discovered_countries}")
    del meta_df
    
    # 2. Load trained production model
    model_path = os.path.join(config.model_dir, "production_lightgbm.joblib")
    if not os.path.isfile(model_path):
        model_path = os.path.join(config.model_dir, "er_model_bundle.joblib")
    print(f"Loading production model from {model_path}...")
    model = EntityResolutionModel()
    model.load(model_path)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Open output TSV files
    f_match = open(config.matching_output_path, 'w', encoding='utf-8', newline='\n')
    f_cand = open(config.candidate_output_path, 'w', encoding='utf-8', newline='\n')
    
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
    total_candidates_count = 0
    total_matches_count = 0
    total_singletons_count = 0
    total_entities_written = 0
    
    # Dual source thresholds
    TH_S2 = 0.85
    TH_S3 = 0.80
    BUFFER_SIZE = 25000
    
    MAX_CANDS_PER_SRC = 60
    MAX_ADDR_CANDS = 20
    
    for country in discovered_countries:
        print("\n" + "=" * 70)
        print(f"PROCESSING COUNTRY PARTITION: {country}")
        print("=" * 70)
        t_country = time.time()
        
        # Load S1 entities for this country
        print(f"Loading test S1 records for {country}...")
        country_s1_df = pl.read_csv(test_s1_path, separator='\t').filter(pl.col('country') == country)
        n_country_s1 = country_s1_df.height
        country_s1_ids = country_s1_df['entity_id'].to_list()
        print(f"  Total S1 entities for {country}: {n_country_s1:,}")
        
        # Pre-load S2 targets for this country
        print(f"Loading test S2 records for {country}...")
        t_s2 = time.time()
        s2_country_df = pl.read_csv(test_s2_path, separator='\t').filter(pl.col('country') == country)
        s2_ids = s2_country_df['entity_id'].to_list()
        s2_names = s2_country_df['business_name'].to_list()
        s2_addrs = s2_country_df['business_address'].to_list()
        del s2_country_df
        print(f"  Loaded {len(s2_ids):,} S2 entities in {time.time()-t_s2:.2f}s.")
        
        # Pre-load S3 targets for this country
        print(f"Loading test S3 records for {country}...")
        t_s3 = time.time()
        s3_country_df = pl.read_csv(test_s3_path, separator='\t').filter(pl.col('country') == country)
        s3_ids = s3_country_df['entity_id'].to_list()
        s3_names = s3_country_df['business_name'].to_list()
        s3_addrs = s3_country_df['business_address'].to_list()
        del s3_country_df
        print(f"  Loaded {len(s3_ids):,} S3 entities in {time.time()-t_s3:.2f}s.")
        
        # Build in-memory multi-strategy index on S1
        print(f"Indexing {n_country_s1:,} S1 entities for {country}...")
        t_idx = time.time()
        blocker = CandidateBlocker(max_bucket_size=config.max_candidates_per_key, max_candidates_per_s1=60)
        blocker.build_s1_index(country_s1_df)
        del country_s1_df
        print(f"  Indexed in {time.time()-t_idx:.2f}s.")
        
        country_candidates: Dict[str, Set[str]] = defaultdict(set)
        # 1-to-1 Target Exclusivity map: eid -> (best_s1_id, best_prob)
        best_target_match: Dict[str, Tuple[str, float]] = {}
        
        p1_buffer = []
        p2_buffer = []
        ref_buffer = []
        tids_buffer = []
        
        def flush_buffer():
            if not p1_buffer:
                return
            n_buf = len(p1_buffer)
            buf_X = np.empty((n_buf, N_FEATURES), dtype=np.float32)
            compute_features_batch(p1_buffer, p2_buffer, buf_X, tids_buffer)
            probs = model.predict_proba_raw(buf_X)
            
            for (s1_id, eid), prob, feat in zip(ref_buffer, probs, buf_X):
                # Fatal conflict protection: reject if both street number AND postal code explicitly conflict
                if feat[40] == 1.0:
                    continue
                req_th = TH_S2 if eid.startswith("S2-") else TH_S3
                if prob >= req_th:
                    # 1-to-1 Target Exclusivity: Assign target eid to the single highest-scoring S1 entity
                    if eid not in best_target_match or prob > best_target_match[eid][1]:
                        best_target_match[eid] = (s1_id, prob)
                        
            p1_buffer.clear()
            p2_buffer.clear()
            ref_buffer.clear()
            tids_buffer.clear()
            
        # Scan S2 targets against S1 index
        print(f"Scanning S2 targets ({len(s2_ids):,} records)...")
        t_scan2 = time.time()
        s2_cands_per_s1 = defaultdict(set)
        s2_addr_cands_per_s1 = defaultdict(set)
        
        for eid, name, addr in zip(s2_ids, s2_names, s2_addrs):
            trec = CompactRecord(name, addr, country)
            high_m, addr_m = blocker.find_matches_for_target(trec)
            
            for s1_id in high_m:
                c_set = s2_cands_per_s1[s1_id]
                if len(c_set) < MAX_CANDS_PER_SRC and eid not in c_set:
                    c_set.add(eid)
                    country_candidates[s1_id].add(eid)
                    p1_buffer.append(blocker.s1_prepped[s1_id])
                    p2_buffer.append(trec)
                    ref_buffer.append((s1_id, eid))
                    tids_buffer.append(eid)
                    if len(p1_buffer) >= BUFFER_SIZE:
                        flush_buffer()
                        
            for s1_id in addr_m:
                a_set = s2_addr_cands_per_s1[s1_id]
                if len(a_set) < MAX_ADDR_CANDS and eid not in country_candidates[s1_id]:
                    a_set.add(eid)
                    country_candidates[s1_id].add(eid)
                    p1_buffer.append(blocker.s1_prepped[s1_id])
                    p2_buffer.append(trec)
                    ref_buffer.append((s1_id, eid))
                    tids_buffer.append(eid)
                    if len(p1_buffer) >= BUFFER_SIZE:
                        flush_buffer()
                        
        flush_buffer()
        del s2_cands_per_s1, s2_addr_cands_per_s1
        print(f"  Scanned S2 in {time.time()-t_scan2:.2f}s.")
        
        # Scan S3 targets against S1 index
        print(f"Scanning S3 targets ({len(s3_ids):,} records)...")
        t_scan3 = time.time()
        s3_cands_per_s1 = defaultdict(set)
        s3_addr_cands_per_s1 = defaultdict(set)
        
        for eid, name, addr in zip(s3_ids, s3_names, s3_addrs):
            trec = CompactRecord(name, addr, country)
            high_m, addr_m = blocker.find_matches_for_target(trec)
            
            for s1_id in high_m:
                c_set = s3_cands_per_s1[s1_id]
                if len(c_set) < MAX_CANDS_PER_SRC and eid not in c_set:
                    c_set.add(eid)
                    country_candidates[s1_id].add(eid)
                    p1_buffer.append(blocker.s1_prepped[s1_id])
                    p2_buffer.append(trec)
                    ref_buffer.append((s1_id, eid))
                    tids_buffer.append(eid)
                    if len(p1_buffer) >= BUFFER_SIZE:
                        flush_buffer()
                        
            for s1_id in addr_m:
                a_set = s3_addr_cands_per_s1[s1_id]
                if len(a_set) < MAX_ADDR_CANDS and eid not in country_candidates[s1_id]:
                    a_set.add(eid)
                    country_candidates[s1_id].add(eid)
                    p1_buffer.append(blocker.s1_prepped[s1_id])
                    p2_buffer.append(trec)
                    ref_buffer.append((s1_id, eid))
                    tids_buffer.append(eid)
                    if len(p1_buffer) >= BUFFER_SIZE:
                        flush_buffer()
                        
        flush_buffer()
        del s3_cands_per_s1, s3_addr_cands_per_s1
        print(f"  Scanned S3 in {time.time()-t_scan3:.2f}s.")
        
        # Assemble 1-to-1 matches per S1 entity with Expected F0.5 decoding
        print(f"Assembling exclusive matches for {country} ({len(best_target_match):,} target links)...")
        country_s1_matches: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for eid, (s1_id, prob) in best_target_match.items():
            country_s1_matches[s1_id].append((eid, prob))
            
        country_cand_count = 0
        country_match_count = 0
        country_sing_count = 0
        
        # Stream results directly to disk
        for s1_id in country_s1_ids:
            raw_cands = country_candidates.get(s1_id, set())
            valid_cands = set(c for c in raw_cands if c.startswith(("S2-", "S3-")) and c != s1_id)
            
            # Expected F0.5 subset decoding for this entity
            target_list = country_s1_matches.get(s1_id, [])
            valid_matches = set()
            if target_list:
                target_list.sort(key=lambda x: x[1], reverse=True)
                for rank, (eid, prob) in enumerate(target_list, 1):
                    if eid in valid_cands:
                        if rank == 1:
                            valid_matches.add(eid)
                        elif rank <= 3 and prob >= 0.70:
                            valid_matches.add(eid)
                        elif rank > 3 and prob >= 0.78:
                            valid_matches.add(eid)
                            
            cand_str = ",".join(sorted(valid_cands)) if valid_cands else ""
            match_str = ",".join(sorted(valid_matches)) if valid_matches else ""
            
            f_cand.write(f"{s1_id}\t{cand_str}\n")
            f_match.write(f"{s1_id}\t{match_str}\n")
            
            country_cand_count += len(valid_cands)
            country_match_count += len(valid_matches)
            if not valid_matches:
                country_sing_count += 1
            total_entities_written += 1
            
        f_cand.flush()
        f_match.flush()
        
        total_candidates_count += country_cand_count
        total_matches_count += country_match_count
        total_singletons_count += country_sing_count
        
        print(f"Country {country} partition fully processed in {time.time()-t_country:.2f}s:")
        print(f"  Candidates: {country_cand_count:,} (avg {country_cand_count/n_country_s1:.2f}/entity)")
        print(f"  Matches:    {country_match_count:,} (avg {country_match_count/n_country_s1:.2f}/entity)")
        print(f"  Singletons: {country_sing_count:,} ({country_sing_count/n_country_s1*100:.2f}%)")
        
        # Cleanup country memory
        country_candidates.clear()
        best_target_match.clear()
        country_s1_matches.clear()
        blocker.s1_prepped.clear()
        for idx in blocker.indexes.values():
            idx.clear()
        del s2_ids, s2_names, s2_addrs
        del s3_ids, s3_names, s3_addrs
        gc.collect()
        
    f_match.close()
    f_cand.close()
    
    t_total = time.time() - t_start
    print("\n" + "=" * 80)
    print("TEST INFERENCE PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Test S1 Entities:     {total_entities_written:,}")
    print(f"Total Candidates Generated: {total_candidates_count:,} (avg {total_candidates_count/max(1, total_entities_written):.2f}/entity)")
    print(f"Total Matches Predicted:    {total_matches_count:,} (avg {total_matches_count/max(1, total_entities_written):.2f}/entity)")
    print(f"Singleton Predictions:      {total_singletons_count:,} ({total_singletons_count/max(1, total_entities_written)*100:.2f}%)")
    print(f"Thresholds:                 S2={TH_S2}, S3={TH_S3}")
    print(f"Total Inference Time:       {t_total:.2f} seconds ({t_total/60:.2f} minutes)")
    print(f"Matching Results Path:      {config.matching_output_path}")
    print(f"Candidate Pairs Path:       {config.candidate_output_path}")
    print("=" * 80)

if __name__ == '__main__':
    from config import parse_args
    cfg = Config()
    run_test_inference(cfg)
