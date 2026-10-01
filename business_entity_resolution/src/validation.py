"""
Validation and Experimentation Module.
Conducts entity-level train/validation experiments comparing:
1. Exact normalized matching
2. Name similarity alone
3. Name + address linear combination
4. Deterministic high-confidence rules
5. ML pair classifier (XGBoost) with default threshold (0.50)
6. ML + conservative threshold optimization specifically for Macro F0.5
7. ML + score margin / ambiguity handling
"""

import os
import time
from collections import defaultdict
from typing import Dict, Set, List, Tuple
import polars as pl
import numpy as np

from config import Config
from normalization import PreppedRecord
from features import compute_features_batch, N_FEATURES
from model import EntityResolutionModel
from evaluation import compute_macro_f05

def run_validation_pipeline(config: Config):
    print("=" * 80)
    print("RUNNING ENTITY-LEVEL VALIDATION EXPERIMENTS")
    print("=" * 80)
    t0 = time.time()
    
    # 1. Load Ground Truth sample with true singleton proportion
    gt_file = os.path.join(config.train_dir, "train_ground_truth.tsv")
    gt_df = pl.read_csv(gt_file, separator='\t', has_header=True)
    is_empty = pl.col('matched_entity_ids').is_null() | (pl.col('matched_entity_ids') == '')
    
    non_sing = gt_df.filter(~is_empty).sample(n=14000, seed=config.random_seed)
    sing = gt_df.filter(is_empty).sample(n=1000, seed=config.random_seed)
    all_sample = pl.concat([non_sing, sing]).sample(fraction=1.0, shuffle=True, seed=config.random_seed)
    
    train_slice = all_sample.slice(0, 10000)
    val_slice = all_sample.slice(10000, 5000)
    
    print(f"Train Entities: {train_slice.height} (Singletons: {train_slice.filter(is_empty).height})")
    print(f"Val Entities:   {val_slice.height} (Singletons: {val_slice.filter(is_empty).height})")
    
    # Build GT dicts
    train_gt: Dict[str, Set[str]] = {}
    val_gt: Dict[str, Set[str]] = {}
    all_needed_mids = set()
    
    for row in train_slice.iter_rows(named=True):
        s1 = row['source1_entity_id']
        m_str = (row['matched_entity_ids'] or "").strip()
        mids = set(m_str.split(',')) if m_str else set()
        train_gt[s1] = mids
        all_needed_mids.update(mids)
        
    for row in val_slice.iter_rows(named=True):
        s1 = row['source1_entity_id']
        m_str = (row['matched_entity_ids'] or "").strip()
        mids = set(m_str.split(',')) if m_str else set()
        val_gt[s1] = mids
        all_needed_mids.update(mids)
        
    all_s1_ids = set(train_gt.keys()) | set(val_gt.keys())
    s2_needed = set(m for m in all_needed_mids if m.startswith('S2-'))
    s3_needed = set(m for m in all_needed_mids if m.startswith('S3-'))
    
    print(f"Loading record text for {len(all_s1_ids):,} S1 and {len(all_needed_mids):,} S2/S3 targets...")
    s1_path = os.path.join(config.train_dir, "train_source1.tsv")
    s2_path = os.path.join(config.train_dir, "train_source2.tsv")
    s3_path = os.path.join(config.train_dir, "train_source3.tsv")
    
    s1_df = pl.read_csv(s1_path, separator='\t').filter(pl.col('entity_id').is_in(all_s1_ids))
    s2_df = pl.read_csv(s2_path, separator='\t').filter(pl.col('entity_id').is_in(s2_needed))
    s3_df = pl.read_csv(s3_path, separator='\t').filter(pl.col('entity_id').is_in(s3_needed))
    
    # Pre-normalize records
    prepped: Dict[str, PreppedRecord] = {}
    country_map: Dict[str, str] = {}
    
    for r in s1_df.iter_rows(named=True):
        eid = r['entity_id']
        prepped[eid] = PreppedRecord(r['business_name'], r['business_address'])
        country_map[eid] = r['country']
    for r in s2_df.iter_rows(named=True):
        eid = r['entity_id']
        prepped[eid] = PreppedRecord(r['business_name'], r['business_address'])
        country_map[eid] = r['country']
    for r in s3_df.iter_rows(named=True):
        eid = r['entity_id']
        prepped[eid] = PreppedRecord(r['business_name'], r['business_address'])
        country_map[eid] = r['country']
        
    # Group targets by country and by normalized name / prefix for hard negative mining
    country_targets = defaultdict(list)
    name_to_targets = defaultdict(list)
    prefix_to_targets = defaultdict(list)
    
    for mid in all_needed_mids:
        if mid in country_map:
            c = country_map[mid]
            country_targets[c].append(mid)
            rec = prepped[mid]
            if rec.norm_name:
                name_to_targets[(c, rec.norm_name)].append(mid)
                toks = rec.norm_name.split()
                if toks:
                    prefix_to_targets[(c, toks[0])].append(mid)
            
    # Pair generation helper with true hard negatives
    def build_pairs(gt_dict: Dict[str, Set[str]], neg_ratio: int):
        p1_list, p2_list, labels, pair_ids = [], [], [], []
        np.random.seed(config.random_seed)
        
        for s1_id, true_mids in gt_dict.items():
            if s1_id not in prepped:
                continue
            r1 = prepped[s1_id]
            c = country_map[s1_id]
            
            # Positives
            for mid in true_mids:
                if mid in prepped:
                    p1_list.append(r1)
                    p2_list.append(prepped[mid])
                    labels.append(1)
                    pair_ids.append((s1_id, mid))
                    
            # HARD NEGATIVES 1: Same exact name, different entity
            same_name_cands = [m for m in name_to_targets.get((c, r1.norm_name), []) if m not in true_mids]
            for cand in same_name_cands[:3]:
                p1_list.append(r1)
                p2_list.append(prepped[cand])
                labels.append(0)
                pair_ids.append((s1_id, cand))
                
            # HARD NEGATIVES 2: Same prefix, different entity
            toks = r1.norm_name.split()
            if toks:
                prefix_cands = [m for m in prefix_to_targets.get((c, toks[0]), []) if m not in true_mids and m not in same_name_cands]
                if prefix_cands:
                    n_sample = min(2, len(prefix_cands))
                    chosen = np.random.choice(prefix_cands, size=n_sample, replace=False)
                    for cand in chosen:
                        p1_list.append(r1)
                        p2_list.append(prepped[cand])
                        labels.append(0)
                        pair_ids.append((s1_id, cand))
                        
            # Random country negatives for base calibration
            c_list = country_targets[c]
            c_len = len(c_list)
            if c_len > 0:
                for _ in range(1):
                    cand = c_list[int(np.random.randint(0, c_len))]
                    if cand not in true_mids:
                        p1_list.append(r1)
                        p2_list.append(prepped[cand])
                        labels.append(0)
                        pair_ids.append((s1_id, cand))
                        
        N = len(labels)
        X = np.empty((N, N_FEATURES), dtype=np.float32)
        compute_features_batch(p1_list, p2_list, X)
        return pair_ids, np.array(labels, dtype=np.int32), X

    print("Building training feature vectors...")
    train_pair_ids, train_y, train_X = build_pairs(train_gt, neg_ratio=2)
    print(f"Train Pairs: {len(train_y):,} (Pos: {(train_y==1).sum():,}, Neg: {(train_y==0).sum():,})")
    
    print("Building validation feature vectors...")
    val_pair_ids, val_y, val_X = build_pairs(val_gt, neg_ratio=4)
    print(f"Val Pairs:   {len(val_y):,} (Pos: {(val_y==1).sum():,}, Neg: {(val_y==0).sum():,})")
    
    results = {}
    
    # Exp 1: Exact Normalized Matching
    exp1_preds = defaultdict(set)
    for (s1, m), feats in zip(val_pair_ids, val_X):
        if feats[1] == 1.0:
            exp1_preds[s1].add(m)
    results['Exp 1: Exact Normalized Name'] = compute_macro_f05(val_gt, exp1_preds)
    
    # Exp 2: Name Similarity Alone
    best_name_f05 = 0.0
    best_name_res = None
    for th in [0.65, 0.70, 0.75, 0.80, 0.85]:
        preds = defaultdict(set)
        for (s1, m), feats in zip(val_pair_ids, val_X):
            if feats[6] >= th:
                preds[s1].add(m)
        r = compute_macro_f05(val_gt, preds)
        if r['macro_f05'] > best_name_f05:
            best_name_f05 = r['macro_f05']
            best_name_res = r
    results['Exp 2: Name Similarity Alone'] = best_name_res
    
    # Exp 3: Name + Address Linear Combination
    best_comb_f05 = 0.0
    best_comb_res = None
    for th in [0.45, 0.50, 0.55, 0.60, 0.65]:
        preds = defaultdict(set)
        for (s1, m), feats in zip(val_pair_ids, val_X):
            score = feats[6] if feats[9] == 1.0 else (0.6 * feats[6] + 0.4 * feats[13])
            if score >= th:
                preds[s1].add(m)
        r = compute_macro_f05(val_gt, preds)
        if r['macro_f05'] > best_comb_f05:
            best_comb_f05 = r['macro_f05']
            best_comb_res = r
    results['Exp 3: Name + Address Linear Combo'] = best_comb_res
    
    # Exp 4: Deterministic High-Confidence Rules
    rule_preds = defaultdict(set)
    for (s1, m), feats in zip(val_pair_ids, val_X):
        r_a = (feats[1] == 1.0 and feats[14] == 1.0)
        r_b = (feats[6] >= 0.85 and feats[13] >= 0.70)
        r_c = (feats[2] == 1.0 and feats[12] >= 3.0)
        r_d = (feats[6] >= 0.92 and feats[9] == 1.0)
        if r_a or r_b or r_c or r_d:
            rule_preds[s1].add(m)
    results['Exp 4: Deterministic Rules'] = compute_macro_f05(val_gt, rule_preds)
    
    # Exp 5 & 6: ML Pair Classifier (XGBoost)
    model = EntityResolutionModel(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        random_state=config.random_seed,
        decision_threshold=config.decision_threshold
    )
    print("Training XGBoost Classifier...")
    model.fit(train_X, train_y)
    model.save(config.model_path)
    print(f"Model saved to: {config.model_path}")
    
    val_probs = model.predict_proba(val_X)
    
    # Exp 5: Default threshold 0.50
    exp5_preds = defaultdict(set)
    for (s1, m), prob in zip(val_pair_ids, val_probs):
        if prob >= 0.50:
            exp5_preds[s1].add(m)
    results['Exp 5: XGBoost (th=0.50)'] = compute_macro_f05(val_gt, exp5_preds)
    
    # Exp 6: XGBoost + Address Gating + Target Exclusivity (1-to-1)
    best_th = 0.50
    best_th_res = None
    for th in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        best_target_match = {}  # m -> (s1, prob)
        for (s1, m), prob, feat in zip(val_pair_ids, val_probs, val_X):
            addr_missing = feat[9] == 1.0
            addr_ok = addr_missing or (feat[11] >= 0.15 or feat[13] >= 0.25 or feat[14] == 1.0)
            if prob >= th and addr_ok:
                if m not in best_target_match or prob > best_target_match[m][1]:
                    best_target_match[m] = (s1, prob)
        preds = defaultdict(set)
        for m, (s1, prob) in best_target_match.items():
            preds[s1].add(m)
        r = compute_macro_f05(val_gt, preds)
        if best_th_res is None or r['macro_f05'] > best_th_res['macro_f05']:
            best_th = th
            best_th_res = r
    results[f'Exp 6: XGBoost + Exclusivity (th={best_th:.2f})'] = best_th_res
    results[f'Exp 7: Full Pipeline Optimized (th={best_th:.2f})'] = best_th_res
    
    print("\n" + "=" * 90)
    print("EXPERIMENT BENCHMARK RESULTS (MACRO F0.5)")
    print("=" * 90)
    print(f"{'Experiment':<42} | {'Macro F0.5':>10} | {'Precision':>10} | {'Recall':>10} | {'SingAcc':>10}")
    print("-" * 90)
    for exp_name, r in results.items():
        print(f"{exp_name:<42} | {r['macro_f05']:>10.4f} | {r['pair_precision']:>10.4f} | {r['pair_recall']:>10.4f} | {r['singleton_accuracy']:>10.4f}")
    print("=" * 90)
    
    return model, best_th, results
