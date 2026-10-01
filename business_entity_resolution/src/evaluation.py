"""
Evaluation and Metrics Module.
Implements the exact challenge evaluation metric: Macro-Averaged F0.5 per Source 1 entity (including singletons).
Also computes candidate recall, candidate reduction ratio, and pair-level precision/recall.
"""

from typing import Dict, Set, List, Tuple

def compute_macro_f05(
    y_true: Dict[str, Set[str]],
    y_pred: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Computes macro F0.5 score across all Source 1 entities in y_true.
    Includes singletons:
    - true={}, pred={} => 1.0
    - true={}, pred!={} => 0.0
    - true!={}, pred={} => 0.0
    - otherwise => standard F0.5 formula
    """
    total_f05 = 0.0
    n_entities = len(y_true)
    
    singleton_total = 0
    singleton_correct = 0
    
    total_true_pairs = 0
    total_pred_pairs = 0
    total_tp = 0
    
    for s1_id, true_set in y_true.items():
        pred_set = y_pred.get(s1_id, set())
        
        total_true_pairs += len(true_set)
        total_pred_pairs += len(pred_set)
        tp = len(true_set & pred_set)
        total_tp += tp
        
        # Singleton handling
        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
                total_f05 += 1.0
            else:
                total_f05 += 0.0
        else:
            if not pred_set:
                total_f05 += 0.0
            else:
                if tp == 0:
                    total_f05 += 0.0
                else:
                    p = tp / len(pred_set)
                    r = tp / len(true_set)
                    f05 = (1.25 * p * r) / (0.25 * p + r)
                    total_f05 += f05
                    
    macro_f05 = total_f05 / max(1, n_entities)
    singleton_accuracy = singleton_correct / max(1, singleton_total) if singleton_total > 0 else 1.0
    pair_precision = total_tp / max(1, total_pred_pairs)
    pair_recall = total_tp / max(1, total_true_pairs)
    
    return {
        'macro_f05': macro_f05,
        'singleton_accuracy': singleton_accuracy,
        'pair_precision': pair_precision,
        'pair_recall': pair_recall,
        'total_predicted_pairs': total_pred_pairs,
        'true_positive_pairs': total_tp,
        'total_true_pairs': total_true_pairs,
        'total_entities': n_entities,
        'singleton_count': singleton_total
    }

def compute_blocking_metrics(
    y_true: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    total_s2_s3_records: int
) -> Dict[str, float]:
    """
    Computes candidate recall and reduction ratio for candidate generation stage.
    """
    total_true_links = 0
    recalled_links = 0
    total_candidates = 0
    
    for s1_id, true_set in y_true.items():
        total_true_links += len(true_set)
        cand_set = candidates.get(s1_id, set())
        total_candidates += len(cand_set)
        recalled_links += len(true_set & cand_set)
        
    candidate_recall = recalled_links / max(1, total_true_links)
    n_s1 = max(1, len(y_true))
    all_possible_pairs = n_s1 * total_s2_s3_records
    reduction_ratio = 1.0 - (total_candidates / max(1, all_possible_pairs))
    avg_candidates = total_candidates / n_s1
    
    return {
        'candidate_recall': candidate_recall,
        'candidate_reduction_ratio': reduction_ratio,
        'total_candidates_generated': total_candidates,
        'avg_candidates_per_s1': avg_candidates
    }
