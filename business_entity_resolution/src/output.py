"""
Output Generation and Sanitization Module.
Writes sanitized, submission-compliant TSV files for:
1. matching_results.tsv  (scored on leaderboard)
2. candidate_pairs.tsv   (blocking candidate set)

Applies rigorous sanitization:
- Exact tab separation
- S2/S3 ID format enforcement (no S1 self-matches)
- Deduping within ID lists
- Deterministic ID sorting
- Verification that every prediction is a subset of candidates
- Ensuring every test Source 1 entity appears exactly once
- Blank fields for singletons (never 'None' or 'NaN')
"""

import os
from typing import Dict, Set, List, Optional

def sanitize_and_write_outputs(
    s1_all_ids: List[str],
    candidates: Dict[str, Set[str]],
    matches: Dict[str, Set[str]],
    matching_output_path: str,
    candidate_output_path: str
):
    """
    Sanitizes candidate and match mappings and writes official TSV files.
    """
    os.makedirs(os.path.dirname(os.path.abspath(matching_output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(candidate_output_path)), exist_ok=True)
    
    print(f"Writing matching results to: {matching_output_path}")
    print(f"Writing candidate pairs to:   {candidate_output_path}")
    
    with open(matching_output_path, 'w', encoding='utf-8', newline='\n') as f_match, \
         open(candidate_output_path, 'w', encoding='utf-8', newline='\n') as f_cand:
        
        # Headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for s1_id in s1_all_ids:
            # 1. Sanitize Candidates
            raw_cands = candidates.get(s1_id, set())
            valid_cands = set()
            for cid in raw_cands:
                if cid.startswith(("S2-", "S3-")) and cid != s1_id:
                    valid_cands.add(cid)
                    
            # 2. Sanitize Matches
            raw_matches = matches.get(s1_id, set())
            valid_matches = set()
            for mid in raw_matches:
                # Must be valid prefix, not S1, and MUST BE IN CANDIDATES
                if mid.startswith(("S2-", "S3-")) and mid != s1_id and mid in valid_cands:
                    valid_matches.add(mid)
                    
            cand_str = ",".join(sorted(valid_cands)) if valid_cands else ""
            match_str = ",".join(sorted(valid_matches)) if valid_matches else ""
            
            f_cand.write(f"{s1_id}\t{cand_str}\n")
            f_match.write(f"{s1_id}\t{match_str}\n")
            
    print(f"Successfully generated sanitized output files ({len(s1_all_ids):,} entities written).")
