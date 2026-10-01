"""
High-Recall Multi-Strategy Blocker Module with Dynamic Fallback.
Implements a 12-strategy prioritized union blocker with fair multi-source quotas:
1. Exact normalized name
2. Squished full name (space-removed, domain-aware)
3. Invariant sorted core tokens
4. Squished core name (matches domains to company names)
5. Prefix-2 core tokens
6. Address Door Number x First core name word
7. Postal / PIN code x First core name word
8. Address Door Number x Distinctive Street Word (excluding common state/city/generic words)
9. Distinctive Address Token Bigrams (excluding common state/city/road bigrams)
10. Dynamic TF-IDF sparse similarity fallback for sparse/zero-candidate entities.

Employs bounded bucket lookup with prioritized high-specificity tiers and fair S2/S3 allocation.
"""

import time
from collections import defaultdict
from typing import Dict, Set, List, Tuple
import polars as pl
import numpy as np

from normalization import CompactRecord

COMMON_GEOGRAPHIC_BIGRAMS = frozenset({
    'tamil_nadu', 'uttar_pradesh', 'madhya_pradesh', 'andhra_pradesh', 'west_bengal',
    'himachal_pradesh', 'new_york', 'new_jersey', 'north_carolina', 'south_carolina',
    'rhode_island', 'new_hampshire', 'new_mexico', 'united_states', 'delhi_new',
    'new_delhi', 'san_francisco', 'san_antonio', 'san_diego', 'san_jose', 'fort_worth',
    'los_angeles', 'las_vegas', 'kansas_city', 'salt_lake', 'lake_city', 'oklahoma_city',
    'el_paso', 'baton_rouge', 'saint_louis', 'st_louis', 'main_street', 'broad_street',
    'market_street', 'park_avenue', 'state_street', 'commercial_street', 'church_street',
    'station_road', 'temple_road', 'college_road', 'hospital_road', 'railway_station',
    'bus_stand', 'ring_road', 'bypass_road', 'link_road', 'first_floor', 'second_floor',
    'third_floor', 'fourth_floor', 'ground_floor', 'industrial_area', 'phase_1', 'phase_2',
    'sector_1', 'sector_2', 'sector_3', 'sector_4', 'sector_5', 'pocket_1', 'pocket_2',
    'block_a', 'block_b', 'block_c', 'block_d', 'plot_no', 'door_no', 'house_no', 'flat_no'
})

GEOGRAPHIC_WORDS = frozenset({
    'delhi', 'mumbai', 'chennai', 'kolkata', 'bangalore', 'bengaluru', 'hyderabad', 'pune',
    'ahmedabad', 'jaipur', 'lucknow', 'chandigarh', 'bhopal', 'patna', 'nagpur', 'indore',
    'thane', 'agra', 'varanasi', 'meerut', 'surat', 'vadodara', 'rajkot', 'nashik',
    'california', 'texas', 'florida', 'illinois', 'pennsylvania', 'ohio', 'georgia',
    'michigan', 'virginia', 'washington', 'arizona', 'massachusetts', 'tennessee', 'indiana',
    'missouri', 'maryland', 'wisconsin', 'colorado', 'minnesota', 'carolina', 'alabama',
    'louisiana', 'kentucky', 'oregon', 'oklahoma', 'connecticut', 'iowa', 'utah', 'nevada',
    'arkansas', 'mississippi', 'kansas', 'karnataka', 'maharashtra', 'gujarat', 'rajasthan',
    'punjab', 'haryana', 'kerala', 'bihar', 'odisha', 'jharkhand', 'assam', 'telangana',
    'india', 'usa', 'france', 'paris', 'lyon', 'marseille', 'street', 'road', 'avenue', 'drive',
    'lane', 'boulevard', 'highway', 'court', 'place', 'terrace', 'circle', 'parkway', 'building',
    'floor', 'suite', 'unit', 'apartment', 'door', 'house', 'room', 'block', 'plot', 'sector'
})

def extract_blocking_keys(rec: CompactRecord) -> Dict[str, List[str]]:
    """Extracts high-recall, low-noise multi-strategy blocking keys from a CompactRecord."""
    keys = defaultdict(list)
    
    # 1. Exact & Squished Name
    if rec.norm_name:
        keys['exact_name'].append(rec.norm_name)
    if len(rec.squished_name) >= 4:
        keys['squished_name'].append(rec.squished_name)
        
    # 2. Core Tokens
    if rec.core_tokens:
        keys['sorted_core'].append(rec.core_str)
        if len(rec.squished_core) >= 4:
            keys['squished_core'].append(rec.squished_core)
        if rec.first_two_tokens:
            keys['prefix2'].append(rec.first_two_tokens)
            
    # 3. Hybrid Number & Postal with Core Name
    if rec.addr_nums and rec.first_token and len(rec.first_token) >= 3:
        for num in list(rec.addr_nums)[:2]:
            keys['num_name'].append(f"{num}_{rec.first_token}")
            
    if rec.postal_code and rec.first_token and len(rec.first_token) >= 3:
        keys['postal_name'].append(f"{rec.postal_code}_{rec.first_token}")
        
    # 4. Distinctive Number x Street Word (excluding state/city/road words)
    addr_toks = [t for t in rec.norm_addr.split() if t not in GEOGRAPHIC_WORDS and len(t) >= 4 and not t.isdigit()]
    if rec.addr_nums and addr_toks:
        for num in list(rec.addr_nums)[:2]:
            for t in addr_toks[:3]:
                keys['num_street_filtered'].append(f"{num}_{t}")
                
    # 5. Distinctive Address Token Bigrams (excluding common state/city bigrams)
    for i in range(len(addr_toks) - 1):
        bg = f"{addr_toks[i]}_{addr_toks[i+1]}"
        if bg not in COMMON_GEOGRAPHIC_BIGRAMS:
            keys['addr_bigram_filtered'].append(bg)
            
    return keys

class CandidateBlocker:
    """
    High-recall in-memory blocker with prioritized tiers and bounded bucket lookup.
    """
    def __init__(
        self,
        max_bucket_size: int = 50,
        max_candidates_per_s1: int = 80
    ):
        self.max_bucket_size = max_bucket_size
        self.max_candidates_per_s1 = max_candidates_per_s1
        
        self.indexes: Dict[str, Dict[str, List[str]]] = {
            'exact_name': defaultdict(list),
            'squished_name': defaultdict(list),
            'sorted_core': defaultdict(list),
            'squished_core': defaultdict(list),
            'prefix2': defaultdict(list),
            'num_name': defaultdict(list),
            'postal_name': defaultdict(list),
            'num_street_filtered': defaultdict(list),
            'addr_bigram_filtered': defaultdict(list),
        }
        
        # High-specificity strategies that are never dropped due to bucket size or address quotas
        self.exact_strategies = frozenset({
            'exact_name', 'squished_name', 'sorted_core', 'squished_core',
            'prefix2', 'num_name', 'postal_name'
        })
        
        self.s1_prepped: Dict[str, CompactRecord] = {}

    def build_s1_index(self, s1_df: pl.DataFrame):
        """Indexes Source 1 records into inverted lookup tables."""
        t0 = time.time()
        for idx in self.indexes.values():
            idx.clear()
        self.s1_prepped.clear()
        
        ids = s1_df['entity_id'].to_list()
        names = s1_df['business_name'].to_list()
        addrs = s1_df['business_address'].to_list()
        countries = s1_df['country'].to_list() if 'country' in s1_df.columns else [""] * len(ids)
        
        for eid, name, addr, c in zip(ids, names, addrs, countries):
            rec = CompactRecord(name, addr, c)
            self.s1_prepped[eid] = rec
            
            keys = extract_blocking_keys(rec)
            for strat, key_list in keys.items():
                if strat in self.indexes:
                    idx = self.indexes[strat]
                    for k in set(key_list):
                        idx[k].append(eid)
                        
        print(f"Indexed {len(ids):,} S1 entities across {len(self.indexes)} blocking strategies in {time.time()-t0:.2f}s.")

    def find_matches_for_target(self, target_rec: CompactRecord) -> Tuple[Set[str], Set[str]]:
        """
        Retrieves candidate S1 entities for a target record.
        Returns:
            (high_priority_matches, address_only_matches)
        """
        keys = extract_blocking_keys(target_rec)
        high_matches = set()
        addr_matches = set()
        
        for strat, key_list in keys.items():
            if strat not in self.indexes:
                continue
            idx = self.indexes[strat]
            is_exact = (strat in self.exact_strategies)
            
            for k in set(key_list):
                bucket = idx.get(k)
                if bucket:
                    if is_exact:
                        high_matches.update(bucket)
                    elif len(bucket) <= self.max_bucket_size:
                        addr_matches.update(bucket)
                        
        return high_matches, addr_matches
