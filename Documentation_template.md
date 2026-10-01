# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** October 2026

---

## 1. Executive Summary

Each Source 2/3 record is attached to **at most one** Source 1 entity (a property of the training labels). A GPU sparse-TF-IDF retrieval gives every record its 10 most similar Source 1 candidates (98.7% of true links retrieved; candidate-restricted oracle macro F0.5 0.996). A LightGBM matcher then scores the candidates using **only direct evidence between the record and the candidate** (name/address similarity, IDF-weighted token agreement, premise vs unit numbers, raw-text agreement, and competition among the record's own candidates). A per-record second stage re-scores the best candidate, and the record is linked if p ≥ 0.70. Deliberately, no feature depends on *other* S2/S3 records: an earlier variant that used agreement among sibling records scored 0.986 on validation but **0.933 on the leaderboard**, because the test set contains coherent groups of same-name, wrong-address records. On a sealed confirmation fold the final model scores **0.98321 macro F0.5**, against 0.97787 for the original recipe refit on the same data (paired difference +0.0053, 95% CI [+0.0051, +0.0055]).

---

## 2. Methodology

### 2.1 Problem Analysis
* Training data: 2.21M S1 entities, 10.3M S2+S3 records, 7.64M true links. Every matched record links to exactly one S1 entity, links never cross countries, 26% of records match nothing, 5.6% of S1 entities have no match, and matched entities have 3.7 links on average.
* **Name noise:** case, accents, digit-for-letter typos (`Preparat0ry`), word order, legal-suffix changes, `M/s`/`Dr`/`Shri` prefixes, junk prefixes, `doing business as`, `(India)` tags, website/hashtag names, and names in **9 Indic scripts** (~7% of records).
* **Address noise:** missing components (~3–4% empty), reordering, abbreviations, state names vs codes (incl. native script), zero-padded numbers, `<NULL>`/`N/A`.
* **Hard negatives:** near-copies of a real business that differ in one word or one number.
* **Test differs from train:** it adds France, has ~23% more records per S1 entity, and contains many **coherent groups of records agreeing on a same-name, wrong-address business** (likely branches with no S1 entry).

### 2.2 Solution Strategy
**Approach Type:** retrieval + gradient-boosted pair classifier + per-record second stage + one-S1-per-record assignment.
**Core ideas:** (1) match from the record's side (arg-max over its candidates with an abstain option); (2) Indic transliteration plus a dictionary learned from training pairs; (3) GPU retrieval over sparse TF-IDF; (4) **direct evidence only**, with duplication invariance tested explicitly; (5) evaluation that mirrors the test setting (full catalog, all distractors), a stress suite built from real labels, and a sealed confirmation fold.

---

## 3. Candidate Generation (Blocking)

* **Normalization:** one Unicode-offset table transliterates all nine Indic blocks (schwa handling, chillu letters, native digits). A token dictionary (526 entries such as `praivet→private`, `injiniyaring→engineering`) is learned **from TRAIN-fold pairs only**; it came out identical to the dictionary learned on a larger fold. Names are ASCII-folded, legal/stop words removed into a `core` view, websites/hashtags/`dba` handled, digit typos repaired. Addresses are normalized (street/state abbreviations, zero padding, placeholders).
* **Retrieval:** per country (an open label set), query = S2/S3 record, index = S1. Name = character 3-grams of the space-free core name; address = word tokens; score = 0.5·cos_name + 0.5·cos_addr. Vectors are CountSketch-compressed to 1024 dims for a GPU fp16 top-40, then re-scored exactly and cut to the best 10.
* **Blocking keys used:** country partition + TF-IDF(name char-3-grams) + TF-IDF(address tokens).
* **Candidate pairs generated:** 10 per record → 103.2M (train), 99.7M (test) = ~57.5 per test S1 entity. `candidate_pairs.tsv` is exactly the set scored by the model.
* **How true matches were not lost:** 98.7% of true links are retrieved (India 98.3%, US 98.9%; Indic-script records 99.4% after the dictionary). The candidate-restricted oracle macro F0.5 on the development fold is **0.99609**, so retrieval is not what limits the score. The remaining misses are mostly records without an address whose name is shared by many S1 entities.

---

## 4. Matching Model

**Features used (direct evidence only; each depends only on the record, the candidate S1 and S1-catalog statistics):**
- *Retrieval / candidate competition (the record's own 10 candidates):* cosine scores, rank, gap to the best other candidate, margins versus the runner-up.
- *Name:* rapidfuzz ratio / token-set / token-sort / partial on core, space-free and full names; exact flags; token overlap and coverage; lengths; tokens present on one side only and the similarity of those leftovers; **IDF-weighted name agreement** (IDF from the S1 catalog per country, capped), and the rarest token present only in the record or only in the S1 name.
- *Address:* similarity ratios, token overlap/coverage, **IDF-weighted alphabetic token agreement**, rarest contradicting address token, count of rare contradicting tokens.
- *Numbers:* shared house numbers, conflicts, **premise (first) number equality / containment on either side, relative closeness**, whether all record numbers appear in the S1 address; missing numbers are distinguished from contradicting ones.
- *Other:* raw lower-cased name / address equality, S2 vs S3, website-name and Indic flags, number of S1 entities with the same core name (catalog ambiguity).
- **Removed:** the original feature counting how many records retrieve an S1 at rank 1. It depends on other records, and duplicating a rejected wrong-branch record flipped 3 of 20,000 decisions.

**Model type:** first stage LightGBM (255 leaves, lr 0.05, 1,317 trees by early stopping), trained on 15.5M candidate pairs from 25% of TRAIN-fold records. Second stage LightGBM (127 leaves, 1,852 trees) on each record's best candidate, using the first-stage features plus the record's own top-3 first-stage probabilities and margin. It is trained on TRAIN records that the first stage never saw, so its inputs are out-of-sample.
**Assignment and threshold:** each record goes to its highest-probability candidate if p ≥ 0.70. That threshold was chosen on the development fold; the stable region is 0.65–0.80. No country-specific thresholds (no France labels exist).

---

## 5. Results & Error Analysis

**Evaluation design.** S1 entities are split by id: TRAIN (60%), DEV (20%, development), CONF (20%, sealed). Records follow their true S1 entity (or their top candidate if unmatched). All models, the dictionary and thresholds are fit without DEV/CONF labels. Scores are the exact challenge metric (per-S1 `5TP/(5TP+4FP+FN)`, empty-set convention, all entities averaged) over the full 10.3M-record corpus.

| model (same data, same protocol) | DEV macro F0.5 | notes |
|---|---|---|
| B0: original recipe refit on TRAIN | 0.97785 | reference |
| C1: B0 without the population feature | 0.97791 | duplication-invariant |
| C2: + direct-evidence features | 0.98056 | |
| C3: + 2.5× training data | 0.98201 | |
| **C4: + per-record second stage (final)** | **0.98322** | |

**Stress suite from real labels.** In 31,050 DEV groups of same-name businesses at different addresses, one business was removed from the catalog. Its real records (107k) became a coherent same-name, wrong-address cluster pointing at the 121,347 remaining twins.

| | B0 | final |
|---|---|---|
| DEV-stress macro F0.5 | 0.97782 | **0.98324** |
| twins receiving the clusters | 0.97261 | **0.97744** |
| zero-match twins | 0.9662 | **0.9724** |
| distractor records accepted | 1.99% | 1.85% |
| decision flips when a rejected distractor is duplicated ×8 | 3/20,000 | 0 (by construction) |
| cluster prevalence ×8 (exact simulation) | 0.97693 (approx.) | **0.98235** |

- **Sealed confirmation (opened once):** baseline recipe 0.97787, final **0.98321**, paired family-level bootstrap Δ +0.00533 [95% CI +0.00514, +0.00554]. India 0.9749 → 0.9823, US 0.9799 → 0.9838; zero-match S1 entities 0.9779 → 0.9758 (a small, bounded trade-off).
- **Test acceptance:** 57.8% of records linked, versus 58.8% for the leaderboard-scored baseline and 63.7% for the failed sibling model.
- **Common false positives:** near-copy records that differ in one word or number; name-only records whose generic name belongs to several S1 entities.
- **Common false negatives:** records without an address and an ambiguous name; heavily corrupted names with partial addresses.

**What is not measured:** France (no labels) and the private leaderboard. Internal folds and the stress suite are labelled accordingly and do not predict the leaderboard score.

---

## 6. Conclusion
Precision-weighted entity resolution rewards being right about *which* business a record belongs to, and it punishes evidence that a different data distribution can counterfeit. Direct record-to-candidate evidence, IDF-weighted identity tokens, premise-aware numbers, more training data and a per-record second stage raised the sealed-fold score from 0.9779 to 0.9832, without the sibling-agreement signal that failed on the test set.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/` contains the base stages (`prepare_data.py`, `build_normalized.py`, `candidates.py`, `build_features.py`) and `v2/` (the final pipeline: `run_v2.py` orchestrates `v2_prepare.py`, `make_qf.py`, `featx.py`, `train_v2.py`, `stack_v2.py`, `predict_v2.py`, plus the evaluation tools). See `README.md` for exact commands. All stages are resumable, with checksummed manifests.

### B. Additional Results
DEV threshold sweep for the final model: F0.5 = 0.98312 / 0.98322 / 0.98321 / 0.98306 at t = 0.65 / 0.70 / 0.75 / 0.80 (within 0.00016 of the maximum across 0.65–0.80); t = 0.70 was frozen before the confirmation fold was opened. Retrieval ceiling (oracle) 0.99609 versus final 0.98322: the remaining gap is in classification of ambiguous and no-address records, not in retrieval.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.


---

## Addendum: France (unseen country) handling — v4

**Why:** local validation 0.983 but leaderboard 0.970. US/India test behave like DEV, while France (15% of test, no training data) has twice the uncertain band. A labelled proxy (train on one country, test on the other) shows a 0.04–0.05 loss on an unseen country.

**French-locale normalisation (France records only):** noise vocabulary was mined without labels from high-confidence test matches. Region and department names in whole address components map to one region code (the S1 side carries the region, S2/S3 records often the department; the analogue of US state name vs code). Also handled: `bis/ter`, `crs` = cours, rond-point, and the legal forms `cie`/`compagnie`/`ei`. France is re-retrieved and re-scored; US/India are untouched.

**One-round self-training for France:** the matcher is refit on labelled US+India rows plus France pseudo-labels (records whose top candidate probability is ≥ 0.98 → match / ≤ 0.02 → non-match). It is validated on the transfer proxy in both directions: US→India best 0.9344 vs 0.9295; India→US 0.9453 vs 0.9455 (neutral). A second round was worse and is not used. France threshold 0.9: an unseen country prefers a higher threshold in every proxy setting.

**Result files:** US/India identical to the scored 0.970441 submission; only France changes. No France labels exist, so the France effect is measured only by the leaderboard.
