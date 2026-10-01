# Business Entity Resolution — reproducible pipeline

Produces `output/matching_results.tsv` (leaderboard file) and `output/candidate_pairs.tsv` (the exact
candidate set the models score) from the challenge TSVs.

**Final model (v2):** retrieval (10 S1 candidates per S2/S3 record) → first-stage LightGBM on direct
record↔S1 evidence (no features that depend on other records) → per-record second stage → each record is
linked to its best S1 if p ≥ 0.70. Development macro F0.5 0.98322; one-shot confirmation fold 0.98321
(baseline recipe on the same fold 0.97787). See `../Documentation_template.md`.

## Environment

* Python 3.10, Windows or Linux, 16 GB RAM. Every stage streams data in batches, and a memory watchdog
  aborts a stage cleanly (it resumes later) rather than exhausting the machine.
* An NVIDIA GPU is strongly recommended for candidate generation (tested on an RTX 4050 Laptop, 6 GB).
  Without CUDA the same code runs on CPU, but much slower.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install -r requirements.txt
```

## Data layout

```
<repo>/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
<repo>/dataset/test/test_source{1,2,3}.tsv
```
Locations can be overridden with `ER_DATA_DIR`, `ER_CACHE_DIR` and `ER_OUTPUT_DIR`.

## Run (final v2 pipeline)

1. Base stages (normalization, Indic transliteration, retrieval, base pair features):
   ```bash
   cd src
   python prepare_data.py
   python build_normalized.py train test
   python candidates.py train test
   python build_features.py train test
   ```
2. v2 stages (split manifest, TRAIN-fold dictionary, direct features, models, submission bundle):
   ```bash
   cd src/v2
   python run_v2.py 0.7
   ```
   Output bundle: `cache/runs/submission_C4_stack_on_C3_direct_x_more_data/{matching_results.tsv,candidate_pairs.tsv,bundle.json}`.
   Copy both TSVs to `output/`.
3. Validate (run on its own — the validator needs ~10 GB RAM):
   ```bash
   python utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir dataset/test
   ```

Every stage writes to `cache/runs/<run_id>/` with a `manifest.json` (config, code fingerprint, per-output
sha256). Re-running a stage skips completed work; an interrupted stage resumes from its last completed file
(verified: killed after 4/53 files, resumed, outputs byte-identical to a clean recomputation). Re-running
`run_v2.py` on completed stages regenerated a byte-identical `matching_results.tsv`.

## Evaluation tools (`src/v2/`)

| script | purpose |
|---|---|
| `scorer.py` | exact macro-per-S1 F0.5 `5TP/(5TP+4FP+FN)`, empty-set convention, unit tests, paired bootstrap |
| `eval_v2.py` | DEV evaluation, threshold sweep, slices, candidate-restricted oracle ceiling |
| `stress_build.py`, `stress_eval.py` | clustered same-name/wrong-address stress suite from real labels + duplication-invariance check |
| `prevalence.py` | cluster-prevalence sensitivity (×1–×8) |
| `confirm_v2.py` | one-shot confirmation on the sealed CONF fold (refuses to run twice) |

## Source files

| file | role |
|---|---|
| `config.py` | paths / seed |
| `prepare_data.py` | TSV → parquet (no quoting, all columns as strings) |
| `normalization.py` | name & address normalization, Indic-script transliteration, legal-suffix removal |
| `build_normalized.py` | multiprocess normalization of every record |
| `indic_dictionary.py`, `apply_indic_dict.py` | v1 dictionary tools (v2 learns the dictionary on the TRAIN fold in `v2/v2_prepare.py`; identical result) |
| `retrieval.py` | per-country TF-IDF (name char-3-grams + address tokens), GPU CountSketch top-40, exact re-scoring |
| `candidates.py` | candidate generation (query = S2/S3 record, index = S1); `patch_queries` re-generates a subset |
| `features.py` | 64 base pair features (rapidfuzz similarities, token / house-number overlap, candidate context) |
| `build_features.py` | base features for every candidate chunk |
| `v2/runlib.py` | run directories, manifests, atomic writes, process lock, memory watchdog, progress log |
| `v2/v2_prepare.py` | TRAIN/DEV/CONF split, TRAIN-fold Indic dictionary, v2 tables, Indic re-retrieval |
| `v2/make_qf.py` | compact label / fold table |
| `v2/featx.py` | 16 direct-evidence features (IDF-weighted name/address agreement, premise vs unit numbers, raw-text equality) |
| `v2/configs_v2.py` | model configurations |
| `v2/train_v2.py` | first-stage LightGBM (TRAIN fold only), scoring of any split |
| `v2/stack_v2.py` | per-record second stage (out-of-sample first-stage inputs) |
| `v2/predict_v2.py` | submission bundle from one run: matches ⊆ candidates, one S1 per record, all S1 rows |
| `v2/run_v2.py` | orchestrates the v2 stages |
| `train.py`, `predict.py`, `metrics.py`, `run_all.py` | v1 pipeline (kept for reference; `predict.write_grouped` is reused) |

## v4: France handling (current recommendation)

France appears only in test. Two France-only steps run on top of the v2 pipeline (US/India outputs are unchanged):

```bash
cd src/v4
python v4_prepare.py                         # v4test split: French-locale re-normalisation of France records
cd ..                                        # re-retrieve / re-featurise France (US/India reused), score C3/C4
python -c "import candidates; candidates.generate('v4test')"
python -c "import build_features; build_features.build('v4test')"
cd v2 && python featx.py v4test && python train_v2.py C3_direct_x_more_data v4test && python stack_v2.py C3_direct_x_more_data v4test
cd ../v4
python fr_pseudo.py fr1 0.98 0.02 0.2 0.10    # self-training model for France (pseudo-labels from confident C3 predictions)
python assemble.py c4fr_frp_t90 frp_fr1 0.9 c4   # -> output/v4_c4fr_frp_t90/ (validated)
```

| file | role |
|---|---|
| `v4/fr_locale.py` | French-locale normaliser (region/department codes, bis/ter, cours, legal forms) |
| `v4/v4_prepare.py` | builds the `v4test` split (France re-normalised; US/India hard-linked) |
| `v4/fr_pseudo.py` | one-round self-training for France (US+India labelled rows + France pseudo-labels) |
| `v4/assemble.py` | final file: C4 for US/India, chosen France source/threshold; validator + subset checks |
| `v4/tx.py`, `v4/tx_pseudo.py`, `v4/tx_errors.py` | cross-country transfer proxy (train one country, evaluate the other) |
| `v4/diag_shift.py`, `v4/mine_noise_tokens.py`, `v4/compare_fr.py` | label-free test diagnostics |

Build the zip: `python make_submission_v4.py <team> v4_c4fr_frp_t90` (from the repo root).
