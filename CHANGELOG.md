# CHANGELOG — All Modifications to Fashion Outfit Recommendation

## Session Summary
All changes made during the learn-and-build tutoring sessions with Claude.
Use this for git commit messages and project documentation.

---

## Phase 1: Bug Fixes & Observability (from Claude Code session)

### src/train.py
1. **Added `import time`** — for tracking epoch duration
2. **MPS device support** — detect Apple Silicon GPU (`torch.backends.mps`) instead of falling back to CPU
3. **Progress prints** — epoch header with % done, total batches count before training starts
4. **Batch loss logging** — `tqdm.write` after each batch for visibility
5. **ETA tracking** — time elapsed + estimated time remaining after each epoch
6. **Typo fix** — `state_dßict` → `state_dict` (would have crashed on model save)

### src/evaluate.py
1. **MPS device support** — same Apple Silicon GPU fix

### src/recommend.py
1. **MPS device support** — same Apple Silicon GPU fix

### src/dataset.py (original Claude Code changes)
1. **Complete rewrite** — replaced JSON-based loading with HuggingFace `load_from_disk` for Marqo/Polyvore dataset
2. **Removed FITBDataset** — not available in Marqo/Polyvore format
3. **Outfit ID parsing** — items grouped by splitting `item_ID` on `_` separator
4. **Train/val split** — ratio-based split (90/10) instead of separate JSON files
5. **Outfit cap** — limited to 5000 outfits for fast training on CPU/MPS

### src/encoders.py (original Claude Code changes)
1. **Removed `freeze_backbone` parameter** — backbone now trains by default (fine-tuning for fashion-specific features)
2. **Added `emb = emb.clone()`** — fixes autograd issue when exiting SBERT inference mode

### src/model.py (original Claude Code changes)
1. **No functional changes** — architecture remained the same

---

## Phase 2: Experiment Tracking System (from Claude.ai tutoring session)

### src/dataset.py — MODIFIED
- **Added `num_outfits` parameter** to `PolyvoreDataset.__init__()` (default=5000)
- Replaces hardcoded `outfit_ids[:5000]` with configurable `outfit_ids[:num_outfits]`
- Backward compatible — existing code works without changes
- **Commit message:** `feat(dataset): add configurable num_outfits parameter for experiment control`

### src/model.py — MODIFIED
- **Added `dropout` parameter** to `FashionCompatibilityModel.__init__()` (default=0.3)
- Replaces hardcoded `nn.Dropout(0.3)` with configurable `nn.Dropout(dropout)`
- Backward compatible — existing code works without changes
- **Commit message:** `feat(model): add configurable dropout parameter for experiment tuning`

### src/experiment.py — NEW FILE
- **Full experiment runner** with automated insight generation
- Self-contained: does NOT call train.py or evaluate.py — has its own training + eval loop
- Accepts CLI arguments: `--name`, `--outfits`, `--epochs`, `--lr`, `--freeze`, `--dropout`, `--weight_decay`
- **Auto-generates per experiment:**
  - `config.json` — exact settings used (reproducibility)
  - `training_log.csv` — per-epoch metrics (epoch, train_loss, train_acc, val_loss, val_acc, lr, time)
  - `training_curves.png` — loss + accuracy curves with annotations
  - `roc_curve.png` — ROC curve with AUC score
  - `score_distribution.png` — histogram of compatible vs incompatible pair scores
  - `eval_results.json` — final evaluation metrics
  - `best_model.pt` — saved best checkpoint
- **Comparison mode** (`--compare`): generates side-by-side bar charts, overlaid training curves, and summary CSV across all experiments
- All outputs saved to `experiments/{experiment_name}/` for clear organization
- **Commit message:** `feat(experiment): add experiment runner with automated insight generation and cross-experiment comparison`

### run_all_experiments.sh — NEW FILE
- Shell script to run all 6 experiments sequentially
- Experiments planned:
  1. `exp1_baseline_5k_unfrozen` — baseline (5k outfits, unfrozen, lr=1e-4)
  2. `exp2_frozen_backbone` — freeze EfficientNet backbone
  3. `exp3_more_data_10k` — double training data to 10k outfits
  4. `exp4_lower_lr_1e5` — reduce learning rate 10x
  5. `exp5_high_dropout` — increase dropout from 0.3 to 0.5
  6. `exp6_best_combo` — frozen + 10k outfits + low LR
- Auto-runs `--compare` after all experiments complete
- **Commit message:** `feat(scripts): add experiment batch runner for 6 hyperparameter iterations`

---

## Phase 3: Experiment Runner Improvements

### src/experiment.py — BUG FIX
- **Fixed ValueError on line 240** — old code tried to read CSV as raw text lines, crashed with `could not convert string to float`
- Replaced with clean pandas `pd.read_csv()` approach for best epoch calculation
- **Commit message:** `fix(experiment): fix best_epoch CSV parsing crash`

### run_all_experiments.sh — UPDATED
- **Added sanity check** — runs a quick 100-outfit, 1-epoch test before starting real experiments
- If sanity check fails, script exits immediately instead of wasting hours
- **Added auto-backup** — zips existing experiments to `experiments/backups/backup_YYYYMMDD_HHMMSS.zip` before clearing
- Backups exclude previous backup zips to avoid nesting
- Old experiment folders are cleaned but `backups/` is preserved

### .gitignore — UPDATED
- Added `experiments/backups/` to prevent large zip files from being committed

---

## Phase 4: Web App + Documentation

### src/app.py — NEW FILE
- **Streamlit web application** for interactive fashion recommendations
- **Two input methods:** image upload from gallery + live camera scan
- Camera scan works on mobile browsers (phone accesses laptop server via WiFi)
- Displays top-K recommendations with item images, category, description, and compatibility scores
- Adjustable number of recommendations (3-10) via sidebar slider
- Optional text description input for improved results
- Sidebar shows model info, device, and "How it works" guide
- Team credits in footer
- Uses `@st.cache_resource` for efficient model/catalog loading
- **Tested on:** MacBook (laptop browser) + iPhone (mobile browser via same WiFi)
- **Commit message:** `feat: add Streamlit web app with camera scan and mobile support`

### requirements.txt — UPDATED
- Added `streamlit>=1.30.0`

### README.md — UPDATED
- Updated tech stack to include Streamlit
- Updated project structure with new files (experiment.py, app.py, experiments/, CHANGELOG.md)
- Added web app usage instructions with mobile camera scanning
- Added hyperparameter experiment commands
- Added experiment results table (6 experiments)
- Updated results from "500 outfits on CPU" to "5,000 outfits on M4 Max MPS"
- Removed "Build Streamlit web interface" from Future Improvements (completed!)
- Updated limitations

### CHANGELOG.md — UPDATED
- Added Phase 3 (experiment fixes) and Phase 4 (web app + docs) sections

---

## Git Commit Sequence (recommended order)

```bash
# Commit 1: Original codebase fixes
git add src/train.py src/evaluate.py src/recommend.py src/encoders.py src/dataset.py
git commit -m "fix: MPS device support, observability improvements, HuggingFace dataset migration"

# Commit 2: Parameterize for experiments
git add src/dataset.py src/model.py
git commit -m "feat: add configurable num_outfits and dropout parameters for experiment control"

# Commit 3: Experiment system
git add src/experiment.py run_all_experiments.sh
git commit -m "feat: add experiment runner with automated insight generation and comparison"

# Commit 4: After experiments complete — add results
git add experiments/
git commit -m "results: add outputs from 6 hyperparameter tuning experiments"
```

---

## Experiment Results (Final)

| Experiment | Outfits | Backbone | LR | Dropout | Accuracy | AUC | Best Epoch |
|---|---|---|---|---|---|---|---|
| exp1_baseline_5k_unfrozen | 5,000 | Unfrozen | 1e-4 | 0.3 | 69.00% | 0.7481 | 2 |
| exp2_frozen_backbone | 5,000 | Frozen | 1e-4 | 0.3 | 66.89% | 0.7385 | 3 |
| exp3_more_data_10k | 10,000 | Unfrozen | 1e-4 | 0.3 | 68.19% | 0.7550 | 2 |
| exp4_lower_lr_1e5 | 5,000 | Unfrozen | 1e-5 | 0.3 | 61.86% | 0.6677 | 5 |
| exp5_high_dropout | 5,000 | Unfrozen | 1e-4 | 0.5 | 69.06% | 0.7520 | 3 |
| exp6_best_combo | 10,000 | Frozen | 1e-5 | 0.3 | 63.53% | 0.6847 | 5 |
| exp7_20k_10epochs | 20,000 | Unfrozen | 1e-4 | 0.3 | 69.23% | 0.7665 | 3 |
| exp12_allpairs_hardneg_5k | 5,000 | Unfrozen | 1e-4 | 0.3 | 55.95% | 0.5862 | 1 |
| **exp13_40k_10epochs** | **40,000** | **Unfrozen** | **1e-4** | **0.3** | **70.14%** | **0.7767** | **2** |
| exp14_anti_overfit | 40,000 | Unfrozen | 5e-5 | 0.5 | 60.69% | 0.6513 | 3 |

**Best model: exp13_40k_10epochs** (AUC 0.7767, Accuracy 70.14%)

Key insights:
- More data consistently helped (5k→10k→20k→40k improved AUC)
- LR 1e-4 validated as optimal; 1e-5 too slow for epoch budget
- Fine-tuning backbone matters (frozen AUC dropped)
- Hard negatives need more data to work (failed on 5k)
- Anti-overfit settings (lower LR + higher dropout) were too restrictive
- Architecture ceiling at ~70% accuracy — further gains need architectural changes

---

## Phase 5: Extended Experiments & Model Optimization

### Experiments 7-14
- **exp7_20k_10epochs** — scaled to 20k outfits, 10 epochs. AUC 0.7665.
- **exp12_allpairs_hardneg_5k** — all within-outfit pairs + hard negative sampling on 5k. AUC dropped to 0.5862. Hard negatives too difficult for small dataset.
- **exp13_40k_10epochs** — 40k outfits, 10 epochs. NEW BEST: AUC 0.7767, Acc 70.14%.
- **exp14_anti_overfit** — 40k outfits, lr=5e-5, dropout=0.5, weight_decay=5e-4. Too restrictive, AUC dropped to 0.6513.

### src/dataset.py — v2 MAJOR UPDATE
- **All within-outfit pairs** — uses `itertools.combinations` to generate all item pairs, not just consecutive. ~2.7x more training pairs from same data.
- **Hard negative sampling** — negatives sampled from same category as positive partner, forcing model to learn subtle compatibility differences.
- **New parameters:** `pair_mode` ('consecutive' or 'all_pairs'), `hard_negatives` (True/False)
- **Backward compatible** — old experiments still work with original settings.

### src/experiment.py — UPDATED
- Added `--pair_mode` and `--no_hard_neg` CLI flags
- Config JSON logs pair generation strategy

### src/recommend.py — UPDATED
- Catalog size increased from 500 → 5,000 → 10,000 items
- Better recommendation coverage with larger catalog

### Best model deployed
- `models/best_model.pt` — exp13's checkpoint (40k outfits, epoch 2)
- `models/catalog_embeddings.pt` — 10,000 item catalog embeddings

---

## Phase 6: App Merge & Final Polish

### src/app.py — MERGED (Kishan + Jahnavi)
- Combined Kishan's outfit scorer + Jahnavi's text search into unified app
- **Three modes in sidebar:**
  1. Recommend items — upload/camera scan with auto-detect and outfit building
  2. Text search (Jahnavi) — describe an item, find match, build outfit
  3. Outfit scorer (Kishan) — upload 2-6 items, get compatibility score
- **Shared functions:** `build_outfit()` and `show_outfit()` used by all modes
- **Cosine similarity** for recommendations (reliable)
- **Compatibility head** for outfit scoring (trained scorer)
- Resolved merge conflict with Jahnavi's remote changes
- Pushed to Jahnavi's repo: `github.com/Jahnavirdy2003/Fashion-Outfit-Recommendation`