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

## Experiment Results (to be filled after runs complete)

| Experiment | Outfits | Backbone | LR | Dropout | Accuracy | AUC | Best Epoch |
|---|---|---|---|---|---|---|---|
| exp1_baseline_5k_unfrozen | 5,000 | Unfrozen | 1e-4 | 0.3 | _pending_ | _pending_ | _pending_ |
| exp2_frozen_backbone | 5,000 | Frozen | 1e-4 | 0.3 | _pending_ | _pending_ | _pending_ |
| exp3_more_data_10k | 10,000 | Unfrozen | 1e-4 | 0.3 | _pending_ | _pending_ | _pending_ |
| exp4_lower_lr_1e5 | 5,000 | Unfrozen | 1e-5 | 0.3 | _pending_ | _pending_ | _pending_ |
| exp5_high_dropout | 5,000 | Unfrozen | 1e-4 | 0.5 | _pending_ | _pending_ | _pending_ |
| exp6_best_combo | 10,000 | Frozen | 1e-5 | 0.3 | _pending_ | _pending_ | _pending_ |

Previous single-run results (before experiment system):
- Accuracy: 69.75% (70.19% at best epoch)
- AUC: 0.7632 (0.7517 on re-eval)