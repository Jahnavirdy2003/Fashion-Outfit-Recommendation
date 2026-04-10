# Multimodal Fashion Outfit Recommendation

A multimodal machine learning system that recommends compatible clothing items given a fashion item description. Built using image embeddings (EfficientNet-B0) and text embeddings (Sentence-BERT), fused together to learn fashion compatibility from real outfit data.

---

## Team Members

| Name | Email |
|------|-------|
| Jahnavi Reddy Poddutoori | poddutoori.j@northeastern.edu |
| Kishan Kumar Selvakumar Anandraj | selvakumaranandraj.k@northeastern.edu |
| Tridev Prabhu | prabhu.t@northeastern.edu |

---

## Project Overview

Given a clothing item (image + text description), this system recommends other items that are **stylistically compatible** — for example:

> Input: *"black leather boots"*
> Output: black leggings, knit dress, tank top, sweatshirt

The system learns compatibility by studying thousands of real outfits curated by fashion stylists, combining both the **visual appearance** and **semantic meaning** of each clothing item.

---

## Architecture

```
Query Image + Text Description
                │
                ▼
┌───────────────────────────────┐
│        ENCODING LAYER         │
│  ImageEncoder  + TextEncoder  │
│  (EfficientNet)  (SBERT)      │
│       ↓               ↓       │
│     256-dim         256-dim   │
│          └─── + ───┘          │
│         Fused Embedding       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      COMPATIBILITY LAYER      │
│  |emb_A - emb_B| → MLP →      │
│  score (0=incompatible,       │
│         1=compatible)         │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│         OUTPUT LAYER          │
│  Top-K compatible items       │
│  ranked by cosine similarity  │
└───────────────────────────────┘
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Deep learning framework |
| EfficientNet-B0 | Image feature extraction (pretrained on ImageNet) |
| Sentence-BERT (all-MiniLM-L6-v2) | Text semantic embeddings |
| Marqo/Polyvore Dataset | 94,096 fashion items available; 5,000–10,000 outfits used for training |
| scikit-learn | AUC metric computation |
| HuggingFace Datasets | Dataset loading |
| Streamlit | Web UI with camera scan + image upload |

---

## Project Structure

```
Fashion-Outfit-Recommendation/
│
├── data/
│   └── polyvore_outfits/      # Dataset (downloaded via setup instructions)
│
├── models/
│   ├── best_model.pt          # Trained model checkpoint
│   └── catalog_embeddings.pt  # Pre-computed catalog embeddings
│
├── experiments/               # Auto-generated experiment results
│   ├── exp1_baseline_5k_unfrozen/
│   ├── exp2_frozen_backbone/
│   ├── ...                    # Each with config, logs, curves, ROC, scores
│   ├── comparison.png         # Cross-experiment comparison chart
│   ├── curves_overlay.png     # Overlaid training curves
│   └── summary.csv            # Results summary table
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis (in progress)
│   └── 02_demo.ipynb          # End-to-end demo (in progress)
│
├── src/
│   ├── dataset.py             # Dataset loading + compatibility pairs
│   ├── encoders.py            # Image + Text encoders
│   ├── model.py               # Fusion + compatibility model
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Evaluation metrics (AUC + Accuracy)
│   ├── recommend.py           # Inference + recommendations
│   ├── experiment.py          # Experiment runner with auto insight generation
│   ├── app.py                 # Streamlit web app (camera scan + upload)
│   └── main.py                # CLI entry point
│
├── run_all_experiments.sh     # Batch runner for all experiments
├── requirements.txt
├── CHANGELOG.md               # Full development history
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Jahnavirdy2003/Fashion-Outfit-Recommendation.git
cd Fashion-Outfit-Recommendation
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
```bash
python -c "from datasets import load_dataset; ds = load_dataset('Marqo/polyvore'); ds.save_to_disk('data/polyvore_outfits'); print('Done!')"
```
> Note: The full dataset contains 94,096 items. We train on a subset of 500 outfits due to CPU constraints. A GPU is recommended for full dataset training.

---

## Usage

### Train the model
```bash
python src/main.py train
```

### Evaluate the model
```bash
python src/main.py evaluate
```

### Get recommendations (CLI)
```bash
python src/main.py recommend --text "black leather boots" --topk 5
```

### Launch web app (camera scan + image upload)
```bash
streamlit run src/app.py
```
Opens at `http://localhost:8501`. Access from your phone on the same WiFi using the Network URL for live camera scanning.

### Run hyperparameter experiments
```bash
# Run a single experiment
python src/experiment.py --name my_exp --outfits 5000 --epochs 5 --lr 1e-4

# Run all 6 predefined experiments
./run_all_experiments.sh

# Compare results across all experiments
python src/experiment.py --compare
```

### Experiment options
```bash
python src/experiment.py \
  --name experiment_name \    # Required: unique name for this run
  --outfits 5000 \            # Number of outfits to train on (default: 5000)
  --epochs 5 \                # Training epochs (default: 5)
  --lr 1e-4 \                 # Learning rate (default: 1e-4)
  --batch_size 32 \           # Batch size (default: 32)
  --freeze \                  # Freeze EfficientNet backbone (default: unfrozen)
  --dropout 0.3 \             # Dropout rate (default: 0.3)
  --weight_decay 1e-4 \       # Weight decay (default: 1e-4)
  --compare                   # Compare all experiments (no training)
```

---

## Results

> Results obtained by training on 5,000 outfits for 5 epochs on Apple M4 Max (MPS).

| Metric | Random Baseline | Our Model |
|--------|----------------|-----------|
| Accuracy | 50.00% | **69.75%** |
| AUC | 0.5000 | **0.7632** |

The model is **19.75% more accurate** than random guessing at predicting fashion compatibility.

### Hyperparameter Experiments
We ran 6 experiments to systematically evaluate design choices:

| Experiment | Outfits | Backbone | LR | Dropout | What it tests |
|---|---|---|---|---|---|
| Baseline | 5,000 | Unfrozen | 1e-4 | 0.3 | Starting point |
| Frozen backbone | 5,000 | Frozen | 1e-4 | 0.3 | Does freezing reduce overfitting? |
| More data | 10,000 | Unfrozen | 1e-4 | 0.3 | Does more data help? |
| Lower LR | 5,000 | Unfrozen | 1e-5 | 0.3 | Does gentler fine-tuning help? |
| High dropout | 5,000 | Unfrozen | 1e-4 | 0.5 | Does stronger regularization help? |
| Best combo | 10,000 | Frozen | 1e-5 | 0.3 | Combining best findings |

Full results with training curves, ROC curves, and score distributions are in the `experiments/` folder.

### Sample Output
```
Top 5 recommendations for query item:
──────────────────────────────────────────────────
  1. [0.621] Day Dresses — tibi knit long sleeve dress
  2. [0.560] Day Dresses — oasis faux leather trim shift dress
  3. [0.538] Tank Tops — wildfox cut tank black
  4. [0.536] Leggings — topshop black heavy leggings
  5. [0.491] Sweatshirts — black polka dots sweatshirt
```

---

## How It Works

1. **Dataset**: Items sharing the same outfit ID are treated as compatible (positive pairs). Random items from different outfits form incompatible pairs (negative pairs).

2. **Image Encoder**: EfficientNet-B0 extracts visual features → projected to 256-dim embedding.

3. **Text Encoder**: Sentence-BERT encodes item descriptions → projected to 256-dim embedding.

4. **Fusion**: Image and text embeddings are added element-wise and normalized → single 256-dim item embedding.

5. **Compatibility Scoring**: `|emb_A - emb_B|` is passed through an MLP → compatibility score (0–1).

6. **Recommendation**: Query item is embedded and compared against all catalog items using cosine similarity → top-K results returned.

---

## Limitations

- No category filtering (may recommend same category items)
- Static catalog — new items require rebuilding embeddings
- Recommendation quality improves significantly with more training data
- Web app image lookup is sequential (could be optimized with index)

## Future Improvements

- Train on full dataset using GPU for higher accuracy
- Add category-aware filtering (e.g., boots → recommend tops/bottoms only)
- Fine-tune the text encoder for fashion-specific language
- Hard negative mining for better compatibility learning
- Deploy Streamlit app to cloud for public access
- Add user preference personalization