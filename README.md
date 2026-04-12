# Multimodal Fashion Outfit Recommendation

A multimodal machine learning system that recommends compatible clothing items given a fashion item image and/or text description. Built using image embeddings (EfficientNet-B0) and text embeddings (Sentence-BERT), fused together to learn fashion compatibility from real outfit data.

---

## Team Members

| Name | Email |
|------|-------|
| Jahnavi Reddy Poddutoori | poddutoori.j@northeastern.edu |
| Kishan Kumar Selvakumar Anandraj | selvakumaranandraj.k@northeastern.edu |
| Tridev Prabhu | prabhu.t@northeastern.edu |

---

## Project Overview

Given a clothing item (image + text description), this system recommends other items that are **stylistically compatible** and builds a **complete outfit** — for example:

> Input: *"wide leg blue jeans"*
> Output: black bandeau top + black leather boots + denim jacket + Chanel bag + bracelet

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
│       ↓               ↓      │
│     256-dim         256-dim   │
│          └─── + ───┘         │
│         Fused Embedding       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      COMPATIBILITY LAYER      │
│  |emb_A - emb_B| → MLP →     │
│  score (0=incompatible,       │
│         1=compatible)         │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│         OUTPUT LAYER          │
│  Complete outfit built via    │
│  category-aware cosine        │
│  similarity search            │
└───────────────────────────────┘
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Deep learning framework |
| EfficientNet-B0 | Image feature extraction (pretrained on ImageNet) |
| Sentence-BERT (all-MiniLM-L6-v2) | Text semantic embeddings |
| Marqo/Polyvore Dataset | 94,096 fashion items; trained on all ~19,000 outfits using GPU |
| scikit-learn | AUC metric computation |
| HuggingFace Datasets | Dataset loading |
| Streamlit | Web UI with image upload, camera scan, and text search |
| Google Colab (T4 GPU) | Model training |

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
│   └── catalog_embeddings.pt  # Pre-computed catalog embeddings (10,000 items)
│
├── experiments/               # Auto-generated experiment results
│   ├── exp1_baseline_5k_unfrozen/
│   ├── exp2_frozen_backbone/
│   ├── ...
│   ├── comparison.png
│   ├── curves_overlay.png
│   └── summary.csv
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
│   ├── app.py                 # Streamlit web app
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
> Note: The full dataset contains 94,096 items across ~19,000 outfits. Training on the full dataset requires a GPU. We trained on Google Colab using a Tesla T4 GPU.

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

### Launch web app
```bash
streamlit run src/app.py
```
Opens at `http://localhost:8501`. The app has 3 tabs:
- **Upload image** — upload a clothing photo to get a complete outfit
- **Camera scan** — use your webcam to scan a clothing item
- **Text search** — type a description to build an outfit

### Run hyperparameter experiments
```bash
# Run a single experiment
python src/experiment.py --name my_exp --outfits 5000 --epochs 5 --lr 1e-4

# Run all predefined experiments
./run_all_experiments.sh

# Compare results across all experiments
python src/experiment.py --compare
```

---

## Results

> Trained on all ~19,000 outfits (94,096 items) for 10 epochs on Google Colab Tesla T4 GPU. Best model saved at epoch 3 based on validation loss.

| Metric | Random Baseline | Our Model |
|--------|----------------|-----------|
| Accuracy | 50.00% | **68.57%** |
| AUC | 0.5000 | **0.7632** |

The model is **18.57% more accurate** than random guessing at predicting fashion compatibility.

### Training Progress
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.6508 | 60.48% | 0.6229 | 65.16% |
| 2 | 0.5802 | 69.57% | 0.6069 | 67.49% |
| **3** | **0.5119** | **75.42%** | **0.6052** | **68.57% ← best** |
| 4 | 0.4355 | 80.73% | 0.6336 | 69.14% |
| 5 | 0.3502 | 85.46% | 0.6523 | 69.80% |

> Model began overfitting after epoch 3 — best checkpoint saved automatically.

### Hyperparameter Experiments
| Experiment | Outfits | Backbone | LR | Dropout | What it tests |
|---|---|---|---|---|---|
| Baseline | 5,000 | Unfrozen | 1e-4 | 0.3 | Starting point |
| Frozen backbone | 5,000 | Frozen | 1e-4 | 0.3 | Does freezing reduce overfitting? |
| More data | 10,000 | Unfrozen | 1e-4 | 0.3 | Does more data help? |
| Lower LR | 5,000 | Unfrozen | 1e-5 | 0.3 | Does gentler fine-tuning help? |
| High dropout | 5,000 | Unfrozen | 1e-4 | 0.5 | Does stronger regularization help? |
| Best combo | 10,000 | Frozen | 1e-5 | 0.3 | Combining best findings |

---

## How It Works

1. **Dataset**: Items sharing the same outfit ID are compatible (positive pairs). Random items from different outfits form incompatible pairs (negative pairs).
2. **Image Encoder**: EfficientNet-B0 extracts visual features → projected to 256-dim embedding.
3. **Text Encoder**: Sentence-BERT encodes item descriptions → projected to 256-dim embedding.
4. **Fusion**: Image and text embeddings are added element-wise and normalized → single 256-dim item embedding.
5. **Compatibility Scoring**: `|emb_A - emb_B|` is passed through an MLP → compatibility score (0–1).
6. **Outfit Building**: Category-aware filtering ensures one item per slot (top, bottom, shoes, bag, jewelry, accessory) with no duplicates.
7. **Recommendation**: Query item is embedded and compared against 10,000 catalog items using cosine similarity → complete outfit returned.

---

## Limitations

- Model overfits after epoch 3 — more regularization needed for longer training
- Text-only search produces lower confidence scores since the model is inherently multimodal
- Static catalog — new items require rebuilding embeddings
- Cosine similarity scores are low (0.3–0.5) due to frozen text encoder and limited training

## Future Improvements

- Fine-tune the text encoder for fashion-specific language
- Hard negative mining for better compatibility learning
- Deploy Streamlit app to cloud (Streamlit Community Cloud) for public access
- Add user preference personalization
- Use contrastive loss instead of BCE for better embedding space geometry