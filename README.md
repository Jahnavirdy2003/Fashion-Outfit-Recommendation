# Multimodal Fashion Outfit Recommendation

A multimodal machine learning system that recommends compatible clothing items given a fashion item image and text description. Built using image embeddings (EfficientNet-B0) and text embeddings (Sentence-BERT), fused together to learn fashion compatibility from real outfit data.

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
| Marqo/Polyvore Dataset | 94,096 fashion items with images + descriptions |
| scikit-learn | AUC metric computation |
| HuggingFace Datasets | Dataset loading |

---

## Project Structure

```
Fashion-Outfit-Recommendation/
│
├── data/
│   └── polyvore_outfits/      # Dataset (downloaded separately)
│
├── models/
│   ├── best_model.pt          # Trained model checkpoint
│   └── catalog_embeddings.pt  # Pre-computed catalog embeddings
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   └── 02_demo.ipynb          # End-to-end demo
│
├── src/
│   ├── dataset.py             # Dataset loading + compatibility pairs
│   ├── encoders.py            # Image + Text encoders
│   ├── model.py               # Fusion + compatibility model
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Evaluation metrics
│   ├── recommend.py           # Inference + recommendations
│   └── main.py                # CLI entry point
│
├── requirements.txt
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

### Get recommendations
```bash
python src/main.py recommend --text "black leather boots" --topk 5
```

---

## Results

| Metric | Random Baseline | Our Model |
|--------|----------------|-----------|
| Accuracy | 50.00% | **69.75%** |
| AUC | 0.5000 | **0.7632** |

The model is **19.75% more accurate** than random guessing at predicting fashion compatibility.

### Sample Output
```
Top 5 recommendations for: 'black leather boots'
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

- Trained on a subset of 500 outfits due to CPU constraints
- No category filtering (may recommend same category items)
- Static catalog — new items require rebuilding embeddings

## Future Improvements

- Train on full dataset using GPU for higher accuracy
- Add category-aware filtering (e.g., boots → recommend tops/bottoms only)
- Build a Streamlit web interface for interactive demos
- Fine-tune the text encoder for fashion-specific language