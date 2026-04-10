"""
Recommendation script.
Given a query image + text, find the top-K most compatible items
from the catalog using cosine similarity on fused embeddings.
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from datasets import load_from_disk
from model import FashionCompatibilityModel

CKPT_PATH  = "models/best_model.pt"
EMB_CACHE  = "models/catalog_embeddings.pt"
DATA_PATH  = "data/polyvore_outfits"
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_model():
    model = FashionCompatibilityModel().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()
    return model


def build_catalog(model, max_items: int = 500):
    """Embed catalog items and save to disk. Run once."""
    print("Building catalog embeddings...")
    ds = load_from_disk(DATA_PATH)["data"].select(range(max_items))

    embeddings = []
    metadata   = []

    for i in range(len(ds)):
        row  = ds[i]
        img  = row["image"].convert("RGB")
        img_t = val_transform(img).unsqueeze(0).to(DEVICE)
        text  = f"{row['category']}. {row['text']}".strip()

        with torch.no_grad():
            emb = model.encode_item(img_t, [text], DEVICE).squeeze(0).cpu()

        embeddings.append(emb)
        metadata.append({
            "item_id":  row["item_ID"],
            "category": row["category"],
            "text":     row["text"],
        })

        if (i + 1) % 50 == 0:
            print(f"  Embedded {i+1}/{len(ds)} items...")

    torch.save({"embeddings": embeddings, "metadata": metadata}, EMB_CACHE)
    print(f"Catalog saved to {EMB_CACHE}")
    return embeddings, metadata


def recommend(query_img_path: str, query_text: str = "",
              top_k: int = 5):
    model = load_model()

    # Load or build catalog
    if Path(EMB_CACHE).exists():
        cache      = torch.load(EMB_CACHE, map_location="cpu")
        embeddings = cache["embeddings"]
        metadata   = cache["metadata"]
        print(f"Loaded catalog with {len(metadata)} items")
    else:
        embeddings, metadata = build_catalog(model)

    # Embed query
    img   = Image.open(query_img_path).convert("RGB")
    img_t = val_transform(img).unsqueeze(0).to(DEVICE)
    text  = query_text if query_text else "fashion item"

    with torch.no_grad():
        query_emb = model.encode_item(img_t, [text], DEVICE).squeeze(0).cpu()

    # Cosine similarity against catalog
    cat_matrix = torch.stack(embeddings)          # (N, 256)
    q_norm     = query_emb / (query_emb.norm() + 1e-8)
    c_norm     = cat_matrix / (cat_matrix.norm(dim=1, keepdim=True) + 1e-8)
    sims       = (c_norm @ q_norm).numpy()        # (N,)

    top_indices = np.argsort(sims)[::-1][:top_k]

    print(f"\nTop {top_k} recommendations for: '{text}'")
    print("─" * 50)
    results = []
    for rank, idx in enumerate(top_indices, 1):
        item = metadata[idx]
        print(f"  {rank}. [{sims[idx]:.3f}] {item['category']} — {item['text']}")
        results.append({**item, "score": float(sims[idx])})

    return results


if __name__ == "__main__":
    import sys

    # Build catalog first if it doesn't exist
    if not Path(EMB_CACHE).exists():
        m = load_model()
        build_catalog(m, max_items=500)

    # Use a sample image from the dataset for demo
    print("\nLoading a sample query image from dataset...")
    ds  = load_from_disk(DATA_PATH)["data"]
    row = ds[0]
    img = row["image"].convert("RGB")
    img.save("data/sample_query.jpg")
    query_text = f"{row['category']}. {row['text']}"
    print(f"Query item: {query_text}")

    recommend("data/sample_query.jpg", query_text, top_k=5)