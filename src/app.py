"""
Fashion Outfit Recommendation — Web App
Allows users to upload or scan a clothing item and get compatible recommendations.

Usage:
  streamlit run src/app.py
"""
import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from datasets import load_from_disk

# Must import from src directory
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import FashionCompatibilityModel

# ── Config ───────────────────────────────────────────────
CKPT_PATH = "models/best_model.pt"
EMB_CACHE = "models/catalog_embeddings.pt"
DATA_PATH = "data/polyvore_outfits"

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

# Map dataset categories to groups
TOPS = ["Tops", "Blouses", "T-Shirts", "Tank Tops", "Sweaters", "Sweatshirts"]
OUTERWEAR = ["Jackets", "Coats"]
BOTTOMS = ["Pants", "Skinny Jeans", "Shorts", "Knee Length Skirts", "Leggings"]
DRESSES = ["Day Dresses", "Cocktail Dresses", "Maxi Dresses"]
SHOES = ["Sandals", "Pumps", "Ankle Booties", "Sneakers", "Boots", "Flats"]
BAGS = ["Shoulder Bags", "Clutches", "Handbags", "Tote Bags", "Backpacks"]
JEWELRY = ["Earrings", "Necklaces", "Bracelets & Bangles", "Rings"]
ACCESSORIES = ["Sunglasses", "Hats", "Watches", "Belts", "Scarves"]

# What to recommend for each group
COMPATIBLE = {
    "tops":       BOTTOMS + SHOES + BAGS + OUTERWEAR + JEWELRY + ACCESSORIES,
    "outerwear":  TOPS + BOTTOMS + SHOES + BAGS + DRESSES,
    "bottoms":    TOPS + SHOES + BAGS + OUTERWEAR + JEWELRY + ACCESSORIES,
    "dresses":    SHOES + BAGS + OUTERWEAR + JEWELRY + ACCESSORIES,
    "shoes":      TOPS + BOTTOMS + DRESSES + BAGS,
    "bags":       TOPS + BOTTOMS + DRESSES + SHOES + OUTERWEAR,
    "jewelry":    TOPS + BOTTOMS + DRESSES,
    "accessories": TOPS + BOTTOMS + DRESSES + SHOES,
}

def get_category_group(category: str) -> str:
    if category in TOPS: return "tops"
    if category in OUTERWEAR: return "outerwear"
    if category in BOTTOMS: return "bottoms"
    if category in DRESSES: return "dresses"
    if category in SHOES: return "shoes"
    if category in BAGS: return "bags"
    if category in JEWELRY: return "jewelry"
    if category in ACCESSORIES: return "accessories"
    return "unknown"

# ── Load model + catalog (cached so it only runs once) ──
@st.cache_resource
def load_model():
    model = FashionCompatibilityModel().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()
    return model


@st.cache_resource
def load_catalog():
    if Path(EMB_CACHE).exists():
        cache = torch.load(EMB_CACHE, map_location="cpu")
        return cache["embeddings"], cache["metadata"]
    else:
        st.error("Catalog embeddings not found! Run `python src/recommend.py` first to build them.")
        st.stop()


@st.cache_resource
def load_dataset_images():
    ds = load_from_disk(DATA_PATH)["data"]
    return ds


# ── Recommendation logic ────────────────────────────────
def get_recommendations(query_img: Image.Image, query_text: str, top_k: int):
    model = load_model()
    embeddings, metadata = load_catalog()

    # Embed query
    img_t = val_transform(query_img).unsqueeze(0).to(DEVICE)
    text = query_text if query_text else "fashion item"

    with torch.no_grad():
        query_emb = model.encode_item(img_t, [text], DEVICE).squeeze(0).cpu()

    # Cosine similarity
    cat_matrix = torch.stack(embeddings)
    q_norm = query_emb / (query_emb.norm() + 1e-8)
    c_norm = cat_matrix / (cat_matrix.norm(dim=1, keepdim=True) + 1e-8)
    sims = (c_norm @ q_norm).numpy()

    sorted_indices = np.argsort(sims)[::-1]

    # Detect query category from text or let user pick
    query_group = None
    if query_text:
        # Check if query text matches any known category
        for cat in TOPS + OUTERWEAR + BOTTOMS + DRESSES + SHOES + BAGS + JEWELRY + ACCESSORIES:
            if cat.lower() in query_text.lower():
                query_group = get_category_group(cat)
                break
        # Also check common words
        qt = query_text.lower()
        if not query_group:
            if any(w in qt for w in ["shirt", "polo", "tee", "top", "blouse", "sweater"]):
                query_group = "tops"
            elif any(w in qt for w in ["pant", "jean", "short", "skirt", "legging"]):
                query_group = "bottoms"
            elif any(w in qt for w in ["dress"]):
                query_group = "dresses"
            elif any(w in qt for w in ["shoe", "boot", "sandal", "sneaker", "pump", "heel"]):
                query_group = "shoes"
            elif any(w in qt for w in ["jacket", "coat"]):
                query_group = "outerwear"
            elif any(w in qt for w in ["bag", "clutch", "tote", "backpack"]):
                query_group = "bags"

    results = []
    compatible_cats = COMPATIBLE.get(query_group, []) if query_group else []

    for idx in sorted_indices:
        item = metadata[idx]
        # If we know query group, only show compatible categories
        if compatible_cats and item["category"] not in compatible_cats:
            continue
        results.append({
            "item_id": item["item_id"],
            "category": item["category"],
            "text": item["text"],
            "score": float(sims[idx]),
            "catalog_idx": idx,
        })
        if len(results) >= top_k:
            break
        if len(results) >= top_k:
            break
    return results


def get_item_image(item_id: str):
    """Load image for a catalog item from the dataset."""
    ds = load_dataset_images()
    # Find the item by ID
    for i in range(len(ds)):
        if ds[i]["item_ID"] == item_id:
            return ds[i]["image"].convert("RGB")
    return None


# ── Streamlit UI ────────────────────────────────────────
st.set_page_config(
    page_title="Fashion Outfit Recommender",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Fashion Outfit Recommender")
st.markdown("Upload a clothing item or scan one with your camera to get compatible outfit recommendations!")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of recommendations", 3, 10, 5)
    query_text = st.text_input(
        "Item description (helps filter results)",
        placeholder="e.g., polo shirt, blue denim jacket, summer dress"
    )
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. Upload or scan a clothing item\n"
        "2. The model encodes it using EfficientNet + SBERT\n"
        "3. Compares against catalog items via cosine similarity\n"
        "4. Returns the most compatible items"
    )
    st.markdown("---")
    st.markdown(
        "**Model:** EfficientNet-B0 + Sentence-BERT\n\n"
        "**Dataset:** Polyvore Outfits\n\n"
        f"**Device:** `{DEVICE}`"
    )

# Input methods
tab_upload, tab_camera = st.tabs(["📁 Upload image", "📸 Camera scan"])

query_image = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a clothing image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a photo of a clothing item"
    )
    if uploaded_file:
        query_image = Image.open(uploaded_file).convert("RGB")

with tab_camera:
    camera_photo = st.camera_input("Scan a clothing item")
    if camera_photo:
        query_image = Image.open(camera_photo).convert("RGB")

# Process and show recommendations
if query_image:
    st.markdown("---")

    col_query, col_results = st.columns([1, 3])

    with col_query:
        st.subheader("Your item")
        st.image(query_image, use_container_width=True)
        if query_text:
            st.caption(f"Description: *{query_text}*")

    with col_results:
        st.subheader(f"Top {top_k} compatible items")

        with st.spinner("Finding compatible items..."):
            results = get_recommendations(query_image, query_text, top_k)

        # Display in a grid
        cols = st.columns(min(top_k, 5))
        for i, item in enumerate(results):
            with cols[i % 5]:
                # Try to load the actual image
                item_img = get_item_image(item["item_id"])
                if item_img:
                    st.image(item_img, use_container_width=True)
                else:
                    st.markdown("🖼️ *Image not available*")

                # Score as progress bar
                st.progress(max(0.0, min(1.0, item["score"])))

                st.markdown(f"**{item['category']}**")
                st.caption(item["text"][:60] + "..." if len(item["text"]) > 60 else item["text"])
                st.markdown(f"Score: `{item['score']:.3f}`")

        # Show more results if top_k > 5
        if top_k > 5:
            cols2 = st.columns(min(top_k - 5, 5))
            for i, item in enumerate(results[5:]):
                with cols2[i % 5]:
                    item_img = get_item_image(item["item_id"])
                    if item_img:
                        st.image(item_img, use_container_width=True)
                    else:
                        st.markdown("🖼️ *Image not available*")

                    st.progress(max(0.0, min(1.0, item["score"])))
                    st.markdown(f"**{item['category']}**")
                    st.caption(item["text"][:60] + "..." if len(item["text"]) > 60 else item["text"])
                    st.markdown(f"Score: `{item['score']:.3f}`")

else:
    # Show placeholder
    st.markdown("---")
    st.info("👆 Upload an image or use the camera to scan a clothing item to get started!")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built by Kishan, Jahnavi & Tridev | "
    "Multimodal Fashion Recommendation | "
    "Northeastern University"
    "</div>",
    unsafe_allow_html=True
)