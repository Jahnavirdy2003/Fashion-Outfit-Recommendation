"""
Fashion Outfit Recommendation — Web App
Allows users to upload, scan, or describe a clothing item
and get a complete compatible outfit recommendation.

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

# ── Category definitions ─────────────────────────────────
TOPS        = ["Tops", "Blouses", "T-Shirts", "Tank Tops", "Sweaters", "Sweatshirts"]
OUTERWEAR   = ["Jackets", "Coats"]
BOTTOMS     = ["Pants", "Skinny Jeans", "Shorts", "Knee Length Skirts", "Leggings"]
DRESSES     = ["Day Dresses", "Cocktail Dresses", "Maxi Dresses"]
SHOES       = ["Sandals", "Pumps", "Ankle Booties", "Sneakers", "Boots", "Flats"]
BAGS        = ["Shoulder Bags", "Clutches", "Handbags", "Tote Bags", "Backpacks"]
JEWELRY     = ["Earrings", "Necklaces", "Bracelets & Bangles", "Rings"]
ACCESSORIES = ["Sunglasses", "Hats", "Watches", "Belts", "Scarves"]

COMPATIBLE = {
    "tops":        BOTTOMS + SHOES + BAGS + OUTERWEAR + JEWELRY + ACCESSORIES,
    "outerwear":   TOPS + BOTTOMS + SHOES + BAGS + DRESSES,
    "bottoms":     TOPS + SHOES + BAGS + OUTERWEAR + JEWELRY + ACCESSORIES,
    "dresses":     SHOES + BAGS + OUTERWEAR + JEWELRY + ACCESSORIES,
    "shoes":       TOPS + BOTTOMS + DRESSES + BAGS,
    "bags":        TOPS + BOTTOMS + DRESSES + SHOES + OUTERWEAR,
    "jewelry":     TOPS + BOTTOMS + DRESSES,
    "accessories": TOPS + BOTTOMS + DRESSES + SHOES,
}

OUTFIT_SLOTS = {
    "tops":        "👕 Top",
    "bottoms":     "👖 Bottom",
    "shoes":       "👟 Shoes",
    "outerwear":   "🧥 Outerwear",
    "bags":        "👜 Bag",
    "jewelry":     "💍 Jewelry",
    "accessories": "🕶️ Accessory",
}

def get_category_group(category: str) -> str:
    if category in TOPS:        return "tops"
    if category in OUTERWEAR:   return "outerwear"
    if category in BOTTOMS:     return "bottoms"
    if category in DRESSES:     return "dresses"
    if category in SHOES:       return "shoes"
    if category in BAGS:        return "bags"
    if category in JEWELRY:     return "jewelry"
    if category in ACCESSORIES: return "accessories"
    return "unknown"

def detect_group_from_text(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["shirt", "polo", "tee", "top", "blouse", "sweater", "sweatshirt"]):
        return "tops"
    if any(w in t for w in ["pant", "jean", "short", "skirt", "legging"]):
        return "bottoms"
    if any(w in t for w in ["dress"]):
        return "dresses"
    if any(w in t for w in ["shoe", "boot", "sandal", "sneaker", "pump", "heel", "flat"]):
        return "shoes"
    if any(w in t for w in ["jacket", "coat"]):
        return "outerwear"
    if any(w in t for w in ["bag", "clutch", "tote", "backpack", "handbag"]):
        return "bags"
    if any(w in t for w in ["earring", "necklace", "bracelet", "ring"]):
        return "jewelry"
    return "unknown"

# ── Cached resources ─────────────────────────────────────
@st.cache_resource
def load_model():
    model = FashionCompatibilityModel().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE), strict=False)
    model.eval()
    return model

@st.cache_resource
def load_catalog():
    if Path(EMB_CACHE).exists():
        cache = torch.load(EMB_CACHE, map_location="cpu")
        return cache["embeddings"], cache["metadata"]
    else:
        st.error("Catalog embeddings not found! Run `python src/recommend.py` first.")
        st.stop()

@st.cache_resource
def load_dataset_images():
    ds = load_from_disk(DATA_PATH)["data"]
    # Build a lookup dict: item_ID → index for fast access
    lookup = {ds[i]["item_ID"]: i for i in range(len(ds))}
    return ds, lookup

def get_item_image(item_id: str):
    ds, lookup = load_dataset_images()
    idx = lookup.get(item_id)
    if idx is not None:
        return ds[idx]["image"].convert("RGB")
    return None

def detect_category(query_img, model, embeddings, metadata):
    img_t = val_transform(query_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_item(img_t, ["fashion item"], DEVICE).squeeze(0).cpu()
    cat_matrix = torch.stack(embeddings)
    q_norm = emb / (emb.norm() + 1e-8)
    c_norm = cat_matrix / (cat_matrix.norm(dim=1, keepdim=True) + 1e-8)
    sims = (c_norm @ q_norm).numpy()
    top_indices = np.argsort(sims)[::-1][:10]
    from collections import Counter
    counts = Counter([metadata[i]["category"] for i in top_indices]).most_common()
    for cat, _ in counts:
        if cat != "Clothing":
            return cat
    return counts[0][0]

# ── Core outfit builder ───────────────────────────────────
def build_outfit(query_emb, query_group, embeddings, metadata, top_k):
    """Returns one best item per outfit slot, filtered by compatibility."""
    compatible_cats = COMPATIBLE.get(query_group, [])
    cat_matrix = torch.stack(embeddings)
    q_norm = query_emb / (query_emb.norm() + 1e-8)
    c_norm = cat_matrix / (cat_matrix.norm(dim=1, keepdim=True) + 1e-8)
    sims = (c_norm @ q_norm).numpy()
    sorted_indices = np.argsort(sims)[::-1]

    slots = {k: None for k in OUTFIT_SLOTS if k != query_group}

    for idx in sorted_indices:
        item = metadata[idx]
        group = get_category_group(item["category"])
        if group in slots and slots[group] is None:
            if query_group == "unknown" or item["category"] in compatible_cats:
                slots[group] = {
                    "item_id":  item["item_id"],
                    "category": item["category"],
                    "text":     item["text"],
                    "score":    float(sims[idx]),
                }
        if all(v is not None for v in slots.values()):
            break
    return slots

def show_outfit(slots):
    """Render the outfit grid with images, category, description and score."""
    filled = [(k, v) for k, v in slots.items() if v is not None]
    cols = st.columns(min(len(filled), 4))
    for i, (group, item) in enumerate(filled):
        with cols[i % 4]:
            st.markdown(f"**{OUTFIT_SLOTS[group]}**")
            img = get_item_image(item["item_id"])
            if img:
                st.image(img, use_container_width=True)
            else:
                st.markdown("🖼️ *No image*")
            st.markdown(f"**{item['category']}**")
            st.caption(item["text"][:55] + "..." if len(item["text"]) > 55 else item["text"])
            st.progress(max(0.0, min(1.0, item["score"])))
            st.markdown(f"Score: `{item['score']:.3f}`")

# ── Page layout ───────────────────────────────────────────
st.set_page_config(page_title="Fashion Outfit Recommender", page_icon="👗", layout="wide")
st.title("👗 Fashion Outfit Recommender")
st.markdown("Upload a clothing item, scan with your camera, or describe it — and we'll build a complete outfit!")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of recommendations", 3, 10, 5)
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. Upload, scan, or describe a clothing item\n"
        "2. Model encodes it using EfficientNet + SBERT\n"
        "3. Compares against catalog via cosine similarity\n"
        "4. Builds a complete compatible outfit"
    )
    st.markdown("---")
    st.markdown(f"**Model:** EfficientNet-B0 + Sentence-BERT\n\n**Device:** `{DEVICE}`")

# ── Tabs ──────────────────────────────────────────────────
tab_upload, tab_camera, tab_text = st.tabs(["📁 Upload image", "📷 Camera scan", "🔍 Text search"])

query_image = None
sidebar_text = ""

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a clothing image",
        type=["jpg", "jpeg", "png", "webp"]
    )
    sidebar_text = st.text_input(
        "Item description (optional — helps filtering)",
        placeholder="e.g., blue denim jacket, floral dress",
        key="upload_text"
    )
    if uploaded_file:
        query_image = Image.open(uploaded_file).convert("RGB")

with tab_camera:
    camera_photo = st.camera_input("Scan a clothing item")
    if camera_photo:
        query_image = Image.open(camera_photo).convert("RGB")

with tab_text:
    st.markdown("### 👗 Build Your Outfit")
    st.markdown("Describe a clothing item and we'll build a complete outfit around it!")
    text_query = st.text_input(
        "Describe your item",
        placeholder="e.g., wide leg blue jeans, black leather boots, floral summer dress"
    )
    search_clicked = st.button("✨ Build Outfit", type="primary")

    if search_clicked and text_query:
        with st.spinner("Building your outfit..."):
            model      = load_model()
            embeddings, metadata = load_catalog()
            query_group = detect_group_from_text(text_query)

            # Find best matching catalog image for the text
            with torch.no_grad():
                text_emb = model.text_encoder([text_query], DEVICE).squeeze(0).cpu()
            cat_matrix = torch.stack(embeddings)
            t_norm = text_emb / (text_emb.norm() + 1e-8)
            c_norm = cat_matrix / (cat_matrix.norm(dim=1, keepdim=True) + 1e-8)
            text_sims  = (c_norm @ t_norm).numpy()
            best_idx   = int(np.argmax(text_sims))
            best_img   = get_item_image(metadata[best_idx]["item_id"])

            # Full multimodal embedding
            use_img = best_img if best_img else Image.new("RGB", (224, 224), (255, 255, 255))
            img_t   = val_transform(use_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                query_emb = model.encode_item(img_t, [text_query], DEVICE).squeeze(0).cpu()

            slots = build_outfit(query_emb, query_group, embeddings, metadata, top_k)

        st.markdown("---")
        st.markdown(f"## ✨ Complete Outfit for: *{text_query}*")

        # Show detected query item
        st.markdown("### Your Item")
        q_col, _ = st.columns([1, 4])
        with q_col:
            if best_img:
                st.image(best_img, use_container_width=True)
            st.markdown(f"**{metadata[best_idx]['category']}**")
            st.caption(text_query)
            if query_group != "unknown":
                st.caption(f"Detected: **{query_group}**")

        st.markdown("### Complete the Look")
        show_outfit(slots)

    elif search_clicked and not text_query:
        st.warning("Please enter a description first!")

# ── Image-based recommendations (upload + camera) ────────
if query_image:
    st.markdown("---")
    with st.spinner("Building your outfit..."):
        model      = load_model()
        embeddings, metadata = load_catalog()

        # Detect category
        if sidebar_text:
            query_group = detect_group_from_text(sidebar_text)
            label = sidebar_text
        else:
            detected    = detect_category(query_image, model, embeddings, metadata)
            query_group = get_category_group(detected)
            label       = detected

        # Embed query image
        img_t = val_transform(query_image).unsqueeze(0).to(DEVICE)
        text  = sidebar_text if sidebar_text else label
        with torch.no_grad():
            query_emb = model.encode_item(img_t, [text], DEVICE).squeeze(0).cpu()

        slots = build_outfit(query_emb, query_group, embeddings, metadata, top_k)

    col_query, col_results = st.columns([1, 3])
    with col_query:
        st.subheader("Your item")
        st.image(query_image, use_container_width=True)
        st.caption(f"Category: **{query_group}**")
        if sidebar_text:
            st.caption(f"Description: *{sidebar_text}*")

    with col_results:
        st.subheader("Complete the Look")
        show_outfit(slots)

else:
    if not (search_clicked if 'search_clicked' in dir() else False):
        st.markdown("---")
        st.info("👆 Upload an image, use the camera, or use Text search to get started!")

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built by Kishan, Jahnavi & Tridev | "
    "Multimodal Fashion Recommendation | "
    "Northeastern University"
    "</div>",
    unsafe_allow_html=True
)