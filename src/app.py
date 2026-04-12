"""
Fashion Outfit Recommendation — Web App
Features:
  1. Upload/scan a clothing item → get compatible outfit recommendations
  2. Text search → describe an item and build an outfit around it (Jahnavi)
  3. Outfit scorer → upload multiple items and get compatibility score (Kishan)

Usage:
  streamlit run src/app.py
"""
import streamlit as st
import torch
import numpy as np
from itertools import combinations
from PIL import Image
from pathlib import Path
from torchvision import transforms
from datasets import load_from_disk
from collections import Counter

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
    "tops":        "Top",
    "bottoms":     "Bottom",
    "shoes":       "Shoes",
    "outerwear":   "Outerwear",
    "bags":        "Bag",
    "jewelry":     "Jewelry",
    "accessories": "Accessory",
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


# ── Auto-detect category from image ─────────────────────
def detect_category(query_img, model, embeddings, metadata):
    img_t = val_transform(query_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_item(img_t, ["fashion item"], DEVICE).squeeze(0).cpu()
    cat_matrix = torch.stack(embeddings)
    q_norm = emb / (emb.norm() + 1e-8)
    c_norm = cat_matrix / (cat_matrix.norm(dim=1, keepdim=True) + 1e-8)
    sims = (c_norm @ q_norm).numpy()
    top_indices = np.argsort(sims)[::-1][:10]
    counts = Counter([metadata[i]["category"] for i in top_indices]).most_common()
    for cat, _ in counts:
        if cat != "Clothing":
            return cat
    return counts[0][0]


# ── Core outfit builder (shared by all modes) ────────────
def build_outfit(query_emb, query_group, embeddings, metadata, top_k):
    """Returns one best item per outfit slot, filtered by compatibility."""
    compatible_cats = COMPATIBLE.get(query_group, [])
    cat_matrix = torch.stack(embeddings)
    q_norm = query_emb / (query_emb.norm() + 1e-8)
    c_norm = cat_matrix / (cat_matrix.norm(dim=1, keepdim=True) + 1e-8)
    sims = (c_norm @ q_norm).numpy()
    sorted_indices = np.argsort(sims)[::-1]

    slots = {k: None for k in OUTFIT_SLOTS if k != query_group}
    extras = []

    for idx in sorted_indices:
        item = metadata[idx]
        group = get_category_group(item["category"])

        if group == query_group:
            continue
        if compatible_cats and item["category"] not in compatible_cats:
            continue

        entry = {
            "item_id":  item["item_id"],
            "category": item["category"],
            "text":     item["text"],
            "score":    float(sims[idx]),
        }

        if group in slots and slots[group] is None:
            slots[group] = entry
        elif len(extras) < top_k:
            extras.append(entry)

        if all(v is not None for v in slots.values()):
            break

    return slots


def show_outfit(slots):
    """Render the outfit grid with images, category, description and score."""
    filled = [(k, v) for k, v in slots.items() if v is not None]
    if not filled:
        st.warning("No compatible items found in the catalog.")
        return

    cols = st.columns(min(len(filled), 5))
    for i, (group, item) in enumerate(filled):
        with cols[i % 5]:
            st.markdown(f"**{OUTFIT_SLOTS[group]}**")
            img = get_item_image(item["item_id"])
            if img:
                st.image(img, use_container_width=True)
            else:
                st.markdown("*No image*")
            st.markdown(f"**{item['category']}**")
            st.caption(item["text"][:55] + "..." if len(item["text"]) > 55 else item["text"])
            st.progress(max(0.0, min(1.0, item["score"])))
            st.markdown(f"Score: `{item['score']:.3f}`")


# ── Outfit Scorer logic (Kishan) ─────────────────────────
def score_outfit(images, texts):
    """Score compatibility between all pairs of uploaded items."""
    model = load_model()

    embeddings = []
    for img, text in zip(images, texts):
        img_t = val_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_item(img_t, [text], DEVICE).squeeze(0).cpu()
        embeddings.append(emb)

    pair_scores = []
    pair_details = []
    for (i, j) in combinations(range(len(embeddings)), 2):
        emb_a = embeddings[i].to(DEVICE).unsqueeze(0)
        emb_b = embeddings[j].to(DEVICE).unsqueeze(0)
        diff = torch.abs(emb_a - emb_b)
        with torch.no_grad():
            score = model.compat_head(diff).squeeze().item()
        pair_scores.append(score)
        pair_details.append({
            "item_a": i, "item_b": j,
            "text_a": texts[i], "text_b": texts[j],
            "score": score,
        })

    overall_score = np.mean(pair_scores) if pair_scores else 0.0
    return overall_score, pair_details


def get_score_label(score):
    if score >= 0.75:
        return "Excellent match!", "#1D9E75"
    elif score >= 0.60:
        return "Good match", "#378ADD"
    elif score >= 0.45:
        return "Okay match", "#BA7517"
    else:
        return "Poor match", "#E24B4A"


# ══════════════════════════════════════════════════════════
#                    STREAMLIT UI
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fashion Outfit Recommender",
    page_icon="👗",
    layout="wide"
)

# Main navigation
page = st.sidebar.radio(
    "Mode",
    ["Recommend items", "Text search", "Outfit scorer"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** EfficientNet-B0 + Sentence-BERT\n\n"
    "**Dataset:** Polyvore Outfits\n\n"
    f"**Device:** `{DEVICE}`"
)


# ══════════════════════════════════════════════════════════
#         PAGE 1: RECOMMEND ITEMS (upload + camera)
# ══════════════════════════════════════════════════════════
if page == "Recommend items":
    st.title("Fashion Outfit Recommender")
    st.markdown("Upload a clothing item or scan one with your camera to get a complete outfit!")

    with st.sidebar:
        top_k = st.slider("Number of recommendations", 3, 10, 5)
        query_text_input = st.text_input(
            "Item description (helps filter results)",
            placeholder="e.g., polo shirt, blue denim jacket, summer dress"
        )
        st.markdown("---")
        st.markdown("### How it works")
        st.markdown(
            "1. Upload or scan a clothing item\n"
            "2. Model encodes it using EfficientNet + SBERT\n"
            "3. Compares against catalog via cosine similarity\n"
            "4. Builds a complete compatible outfit"
        )

    tab_upload, tab_camera, tab_text = st.tabs(["Upload image", "Camera scan","Text search"])

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
                dummy_img  = Image.new("RGB", (224, 224), (200, 200, 200))
                img_t      = val_transform(dummy_img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    query_emb = model.encode_item(img_t, [text_query], DEVICE).squeeze(0).cpu()
                slots = build_outfit(query_emb, query_group, embeddings, metadata, 7)

            st.markdown("---")
            st.markdown(f"## ✨ Complete Outfit for: *{text_query}*")
            st.markdown("### Complete the Look")
            show_outfit(slots)

        elif search_clicked and not text_query:
            st.warning("Please enter a description first!")

    if query_image:
        st.markdown("---")
        with st.spinner("Building your outfit..."):
            model = load_model()
            embeddings, metadata = load_catalog()

            # Detect category
            if query_text_input:
                query_group = detect_group_from_text(query_text_input)
                label = query_text_input
            else:
                detected = detect_category(query_image, model, embeddings, metadata)
                query_group = get_category_group(detected)
                label = detected

            # Embed query
            img_t = val_transform(query_image).unsqueeze(0).to(DEVICE)
            text = query_text_input if query_text_input else label
            with torch.no_grad():
                query_emb = model.encode_item(img_t, [text], DEVICE).squeeze(0).cpu()

            slots = build_outfit(query_emb, query_group, embeddings, metadata, top_k)

        col_query, col_results = st.columns([1, 3])
        with col_query:
            st.subheader("Your item")
            st.image(query_image, use_container_width=True)
            st.caption(f"Category: **{label}** ({query_group})")
            if query_text_input:
                st.caption(f"Description: *{query_text_input}*")

        with col_results:
            st.subheader("Complete the Look")
            show_outfit(slots)
    else:
        st.markdown("---")
        st.info("Upload an image or use the camera to scan a clothing item to get started!")


# ══════════════════════════════════════════════════════════
#         PAGE 2: TEXT SEARCH (Jahnavi)
# ══════════════════════════════════════════════════════════
elif page == "Text search":
    st.title("Build an Outfit from Text")
    st.markdown("Describe a clothing item and we'll build a complete outfit around it!")

    with st.sidebar:
        top_k = st.slider("Number of recommendations", 3, 10, 5)
        st.markdown("---")
        st.markdown("### How it works")
        st.markdown(
            "1. Describe a clothing item\n"
            "2. We find the best matching item in our catalog\n"
            "3. Build a complete outfit around it\n"
            "4. All items scored for compatibility"
        )

    text_query = st.text_input(
        "Describe your item",
        placeholder="e.g., wide leg blue jeans, black leather boots, floral summer dress"
    )
    search_clicked = st.button("Build Outfit", type="primary")

    if search_clicked and text_query:
        with st.spinner("Building your outfit..."):
            model = load_model()
            embeddings, metadata = load_catalog()
            query_group = detect_group_from_text(text_query)

            # Find best matching catalog image for the text
            with torch.no_grad():
                text_emb = model.text_encoder([text_query], DEVICE).squeeze(0).cpu()
            cat_matrix = torch.stack(embeddings)
            t_norm = text_emb / (text_emb.norm() + 1e-8)
            c_norm = cat_matrix / (cat_matrix.norm(dim=1, keepdim=True) + 1e-8)
            text_sims = (c_norm @ t_norm).numpy()
            best_idx = int(np.argmax(text_sims))
            best_img = get_item_image(metadata[best_idx]["item_id"])

            # Full multimodal embedding
            use_img = best_img if best_img else Image.new("RGB", (224, 224), (255, 255, 255))
            img_t = val_transform(use_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                query_emb = model.encode_item(img_t, [text_query], DEVICE).squeeze(0).cpu()

            slots = build_outfit(query_emb, query_group, embeddings, metadata, top_k)

        st.markdown("---")
        st.markdown(f"### Complete Outfit for: *{text_query}*")

        # Show detected query item
        q_col, _ = st.columns([1, 4])
        with q_col:
            st.subheader("Your Item")
            if best_img:
                st.image(best_img, use_container_width=True)
            st.markdown(f"**{metadata[best_idx]['category']}**")
            st.caption(text_query)
            if query_group != "unknown":
                st.caption(f"Detected: **{query_group}**")

        st.subheader("Complete the Look")
        show_outfit(slots)

    elif search_clicked and not text_query:
        st.warning("Please enter a description first!")
    else:
        st.markdown("---")
        st.info("Describe a clothing item above and click 'Build Outfit' to get started!")


# ══════════════════════════════════════════════════════════
#         PAGE 3: OUTFIT SCORER (Kishan)
# ══════════════════════════════════════════════════════════
elif page == "Outfit scorer":
    st.title("Outfit Fashion Score")
    st.markdown("Upload 2-6 clothing items and get a compatibility score for your outfit!")

    with st.sidebar:
        st.markdown("### How scoring works")
        st.markdown(
            "1. Upload photos of each clothing item\n"
            "2. The model scores every pair of items\n"
            "3. Overall score = average of all pair scores\n"
            "4. Higher score = more compatible outfit"
        )
        st.markdown("---")
        st.markdown("### Score guide")
        st.markdown(
            "**0.75+** — Excellent match!\n\n"
            "**0.60-0.75** — Good match\n\n"
            "**0.45-0.60** — Okay match\n\n"
            "**Below 0.45** — Poor match"
        )

    st.markdown("### Upload your outfit items")
    st.caption("Upload 2-6 clothing item photos (cropped product images work best)")

    upload_cols = st.columns(3)
    uploaded_items = []
    uploaded_files_list = []

    labels = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6"]
    for i in range(6):
        with upload_cols[i % 3]:
            file = st.file_uploader(
                labels[i],
                type=["jpg", "jpeg", "png", "webp"],
                key=f"outfit_item_{i}"
            )
            if file:
                img = Image.open(file).convert("RGB")
                uploaded_items.append(img)
                uploaded_files_list.append(i)
                st.image(img, use_container_width=True, caption=labels[i])

    if len(uploaded_items) >= 2:
        st.markdown("### Item descriptions (optional, improves accuracy)")
        desc_cols = st.columns(len(uploaded_items))
        texts = []
        for i in range(len(uploaded_items)):
            with desc_cols[i]:
                text = st.text_input(
                    f"Item {uploaded_files_list[i]+1} description",
                    placeholder="e.g., black pants",
                    key=f"desc_{i}",
                )
                texts.append(text if text else "fashion item")

        if st.button("Score my outfit!", type="primary", use_container_width=True):
            with st.spinner("Analyzing outfit compatibility..."):
                overall_score, pair_details = score_outfit(uploaded_items, texts)

            label, color = get_score_label(overall_score)

            st.markdown("---")
            st.markdown("### Overall outfit score")

            score_col1, score_col2 = st.columns([1, 2])
            with score_col1:
                st.markdown(
                    f"<div style='text-align: center; padding: 2rem; "
                    f"border: 3px solid {color}; border-radius: 16px;'>"
                    f"<div style='font-size: 48px; font-weight: bold; color: {color};'>"
                    f"{overall_score:.1%}</div>"
                    f"<div style='font-size: 18px; color: {color};'>{label}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with score_col2:
                st.markdown("#### Pairwise compatibility breakdown")
                for detail in sorted(pair_details, key=lambda x: -x["score"]):
                    pair_label = (f"Item {uploaded_files_list[detail['item_a']]+1} "
                                  f"and Item {uploaded_files_list[detail['item_b']]+1}")
                    pair_score = detail["score"]

                    col_a, col_b, col_c = st.columns([3, 1, 1])
                    with col_a:
                        st.markdown(f"**{pair_label}**: "
                                    f"*{detail['text_a'][:20]}* + *{detail['text_b'][:20]}*")
                    with col_b:
                        st.progress(max(0.0, min(1.0, pair_score)))
                    with col_c:
                        st.markdown(f"`{pair_score:.3f}`")

            st.markdown("---")
            if pair_details:
                worst = min(pair_details, key=lambda x: x["score"])
                best = max(pair_details, key=lambda x: x["score"])

                sug_col1, sug_col2 = st.columns(2)
                with sug_col1:
                    st.success(
                        f"**Best pair:** Item {uploaded_files_list[best['item_a']]+1} + "
                        f"Item {uploaded_files_list[best['item_b']]+1} "
                        f"(score: {best['score']:.3f})"
                    )
                with sug_col2:
                    st.warning(
                        f"**Weakest pair:** Item {uploaded_files_list[worst['item_a']]+1} + "
                        f"Item {uploaded_files_list[worst['item_b']]+1} "
                        f"(score: {worst['score']:.3f})"
                    )
                st.caption(
                    "Tip: Try replacing the weakest-scoring item "
                    "with something more compatible to improve the overall score!"
                )

    elif len(uploaded_items) == 1:
        st.info("Upload at least one more item to score the outfit!")
    else:
        st.info("Upload 2-6 clothing items to get started!")


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