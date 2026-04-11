"""
Fashion Outfit Recommendation — Web App
Features:
  1. Upload/scan a clothing item → get compatible recommendations
  2. Upload multiple items → get outfit fashion score

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

# ── Category mapping ────────────────────────────────────
TOPS = ["Tops", "Blouses", "T-Shirts", "Tank Tops", "Sweaters", "Sweatshirts"]
OUTERWEAR = ["Jackets", "Coats"]
BOTTOMS = ["Pants", "Skinny Jeans", "Shorts", "Knee Length Skirts", "Leggings"]
DRESSES = ["Day Dresses", "Cocktail Dresses", "Maxi Dresses"]
SHOES = ["Sandals", "Pumps", "Ankle Booties", "Sneakers", "Boots", "Flats"]
BAGS = ["Shoulder Bags", "Clutches", "Handbags", "Tote Bags", "Backpacks"]
JEWELRY = ["Earrings", "Necklaces", "Bracelets & Bangles", "Rings"]
ACCESSORIES = ["Sunglasses", "Hats", "Watches", "Belts", "Scarves"]

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
        st.error("Catalog embeddings not found! Run `python src/recommend.py` first.")
        st.stop()


@st.cache_resource
def load_dataset_images():
    ds = load_from_disk(DATA_PATH)["data"]
    return ds


# ── Auto-detect category ────────────────────────────────
def detect_category(query_img: Image.Image, model, embeddings, metadata):
    img_t = val_transform(query_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        query_emb = model.encode_item(img_t, ["fashion item"], DEVICE).squeeze(0).cpu()

    cat_matrix = torch.stack(embeddings)
    q_norm = query_emb / (query_emb.norm() + 1e-8)
    c_norm = cat_matrix / (cat_matrix.norm(dim=1, keepdim=True) + 1e-8)
    sims = (c_norm @ q_norm).numpy()

    top_indices = np.argsort(sims)[::-1][:10]
    categories = [metadata[idx]["category"] for idx in top_indices]

    counts = Counter(categories).most_common()
    for cat, count in counts:
        if cat != "Clothing":
            return cat
    return counts[0][0]


# ── Recommendation logic ────────────────────────────────
def get_recommendations(query_img: Image.Image, query_text: str, top_k: int):
    model = load_model()
    embeddings, metadata = load_catalog()

    img_t = val_transform(query_img).unsqueeze(0).to(DEVICE)
    text = query_text if query_text else "fashion item"

    with torch.no_grad():
        query_emb = model.encode_item(img_t, [text], DEVICE).squeeze(0).cpu()

    cat_matrix = torch.stack(embeddings).to(DEVICE)
    query_exp = query_emb.to(DEVICE).unsqueeze(0).expand(cat_matrix.size(0), -1)

    batch_size = 256
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(cat_matrix), batch_size):
            batch_cat = cat_matrix[i:i+batch_size]
            batch_query = query_exp[i:i+batch_size]
            diff = torch.abs(batch_query - batch_cat)
            scores = model.compat_head(diff).squeeze(1)
            all_scores.append(scores.cpu())
    sims = torch.cat(all_scores).numpy()

    sorted_indices = np.argsort(sims)[::-1]

    # Detect query category
    query_group = None
    if query_text:
        for cat in TOPS + OUTERWEAR + BOTTOMS + DRESSES + SHOES + BAGS + JEWELRY + ACCESSORIES:
            if cat.lower() in query_text.lower():
                query_group = get_category_group(cat)
                break
        if not query_group:
            qt = query_text.lower()
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

    if not query_group:
        detected = detect_category(query_img, model, embeddings, metadata)
        query_group = get_category_group(detected)
        query_text = detected

    # Build outfit — one per category
    outfit_slots = {
        "tops": None, "bottoms": None, "shoes": None,
        "outerwear": None, "bags": None, "jewelry": None, "accessories": None,
    }
    if query_group in outfit_slots:
        del outfit_slots[query_group]

    results = []
    extras = []
    compatible_cats = COMPATIBLE.get(query_group, [])

    for idx in sorted_indices:
        item = metadata[idx]
        item_group = get_category_group(item["category"])

        if item_group == query_group:
            continue
        if compatible_cats and item["category"] not in compatible_cats:
            continue

        entry = {
            "item_id": item["item_id"],
            "category": item["category"],
            "text": item["text"],
            "score": float(sims[idx]),
            "catalog_idx": idx,
        }

        if item_group in outfit_slots and outfit_slots[item_group] is None:
            outfit_slots[item_group] = entry
            results.append(entry)
        elif len(extras) < top_k:
            extras.append(entry)

        if len(results) >= top_k:
            break

    while len(results) < top_k and extras:
        results.append(extras.pop(0))

    return results, query_text, query_group


def get_item_image(item_id: str):
    ds = load_dataset_images()
    for i in range(len(ds)):
        if ds[i]["item_ID"] == item_id:
            return ds[i]["image"].convert("RGB")
    return None


# ── Outfit Scorer logic ─────────────────────────────────
def score_outfit(images, texts):
    """Score compatibility between all pairs of uploaded items."""
    model = load_model()

    # Encode all items
    embeddings = []
    for img, text in zip(images, texts):
        img_t = val_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_item(img_t, [text], DEVICE).squeeze(0).cpu()
        embeddings.append(emb)

    # Score all pairs using the compatibility head
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
            "item_a": i,
            "item_b": j,
            "text_a": texts[i],
            "text_b": texts[j],
            "score": score,
        })

    overall_score = np.mean(pair_scores) if pair_scores else 0.0
    return overall_score, pair_details


def get_score_label(score):
    """Return a label and color for the fashion score."""
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
    ["Recommend items", "Outfit scorer"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** EfficientNet-B0 + Sentence-BERT\n\n"
    "**Dataset:** Polyvore Outfits\n\n"
    f"**Device:** `{DEVICE}`"
)

# ══════════════════════════════════════════════════════════
#              PAGE 1: RECOMMEND ITEMS
# ══════════════════════════════════════════════════════════
if page == "Recommend items":
    st.title("Fashion Outfit Recommender")
    st.markdown("Upload a clothing item or scan one with your camera to get compatible outfit recommendations!")

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
            "2. The model encodes it using EfficientNet + SBERT\n"
            "3. Compares against catalog items via cosine similarity\n"
            "4. Returns the most compatible items"
        )

    tab_upload, tab_camera = st.tabs(["Upload image", "Camera scan"])

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

    if query_image:
        st.markdown("---")
        col_query, col_results = st.columns([1, 3])

        with col_query:
            st.subheader("Your item")
            st.image(query_image, use_container_width=True)

        with col_results:
            st.subheader(f"Top {top_k} compatible items")
            with st.spinner("Finding compatible items..."):
                results, detected_text, detected_group = get_recommendations(
                    query_image, query_text_input, top_k
                )

        with col_query:
            if query_text_input:
                st.caption(f"Description: *{query_text_input}*")
            else:
                st.caption(f"Detected category: **{detected_text}** ({detected_group})")

        with col_results:
            cols = st.columns(min(top_k, 5))
            for i, item in enumerate(results[:5]):
                with cols[i % 5]:
                    item_img = get_item_image(item["item_id"])
                    if item_img:
                        st.image(item_img, use_container_width=True)
                    else:
                        st.markdown("*Image not available*")
                    st.progress(max(0.0, min(1.0, item["score"])))
                    st.markdown(f"**{item['category']}**")
                    st.caption(item["text"][:60] + "..." if len(item["text"]) > 60 else item["text"])
                    st.markdown(f"Score: `{item['score']:.3f}`")

            if top_k > 5 and len(results) > 5:
                cols2 = st.columns(min(top_k - 5, 5))
                for i, item in enumerate(results[5:]):
                    with cols2[i % 5]:
                        item_img = get_item_image(item["item_id"])
                        if item_img:
                            st.image(item_img, use_container_width=True)
                        else:
                            st.markdown("*Image not available*")
                        st.progress(max(0.0, min(1.0, item["score"])))
                        st.markdown(f"**{item['category']}**")
                        st.caption(item["text"][:60] + "..." if len(item["text"]) > 60 else item["text"])
                        st.markdown(f"Score: `{item['score']:.3f}`")
    else:
        st.markdown("---")
        st.info("Upload an image or use the camera to scan a clothing item to get started!")


# ══════════════════════════════════════════════════════════
#              PAGE 2: OUTFIT SCORER
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

    # File uploaders in columns
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

    # Optional text descriptions and scoring
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

        # Score button
        if st.button("Score my outfit!", type="primary", use_container_width=True):
            with st.spinner("Analyzing outfit compatibility..."):
                overall_score, pair_details = score_outfit(uploaded_items, texts)

            label, color = get_score_label(overall_score)

            # Display overall score
            st.markdown("---")
            st.markdown("### Overall outfit score")

            score_col1, score_col2 = st.columns([1, 2])
            with score_col1:
                st.markdown(
                    f"<div style='text-align: center; padding: 2rem; "
                    f"border: 3px solid {color}; "
                    f"border-radius: 16px;'>"
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
                    _, pair_color = get_score_label(pair_score)

                    col_a, col_b, col_c = st.columns([3, 1, 1])
                    with col_a:
                        st.markdown(f"**{pair_label}**: "
                                    f"*{detail['text_a'][:20]}* + *{detail['text_b'][:20]}*")
                    with col_b:
                        st.progress(max(0.0, min(1.0, pair_score)))
                    with col_c:
                        st.markdown(f"`{pair_score:.3f}`")

            # Suggestions
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
