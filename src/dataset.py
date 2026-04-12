"""
Dataset for Marqo/Polyvore fashion compatibility.
Items sharing the same outfit ID (the part before '_' in item_ID) are compatible.

v2 changes:
  - All within-outfit pairs as positives (not just consecutive)
  - Hard negative sampling: pick negatives from same category as positive partner
  - Configurable pairing strategy via pair_mode parameter
"""
import random
from itertools import combinations
from collections import defaultdict
from datasets import load_from_disk
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

DATA_PATH = "data/polyvore_outfits"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class PolyvoreDataset(Dataset):
    def __init__(self, split: str = "train", val_ratio: float = 0.1,
                 num_outfits: int = 5000, pair_mode: str = "all_pairs",
                 hard_negatives: bool = True):
        """
        Args:
            split: 'train' or 'val'
            val_ratio: fraction of outfits for validation
            num_outfits: number of outfits to use
            pair_mode: 'consecutive' (original) or 'all_pairs' (improved)
            hard_negatives: if True, sample negatives from same category
        """
        ds = load_from_disk(DATA_PATH)["data"]

        # Group items by outfit ID (part before '_' in item_ID)
        outfit_to_items = defaultdict(list)
        for idx, item_id in enumerate(ds["item_ID"]):
            outfit_id = item_id.split("_")[0]
            outfit_to_items[outfit_id].append(idx)

        outfit_ids = list(outfit_to_items.keys())
        outfit_ids = outfit_ids[:num_outfits]  # configurable subset size
        random.seed(42)
        random.shuffle(outfit_ids)

        # Split outfits into train / val
        split_point = int(len(outfit_ids) * (1 - val_ratio))
        if split == "train":
            selected = outfit_ids[:split_point]
        else:
            selected = outfit_ids[split_point:]

        self.ds = ds
        self.transform = train_transform if split == "train" else val_transform
        self.all_indices = list(range(len(ds)))

        # Build category-to-indices mapping for hard negative sampling
        self.cat_to_indices = defaultdict(list)
        if hard_negatives:
            for idx in range(len(ds)):
                cat = ds[idx]["category"]
                self.cat_to_indices[cat].append(idx)

        self.hard_negatives = hard_negatives

        # Collect all selected outfit items for exclusion checks
        selected_outfit_items = {}
        for oid in selected:
            items = outfit_to_items[oid]
            selected_outfit_items[oid] = set(items)

        # Build pairs from selected outfits
        self.pairs = []  # (idx_a, idx_b, label)
        for oid in selected:
            items = outfit_to_items[oid]
            if len(items) < 2:
                continue

            # ── Positive pairs ──────────────────────────────
            if pair_mode == "all_pairs":
                # All combinations within outfit
                pos_pairs = list(combinations(items, 2))
            else:
                # Original: consecutive only
                pos_pairs = [(items[i], items[i + 1]) for i in range(len(items) - 1)]

            for item_a, item_b in pos_pairs:
                self.pairs.append((item_a, item_b, 1))

            # ── Negative pairs (one per positive) ───────────
            items_set = set(items)
            for item_a, item_b in pos_pairs:
                if hard_negatives:
                    # Hard negative: find item from same category as item_b
                    # but from a different outfit
                    cat_b = ds[item_b]["category"]
                    candidates = self.cat_to_indices.get(cat_b, [])
                    neg = None
                    attempts = 0
                    while attempts < 20:
                        candidate = random.choice(candidates) if candidates else random.choice(self.all_indices)
                        if candidate not in items_set:
                            neg = candidate
                            break
                        attempts += 1
                    # Fallback to random if hard negative not found
                    if neg is None:
                        neg = random.choice(self.all_indices)
                        while neg in items_set:
                            neg = random.choice(self.all_indices)
                else:
                    # Original: random negative
                    neg = random.choice(self.all_indices)
                    while neg in items_set:
                        neg = random.choice(self.all_indices)

                self.pairs.append((item_a, neg, 0))

        print(f"[{split}] {len(self.pairs)} pairs from {len(selected)} outfits "
              f"(mode={pair_mode}, hard_neg={hard_negatives})")

    def _load(self, idx):
        row = self.ds[idx]
        img = row["image"].convert("RGB")
        img = self.transform(img)
        text = f"{row['category']}. {row['text']}".strip()
        return img, text

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ia, ib, label = self.pairs[idx]
        img_a, text_a = self._load(ia)
        img_b, text_b = self._load(ib)
        return {
            "img_a":  img_a,
            "img_b":  img_b,
            "text_a": text_a,
            "text_b": text_b,
            "label":  torch.tensor(label, dtype=torch.float32),
        }


if __name__ == "__main__":
    # Compare old vs new pair generation
    print("=== Original (consecutive + random negatives) ===")
    ds_old = PolyvoreDataset("train", num_outfits=100,
                             pair_mode="consecutive", hard_negatives=False)
    print(f"  Pairs: {len(ds_old)}")

    print("\n=== Improved (all pairs + hard negatives) ===")
    ds_new = PolyvoreDataset("train", num_outfits=100,
                             pair_mode="all_pairs", hard_negatives=True)
    print(f"  Pairs: {len(ds_new)}")
    print(f"  Increase: {len(ds_new)/len(ds_old):.1f}x more pairs")