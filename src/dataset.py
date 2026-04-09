"""
Dataset for Marqo/Polyvore fashion compatibility.
Items sharing the same outfit ID (the part before '_' in item_ID) are compatible.
"""
import random
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
    def __init__(self, split: str = "train", val_ratio: float = 0.1):
        ds = load_from_disk(DATA_PATH)["data"]

        # Group items by outfit ID (part before '_' in item_ID)
        outfit_to_items = defaultdict(list)
        for idx, item_id in enumerate(ds["item_ID"]):
            outfit_id = item_id.split("_")[0]
            outfit_to_items[outfit_id].append(idx)

        outfit_ids = list(outfit_to_items.keys())
        outfit_ids = outfit_ids[:500]  # use only 500 outfits for fast training
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

        # Build pairs from selected outfits
        self.pairs = []  # (idx_a, idx_b, label)
        for oid in selected:
            items = outfit_to_items[oid]
            if len(items) < 2:
                continue
            # Positive pairs
            for i in range(len(items) - 1):
                self.pairs.append((items[i], items[i + 1], 1))
            # Negative pairs — random item from a different outfit
            for i in range(len(items) - 1):
                neg = random.choice(self.all_indices)
                while neg in items:
                    neg = random.choice(self.all_indices)
                self.pairs.append((items[i], neg, 0))

        print(f"[{split}] {len(self.pairs)} pairs from {len(selected)} outfits")

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
    # Quick test
    ds = PolyvoreDataset("train")
    sample = ds[0]
    print("img_a shape :", sample["img_a"].shape)
    print("text_a      :", sample["text_a"])
    print("text_b      :", sample["text_b"])
    print("label       :", sample["label"])