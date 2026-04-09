"""
Evaluation script.
Metrics:
  1. AUC  — how well the model ranks compatible pairs above incompatible ones
  2. Accuracy — simple correct/incorrect on 0.5 threshold
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dataset import PolyvoreDataset
from model import FashionCompatibilityModel

CKPT_PATH  = "models/best_model.pt"
BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    return {
        "img_a":  torch.stack([b["img_a"]  for b in batch]),
        "img_b":  torch.stack([b["img_b"]  for b in batch]),
        "text_a": [b["text_a"] for b in batch],
        "text_b": [b["text_b"] for b in batch],
        "label":  torch.stack([b["label"]  for b in batch]),
    }


def evaluate():
    print(f"Using device: {DEVICE}")

    # Load validation set
    print("Loading val dataset...")
    val_ds = PolyvoreDataset("val")
    val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    # Load trained model
    print(f"Loading model from {CKPT_PATH}...")
    model = FashionCompatibilityModel().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    criterion = nn.BCELoss()
    all_scores, all_labels = [], []
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Evaluating"):
            img_a  = batch["img_a"].to(DEVICE)
            img_b  = batch["img_b"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            scores, _, _ = model(img_a, batch["text_a"],
                                 img_b, batch["text_b"], DEVICE)

            total_loss += criterion(scores, labels).item()
            preds       = (scores > 0.5).float()
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(val_dl)
    accuracy = correct / total * 100
    auc      = roc_auc_score(all_labels, all_scores)

    print(f"\n── Evaluation Results ──────────────────")
    print(f"  Val Loss : {avg_loss:.4f}")
    print(f"  Accuracy : {accuracy:.2f}%")
    print(f"  AUC      : {auc:.4f}")
    print(f"────────────────────────────────────────")
    print(f"\n  Random baseline accuracy : 50.00%")
    print(f"  Random baseline AUC      : 0.5000")
    print(f"  Our model is {accuracy - 50:.2f}% better than random!")


if __name__ == "__main__":
    evaluate()