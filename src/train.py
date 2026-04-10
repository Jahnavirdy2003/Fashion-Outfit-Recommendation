"""
Training loop for FashionCompatibilityModel.
Loss: Binary Cross-Entropy (compatible=1, incompatible=0)
"""
import os
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import PolyvoreDataset
from model import FashionCompatibilityModel

# ── Config ───────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 5
LR         = 1e-4
SAVE_PATH  = "models/best_model.pt"
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    return {
        "img_a":  torch.stack([b["img_a"]  for b in batch]),
        "img_b":  torch.stack([b["img_b"]  for b in batch]),
        "text_a": [b["text_a"] for b in batch],
        "text_b": [b["text_b"] for b in batch],
        "label":  torch.stack([b["label"]  for b in batch]),
    }


def train():
    print(f"Using device: {DEVICE}")
    os.makedirs("models", exist_ok=True)

    print("Loading datasets...")
    train_ds = PolyvoreDataset("train")
    val_ds   = PolyvoreDataset("val")

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                          num_workers=0, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                          num_workers=0, collate_fn=collate_fn)

    model     = FashionCompatibilityModel().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best_val_loss = float("inf")
    total_batches = len(train_dl) * EPOCHS
    print(f"\nStarting training: {EPOCHS} epochs * {len(train_dl)} batches = {total_batches} total batches\n")
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch}/{EPOCHS}  ({(epoch-1)/EPOCHS*100:.0f}% done)")
        print(f"{'='*50}")
        # ── Train ────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            img_a  = batch["img_a"].to(DEVICE)
            img_b  = batch["img_b"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            scores, _, _ = model(img_a, batch["text_a"],
                                 img_b, batch["text_b"], DEVICE)
            loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            tqdm.write(f"  Batch loss: {loss.item():.4f}")
            preds       = (scores > 0.5).float()
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

        train_loss /= len(train_dl)
        train_acc   = correct / total * 100

        # ── Validate ─────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch}/{EPOCHS} [val]  "):
                img_a  = batch["img_a"].to(DEVICE)
                img_b  = batch["img_b"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                scores, _, _ = model(img_a, batch["text_a"],
                                     img_b, batch["text_b"], DEVICE)
                val_loss   += criterion(scores, labels).item()
                preds       = (scores > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_loss /= len(val_dl)
        val_acc   = val_correct / val_total * 100
        scheduler.step()

        print(f"\nEpoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Saved best model → {SAVE_PATH}")
        elapsed = time.time() - start
        avg_per_epoch = elapsed / epoch
        remaining = avg_per_epoch * (EPOCHS - epoch)
        print(f"  Time elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s")
    print("\nTraining complete!")


if __name__ == "__main__":
    train()