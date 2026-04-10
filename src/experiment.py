"""
Experiment runner for tracking multiple training iterations.
Each experiment gets its own folder with config, logs, curves, and evaluation results.

Usage:
  python src/experiment.py --name baseline --outfits 5000 --epochs 5 --lr 1e-4
  python src/experiment.py --name frozen_backbone --outfits 5000 --epochs 5 --lr 1e-4 --freeze
  python src/experiment.py --name more_data --outfits 10000 --epochs 5 --lr 1e-4
  python src/experiment.py --name low_lr --outfits 5000 --epochs 5 --lr 1e-5
  python src/experiment.py --compare   ← generates comparison across all experiments
"""
import argparse
import csv
import json
import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from dataset import PolyvoreDataset
from model import FashionCompatibilityModel
from encoders import ImageEncoder

# ── Device ──────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

EXPERIMENTS_DIR = "experiments"


def collate_fn(batch):
    return {
        "img_a":  torch.stack([b["img_a"]  for b in batch]),
        "img_b":  torch.stack([b["img_b"]  for b in batch]),
        "text_a": [b["text_a"] for b in batch],
        "text_b": [b["text_b"] for b in batch],
        "label":  torch.stack([b["label"]  for b in batch]),
    }


def run_experiment(name, num_outfits, epochs, lr, batch_size, freeze_backbone,
                   dropout, weight_decay):
    """Run a full experiment: train → evaluate → generate insights."""

    # ── Setup experiment folder ─────────────────────────
    exp_dir = os.path.join(EXPERIMENTS_DIR, name)
    os.makedirs(exp_dir, exist_ok=True)

    config = {
        "name": name,
        "num_outfits": num_outfits,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "freeze_backbone": freeze_backbone,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "device": str(DEVICE),
        "embed_dim": 256,
        "image_encoder": "EfficientNet-B0",
        "text_encoder": "all-MiniLM-L6-v2 (frozen)",
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"  Outfits: {num_outfits} | Epochs: {epochs} | LR: {lr}")
    print(f"  Freeze backbone: {freeze_backbone} | Dropout: {dropout}")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {exp_dir}/")
    print(f"{'='*60}\n")

    # ── Load data ───────────────────────────────────────
    print("Loading datasets...")
    train_ds = PolyvoreDataset("train", num_outfits=num_outfits)
    val_ds   = PolyvoreDataset("val",   num_outfits=num_outfits)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                          num_workers=0, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size, shuffle=False,
                          num_workers=0, collate_fn=collate_fn)

    # ── Build model ─────────────────────────────────────
    model = FashionCompatibilityModel(dropout=dropout).to(DEVICE)

    # Optionally freeze backbone
    if freeze_backbone:
        for p in model.img_encoder.backbone.parameters():
            p.requires_grad = False
        print("  ❄️  Image backbone FROZEN")
    else:
        print("  🔥 Image backbone UNFROZEN (fine-tuning)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({trainable/total*100:.1f}%)\n")

    config["trainable_params"] = trainable
    config["total_params"] = total

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # ── Training loop ───────────────────────────────────
    log_path = os.path.join(exp_dir, "training_log.csv")
    model_path = os.path.join(exp_dir, "best_model.pt")
    best_val_loss = float("inf")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc", "lr", "time_sec"])

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ── Train ───────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs} [train]"):
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
            preds       = (scores > 0.5).float()
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

        train_loss /= len(train_dl)
        train_acc   = correct / total * 100

        # ── Validate ────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch}/{epochs} [val]  "):
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
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                             f"{val_loss:.4f}", f"{val_acc:.2f}",
                             f"{current_lr:.6f}", f"{epoch_time:.0f}"])

        print(f"\nEpoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model → {model_path}")

    total_time = time.time() - total_start
    print(f"\nTraining complete! Total time: {total_time:.0f}s")

    # ── Evaluation ──────────────────────────────────────
    print("\n── Running evaluation on best model ──")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_scores, all_labels = [], []
    eval_loss, eval_correct, eval_total = 0.0, 0, 0

    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Evaluating"):
            img_a  = batch["img_a"].to(DEVICE)
            img_b  = batch["img_b"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            scores, _, _ = model(img_a, batch["text_a"],
                                 img_b, batch["text_b"], DEVICE)
            eval_loss  += criterion(scores, labels).item()
            preds       = (scores > 0.5).float()
            eval_correct += (preds == labels).sum().item()
            eval_total   += labels.size(0)

            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    eval_loss /= len(val_dl)
    accuracy   = eval_correct / eval_total * 100
    auc        = roc_auc_score(all_labels, all_scores)

    eval_results = {
        "val_loss": round(eval_loss, 4),
        "accuracy": round(accuracy, 2),
        "auc": round(auc, 4),
        "total_training_time_sec": round(total_time, 0),
        "best_epoch": int(np.argmin([float(r) for r in
            open(log_path).readlines()[1:]])) + 1 if os.path.exists(log_path) else -1,
    }

    # Recalculate best epoch properly
    import pandas as pd
    df = pd.read_csv(log_path)
    best_epoch = int(df.loc[df["val_loss"].astype(float).idxmin(), "epoch"])
    eval_results["best_epoch"] = best_epoch

    with open(os.path.join(exp_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n── Evaluation Results ({name}) ──────────")
    print(f"  Val Loss : {eval_loss:.4f}")
    print(f"  Accuracy : {accuracy:.2f}%")
    print(f"  AUC      : {auc:.4f}")
    print(f"  Best epoch: {best_epoch}")
    print(f"──────────────────────────────────────────")

    # ── Generate insight plots ──────────────────────────
    generate_training_curves(exp_dir, name)
    generate_roc_curve(all_labels, all_scores, exp_dir, name, auc)
    generate_score_distribution(all_labels, all_scores, exp_dir, name)

    print(f"\n✓ All insights saved to {exp_dir}/")
    return eval_results


def generate_training_curves(exp_dir, name):
    """Generate loss and accuracy curves from training log."""
    import pandas as pd
    df = pd.read_csv(os.path.join(exp_dir, "training_log.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Training curves — {name}", fontsize=16, fontweight='bold', y=1.02)

    # Loss
    ax1.plot(df["epoch"], df["train_loss"].astype(float), 'o-',
             color='#378ADD', linewidth=2.5, markersize=8, label='Train loss')
    ax1.plot(df["epoch"], df["val_loss"].astype(float), 's--',
             color='#D85A30', linewidth=2.5, markersize=8, label='Val loss')
    best_epoch = df.loc[df["val_loss"].astype(float).idxmin(), "epoch"]
    ax1.axvline(x=best_epoch, color='#1D9E75', linestyle=':', linewidth=1.5,
                alpha=0.7, label=f'Best model (epoch {int(best_epoch)})')
    ax1.fill_between(df["epoch"], df["train_loss"].astype(float),
                     df["val_loss"].astype(float), alpha=0.08, color='#D85A30')
    for i, row in df.iterrows():
        ax1.annotate(f'{float(row["train_loss"]):.3f}', (row["epoch"], float(row["train_loss"])),
                     textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9, color='#378ADD')
        ax1.annotate(f'{float(row["val_loss"]):.3f}', (row["epoch"], float(row["val_loss"])),
                     textcoords="offset points", xytext=(0, -16), ha='center', fontsize=9, color='#D85A30')
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss (BCE)', fontsize=13, fontweight='bold')
    ax1.set_title('Loss curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.set_xticks(df["epoch"].tolist())
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(df["epoch"], df["train_acc"].astype(float), 'o-',
             color='#378ADD', linewidth=2.5, markersize=8, label='Train accuracy')
    ax2.plot(df["epoch"], df["val_acc"].astype(float), 's--',
             color='#D85A30', linewidth=2.5, markersize=8, label='Val accuracy')
    ax2.axhline(y=50, color='#888780', linestyle=':', linewidth=1.5,
                alpha=0.5, label='Random baseline (50%)')
    ax2.fill_between(df["epoch"], df["train_acc"].astype(float),
                     df["val_acc"].astype(float), alpha=0.08, color='#D85A30')
    for i, row in df.iterrows():
        ax2.annotate(f'{float(row["train_acc"]):.1f}%', (row["epoch"], float(row["train_acc"])),
                     textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9, color='#378ADD')
        ax2.annotate(f'{float(row["val_acc"]):.1f}%', (row["epoch"], float(row["val_acc"])),
                     textcoords="offset points", xytext=(0, -16), ha='center', fontsize=9, color='#D85A30')
    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Accuracy curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.set_xticks(df["epoch"].tolist())
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "training_curves.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved training_curves.png")


def generate_roc_curve(labels, scores, exp_dir, name, auc):
    """Generate ROC curve plot."""
    fpr, tpr, _ = roc_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color='#378ADD', linewidth=2.5, label=f'Model (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#888780', linestyle='--', linewidth=1.5, label='Random (AUC = 0.5)')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#378ADD')
    ax.set_xlabel('False positive rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True positive rate', fontsize=13, fontweight='bold')
    ax.set_title(f'ROC curve — {name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "roc_curve.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved roc_curve.png")


def generate_score_distribution(labels, scores, exp_dir, name):
    """Generate histogram of compatibility scores for positive vs negative pairs."""
    pos_scores = [s for s, l in zip(scores, labels) if l == 1.0]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0.0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pos_scores, bins=30, alpha=0.6, color='#1D9E75', label='Compatible pairs', edgecolor='white')
    ax.hist(neg_scores, bins=30, alpha=0.6, color='#D85A30', label='Incompatible pairs', edgecolor='white')
    ax.axvline(x=0.5, color='#378ADD', linestyle='--', linewidth=2, label='Decision threshold (0.5)')

    ax.set_xlabel('Predicted compatibility score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax.set_title(f'Score distribution — {name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add stats
    overlap_info = (f"Compatible mean: {np.mean(pos_scores):.3f} | "
                    f"Incompatible mean: {np.mean(neg_scores):.3f} | "
                    f"Separation: {np.mean(pos_scores) - np.mean(neg_scores):.3f}")
    ax.text(0.5, -0.12, overlap_info, transform=ax.transAxes, ha='center',
            fontsize=11, color='#5F5E5A')

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "score_distribution.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved score_distribution.png")


def compare_experiments():
    """Generate comparison charts across all experiments."""
    import pandas as pd

    if not os.path.exists(EXPERIMENTS_DIR):
        print("No experiments found!")
        return

    exp_names = sorted([d for d in os.listdir(EXPERIMENTS_DIR)
                        if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))])

    if len(exp_names) < 2:
        print("Need at least 2 experiments to compare.")
        return

    # Collect results
    results = []
    for name in exp_names:
        eval_path = os.path.join(EXPERIMENTS_DIR, name, "eval_results.json")
        config_path = os.path.join(EXPERIMENTS_DIR, name, "config.json")
        if not os.path.exists(eval_path):
            continue
        with open(eval_path) as f:
            eval_data = json.load(f)
        with open(config_path) as f:
            config_data = json.load(f)
        results.append({
            "name": name,
            "accuracy": eval_data["accuracy"],
            "auc": eval_data["auc"],
            "val_loss": eval_data["val_loss"],
            "best_epoch": eval_data.get("best_epoch", "?"),
            "num_outfits": config_data["num_outfits"],
            "lr": config_data["learning_rate"],
            "freeze": config_data["freeze_backbone"],
            "dropout": config_data["dropout"],
            "training_time": eval_data.get("total_training_time_sec", 0),
        })

    if len(results) < 2:
        print("Need at least 2 completed experiments to compare.")
        return

    df = pd.DataFrame(results)

    # ── Comparison bar charts ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Experiment comparison", fontsize=18, fontweight='bold', y=1.02)

    colors = ['#378ADD', '#D85A30', '#1D9E75', '#7F77DD', '#D4537E']

    # Accuracy comparison
    bars = axes[0].bar(range(len(df)), df["accuracy"],
                       color=colors[:len(df)], edgecolor='white', linewidth=1.5)
    axes[0].set_xticks(range(len(df)))
    axes[0].set_xticklabels(df["name"], rotation=30, ha='right', fontsize=10)
    axes[0].set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    axes[0].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0].axhline(y=50, color='#888780', linestyle=':', alpha=0.5, label='Random')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, df["accuracy"]):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    # AUC comparison
    bars = axes[1].bar(range(len(df)), df["auc"],
                       color=colors[:len(df)], edgecolor='white', linewidth=1.5)
    axes[1].set_xticks(range(len(df)))
    axes[1].set_xticklabels(df["name"], rotation=30, ha='right', fontsize=10)
    axes[1].set_ylabel('AUC', fontsize=13, fontweight='bold')
    axes[1].set_title('AUC (area under ROC curve)', fontsize=14, fontweight='bold')
    axes[1].axhline(y=0.5, color='#888780', linestyle=':', alpha=0.5, label='Random')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, df["auc"]):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')

    # Val loss comparison
    bars = axes[2].bar(range(len(df)), df["val_loss"],
                       color=colors[:len(df)], edgecolor='white', linewidth=1.5)
    axes[2].set_xticks(range(len(df)))
    axes[2].set_xticklabels(df["name"], rotation=30, ha='right', fontsize=10)
    axes[2].set_ylabel('Validation loss', fontsize=13, fontweight='bold')
    axes[2].set_title('Val loss (lower is better)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, df["val_loss"]):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    compare_path = os.path.join(EXPERIMENTS_DIR, "comparison.png")
    plt.savefig(compare_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Comparison chart saved to {compare_path}")

    # ── Overlay training curves ─────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Training curves — all experiments", fontsize=16, fontweight='bold', y=1.02)

    for i, name in enumerate(df["name"]):
        log_path = os.path.join(EXPERIMENTS_DIR, name, "training_log.csv")
        if not os.path.exists(log_path):
            continue
        log_df = pd.read_csv(log_path)
        color = colors[i % len(colors)]
        ax1.plot(log_df["epoch"], log_df["val_loss"].astype(float),
                 'o-', color=color, linewidth=2, markersize=6, label=name)
        ax2.plot(log_df["epoch"], log_df["val_acc"].astype(float),
                 'o-', color=color, linewidth=2, markersize=6, label=name)

    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Validation loss', fontsize=13, fontweight='bold')
    ax1.set_title('Val loss across experiments', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Validation accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Val accuracy across experiments', fontsize=14, fontweight='bold')
    ax2.axhline(y=50, color='#888780', linestyle=':', alpha=0.5)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    overlay_path = os.path.join(EXPERIMENTS_DIR, "curves_overlay.png")
    plt.savefig(overlay_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Curves overlay saved to {overlay_path}")

    # ── Summary table ───────────────────────────────────
    summary_path = os.path.join(EXPERIMENTS_DIR, "summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"✓ Summary table saved to {summary_path}")

    print(f"\n{'='*60}")
    print("  EXPERIMENT COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Name':<25} {'Acc':>7} {'AUC':>8} {'ValLoss':>8} {'BestEp':>7}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{row['name']:<25} {row['accuracy']:>6.2f}% {row['auc']:>8.4f} "
              f"{row['val_loss']:>8.4f} {row['best_epoch']:>7}")
    best_idx = df["auc"].idxmax()
    print(f"\n  🏆 Best experiment: {df.iloc[best_idx]['name']} "
          f"(AUC: {df.iloc[best_idx]['auc']:.4f}, "
          f"Acc: {df.iloc[best_idx]['accuracy']:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fashion ML experiments")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--outfits", type=int, default=5000, help="Number of outfits")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--freeze", action="store_true", help="Freeze image backbone")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--compare", action="store_true", help="Compare all experiments")

    args = parser.parse_args()

    if args.compare:
        compare_experiments()
    elif args.name:
        run_experiment(
            name=args.name,
            num_outfits=args.outfits,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            freeze_backbone=args.freeze,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
        )
    else:
        parser.print_help()
