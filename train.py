"""
Training script for Crop Disease Detector.

Two-phase training strategy:
  Phase 1: Frozen backbone — train only the classifier head (fast convergence)
  Phase 2: Unfreeze last 2 blocks — fine-tune with a much smaller learning rate

All metrics logged to Weights & Biases (W&B).

Usage:
    python train.py
"""

import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix, classification_report

from model import create_model, unfreeze_last_n_blocks
from data_setup import create_dataloaders


# ──────────────────────────────────────────────────
# CONFIG — tweak these for your experiments
# ──────────────────────────────────────────────────
CONFIG = {
    # Data
    "data_dir": "data",
    "batch_size": 32,
    "num_workers": 2,
    "num_classes": 38,

    # Phase 1 — frozen backbone
    "phase1_epochs": 5,
    "phase1_lr": 1e-3,

    # Phase 2 — unfrozen last blocks
    "phase2_epochs": 10,
    "phase2_lr_head": 1e-4,      # Lower LR for head in phase 2
    "phase2_lr_backbone": 1e-5,  # Much lower LR for unfrozen backbone layers
    "unfreeze_blocks": 2,

    # Scheduler
    "scheduler_patience": 2,
    "scheduler_factor": 0.5,

    # Saving
    "save_dir": "checkpoints",
    "project_name": "crop-disease-detector",
}


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100. * correct / total:.1f}%"
        })

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch):
    """Validate the model. Returns average loss, accuracy, all predictions and labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [VAL]  ", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100. * correct / total:.1f}%"
        })

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy, all_preds, all_labels


def log_confusion_matrix(all_preds, all_labels, class_names, epoch):
    """Log confusion matrix to W&B."""
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=class_names
        ),
        "epoch": epoch,
    })


def log_sample_predictions(model, val_loader, class_names, device, num_samples=20):
    """Log a table of sample predictions to W&B — shows what the model gets right and wrong."""
    model.eval()
    table = wandb.Table(columns=["Image", "True Label", "Predicted", "Correct", "Confidence"])

    images_logged = 0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = probs.max(1)

            for i in range(len(images)):
                if images_logged >= num_samples:
                    break

                # Denormalize image for display
                img = images[i] * std + mean
                img = img.clamp(0, 1).cpu()

                true_label = class_names[labels[i]].replace("___", " - ").replace("_", " ")
                pred_label = class_names[predicted[i]].replace("___", " - ").replace("_", " ")
                correct = "✅" if predicted[i] == labels[i] else "❌"

                table.add_data(
                    wandb.Image(img),
                    true_label,
                    pred_label,
                    correct,
                    f"{confidences[i].item():.2%}"
                )
                images_logged += 1

            if images_logged >= num_samples:
                break

    wandb.log({"sample_predictions": table})


def train(config):
    """Full two-phase training pipeline with W&B logging."""

    # ── Setup ──
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Initialize W&B
    run = wandb.init(
        project=config["project_name"],
        config=config,
        name=f"efficientnet-b0-{time.strftime('%m%d-%H%M')}",
    )

    # Data
    train_loader, val_loader, class_names = create_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    # Model
    model = create_model(num_classes=config["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()

    # Save directory
    os.makedirs(config["save_dir"], exist_ok=True)
    best_val_acc = 0.0

    # ═════════════════════════════════════════════
    # PHASE 1 — Frozen backbone, train head only
    # ═════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🔒 PHASE 1 — Training classifier head only (backbone frozen)")
    print("=" * 60)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["phase1_lr"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config["scheduler_patience"],
        factor=config["scheduler_factor"]
    )

    for epoch in range(1, config["phase1_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device, epoch)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # Log to W&B
        wandb.log({
            "phase": 1,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": current_lr,
        })

        print(f"  Phase 1 | Epoch {epoch}/{config['phase1_epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | LR: {current_lr:.2e}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config["save_dir"], "best_model.pth"))
            print(f"  💾 Saved best model — Val Acc: {val_acc:.1f}%")

    # Log confusion matrix after Phase 1
    log_confusion_matrix(all_preds, all_labels, class_names, epoch)

    # ═════════════════════════════════════════════
    # PHASE 2 — Unfreeze last blocks, fine-tune
    # ═════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"🔓 PHASE 2 — Unfreezing last {config['unfreeze_blocks']} blocks for fine-tuning")
    print("=" * 60)

    unfreeze_last_n_blocks(model, n=config["unfreeze_blocks"])

    # New optimizer with different LRs for backbone vs head
    optimizer = Adam([
        {"params": model.features.parameters(), "lr": config["phase2_lr_backbone"]},
        {"params": model.classifier.parameters(), "lr": config["phase2_lr_head"]},
    ])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config["scheduler_patience"],
        factor=config["scheduler_factor"]
    )

    total_phase1 = config["phase1_epochs"]

    for epoch in range(1, config["phase2_epochs"] + 1):
        global_epoch = total_phase1 + epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, global_epoch)
        val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device, global_epoch)
        scheduler.step(val_loss)

        backbone_lr = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"]

        # Log to W&B
        wandb.log({
            "phase": 2,
            "epoch": global_epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate_backbone": backbone_lr,
            "learning_rate_head": head_lr,
        })

        print(f"  Phase 2 | Epoch {epoch}/{config['phase2_epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
              f"LR(bb): {backbone_lr:.2e} | LR(head): {head_lr:.2e}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config["save_dir"], "best_model.pth"))
            print(f"  💾 Saved best model — Val Acc: {val_acc:.1f}%")

    # ── Final logging ──
    print("\n" + "=" * 60)
    print(f"🏁 Training complete! Best Val Accuracy: {best_val_acc:.1f}%")
    print("=" * 60)

    # Log final confusion matrix and sample predictions
    log_confusion_matrix(all_preds, all_labels, class_names, global_epoch)
    log_sample_predictions(model, val_loader, class_names, device, num_samples=20)

    wandb.log({"best_val_acc": best_val_acc})
    wandb.finish()

    print(f"\n✅ Model saved to: {config['save_dir']}/best_model.pth")
    print(f"✅ W&B dashboard: {run.url}")


if __name__ == "__main__":
    train(CONFIG)
