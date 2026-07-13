import argparse
import os
import time
import json

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import UNet, CombinedLoss
from dataset import get_dataloaders
from metrics import evaluate_batch


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_metrics = {"dice": [], "iou": [], "accuracy": [], "sensitivity": [], "specificity": []}

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_m = evaluate_batch(logits.detach(), masks)
        for k, v in batch_m.items():
            all_metrics[k].append(v)

    return total_loss / len(loader), {k: float(np.mean(v)) for k, v in all_metrics.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_metrics = {"dice": [], "iou": [], "accuracy": [], "sensitivity": [], "specificity": []}

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        batch_m = evaluate_batch(logits, masks)
        for k, v in batch_m.items():
            all_metrics[k].append(v)

    return total_loss / len(loader), {k: float(np.mean(v)) for k, v in all_metrics.items()}


@torch.no_grad()
def save_predictions(model, loader, device, output_dir, n_samples=6):
    model.eval()
    images, masks = next(iter(loader))
    images = images[:n_samples].to(device)
    masks = masks[:n_samples]
    logits = model(images)
    preds = logits.argmax(dim=1).cpu()

    fig, axes = plt.subplots(n_samples, 3, figsize=(10, n_samples * 3))
    titles = ["MRI Slice", "Ground Truth", "Prediction"]

    for i in range(n_samples):
        img = images[i, 0].cpu().numpy()
        gt = masks[i].numpy()
        pred = preds[i].numpy()

        for j, (data, title) in enumerate(zip([img, gt, pred], titles)):
            ax = axes[i, j]
            cmap = "gray" if j == 0 else "hot"
            ax.imshow(data, cmap=cmap)
            if i == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            ax.axis("off")

    plt.suptitle("Brain Tumor Segmentation Results", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "predictions.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def plot_training_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_dice"], "b-o", label="Train Dice", markersize=4)
    ax2.plot(epochs, history["val_dice"], "r-o", label="Val Dice", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Dice Score over Epochs")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Segmentation Training")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "kaggle"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(
        dataset_type=args.dataset,
        root=args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = UNet(in_channels=1, num_classes=2).to(device)
    criterion = CombinedLoss(dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    history = {k: [] for k in ["train_loss", "val_loss", "train_dice", "val_dice"]}
    best_val_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_m = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_m["dice"])
        history["val_dice"].append(val_m["dice"])

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Loss {train_loss:.4f}/{val_loss:.4f} | "
            f"Dice {train_m['dice']:.4f}/{val_m['dice']:.4f} | "
            f"IoU {val_m['iou']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_m["dice"] > best_val_dice:
            best_val_dice = val_m["dice"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice": best_val_dice,
                },
                os.path.join(args.output_dir, "best_model.pth"),
            )

    print("\n=== Final Evaluation (best checkpoint) ===")
    ckpt = torch.load(os.path.join(args.output_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, final_m = validate(model, val_loader, criterion, device)
    print(f"  Dice:        {final_m['dice']:.4f}")
    print(f"  IoU:         {final_m['iou']:.4f}")
    print(f"  Accuracy:    {final_m['accuracy']:.4f}")
    print(f"  Sensitivity: {final_m['sensitivity']:.4f}")
    print(f"  Specificity: {final_m['specificity']:.4f}")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"final": final_m, "history": history}, f, indent=2)

    plot_training_curves(history, args.output_dir)
    save_predictions(model, val_loader, device, args.output_dir)

    print(f"\nOutputs saved to {args.output_dir}/")
    print(f"Best Val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
