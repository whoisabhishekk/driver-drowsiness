#!/usr/bin/env python3
"""
Model Definition, Training, and ONNX Export — DrowsinessNet

A compact 3-block CNN that classifies 80×80 grayscale face crops as
alert (0) or drowsy (1).

Produces:
    - drowsiness_model.pth   (best PyTorch checkpoint)
    - drowsiness_model.onnx  (ONNX export for deployment)
    - training_curves.png    (loss & accuracy plots)
    - training_log.csv       (per-epoch metrics)

Usage:
    python pyt2.py --data ./processed_dataset
    python pyt2.py --data ./processed_dataset --epochs 20 --lr 0.0005
"""

import os
import sys
import csv
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class DrowsinessNet(nn.Module):
    """Deliberately small CNN — runs real-time on a CPU-only edge device.

    Architecture (for 80×80 grayscale input):
        Conv block 1: Conv2d(1→16)  → ReLU → BN → MaxPool → 16×40×40
        Conv block 2: Conv2d(16→32) → ReLU → BN → MaxPool → 32×20×20
        Conv block 3: Conv2d(32→64) → ReLU → BN → MaxPool → 64×10×10
        Head:         Flatten(6400) → Linear(128) → ReLU → Dropout(0.5) → Linear(2)
    """

    def __init__(self, input_channels=1, num_classes=2, img_size=80):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flattened_size = 64 * (img_size // 8) * (img_size // 8)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(script_dir, "processed_dataset")

    parser = argparse.ArgumentParser(description="Train DrowsinessNet and export to ONNX.")
    parser.add_argument("--data", default=default_data,
                        help="Path to the processed dataset (must contain train/ and val/ sub-dirs).")
    parser.add_argument("--img-size", type=int, default=80, help="Input image size (default: 80).")
    parser.add_argument("--channels", type=int, default=1, help="Input channels — 1=grayscale (default: 1).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001).")
    parser.add_argument("--output-dir", default=script_dir,
                        help="Directory to save .pth and .onnx models (default: same as script).")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def build_dataloaders(data_dir, img_size, channels, batch_size):
    """Build training and validation dataloaders with appropriate augmentation."""
    data_transforms = {
        "train": transforms.Compose([
            transforms.Grayscale(num_output_channels=channels),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]),
        "val": transforms.Compose([
            transforms.Grayscale(num_output_channels=channels),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]),
    }

    image_datasets = {
        split: datasets.ImageFolder(os.path.join(data_dir, split), data_transforms[split], allow_empty=True)
        for split in ["train", "val"]
    }

    dataloaders = {
        split: DataLoader(image_datasets[split], batch_size=batch_size, shuffle=True, num_workers=0)
        for split in ["train", "val"]
    }

    return image_datasets, dataloaders


def train(model, dataloaders, dataset_sizes, device, criterion, optimizer, num_epochs, save_dir):
    """Standard train/val loop with best-checkpoint saving and metric logging."""
    best_weights = model.state_dict()
    best_acc = 0.0
    pth_path = os.path.join(save_dir, "drowsiness_model.pth")

    # Per-epoch metrics for plotting
    history = {
        "epoch": [],
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        epoch_metrics = {}

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            progress = tqdm(dataloaders[phase], desc=f"  {phase:>5s}")

            for inputs, labels in progress:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                progress.set_postfix(
                    loss=running_loss / dataset_sizes[phase],
                    acc=(running_corrects.double() / dataset_sizes[phase]).item(),
                )

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()
            print(f"  {phase:>5s}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            epoch_metrics[f"{phase}_loss"] = epoch_loss
            epoch_metrics[f"{phase}_acc"] = epoch_acc

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = model.state_dict()
                torch.save(best_weights, pth_path)
                print(f"  ✓ New best model saved → {pth_path}")

        # Record history
        history["epoch"].append(epoch + 1)
        for key in ["train_loss", "train_acc", "val_loss", "val_acc"]:
            history[key].append(epoch_metrics[key])

    return best_weights, best_acc, history


def save_training_log(history, save_dir):
    """Save per-epoch metrics to a CSV file."""
    csv_path = os.path.join(save_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                f"{history['train_loss'][i]:.6f}",
                f"{history['train_acc'][i]:.6f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['val_acc'][i]:.6f}",
            ])
    print(f"  ✓ Training log saved → {csv_path}")
    return csv_path


def plot_training_curves(history, save_dir):
    """Generate and save training/validation loss & accuracy plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not installed — skipping plot generation.")
        print("    Install it with: pip install matplotlib")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = history["epoch"]

    # --- Loss plot ---
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", markersize=4, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Accuracy plot ---
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", markersize=4, label="Train Acc")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r-o", markersize=4, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    # Annotate best val accuracy
    best_idx = history["val_acc"].index(max(history["val_acc"]))
    best_epoch = epochs[best_idx]
    best_val = history["val_acc"][best_idx] * 100
    ax2.annotate(
        f"Best: {best_val:.1f}%\n(epoch {best_epoch})",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 1, best_val - 10),
        arrowprops=dict(arrowstyle="->", color="green"),
        fontsize=9, color="green", fontweight="bold",
    )

    plt.suptitle("DrowsinessNet Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training curves saved → {plot_path}")
    return plot_path


def export_onnx(model, best_weights, device, channels, img_size, save_dir):
    """Reload best weights and export to ONNX."""
    model.load_state_dict(best_weights)
    model.eval()

    onnx_path = os.path.join(save_dir, "drowsiness_model.onnx")
    dummy = torch.randn(1, channels, img_size, img_size).to(device)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"  ✓ ONNX model exported → {onnx_path}")
    return onnx_path


def print_model_summary(model, img_size, channels, device):
    """Print model architecture summary with parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 45)
    print("  MODEL SUMMARY")
    print("=" * 45)
    print(f"  Architecture   : DrowsinessNet")
    print(f"  Input          : {channels}×{img_size}×{img_size} (grayscale)")
    print(f"  Output         : 2 classes (alert, drowsy)")
    print(f"  Total params   : {total_params:,}")
    print(f"  Trainable      : {trainable_params:,}")

    # Estimate model size
    size_mb = total_params * 4 / (1024 * 1024)  # float32
    print(f"  Est. size      : {size_mb:.2f} MB (float32)")

    # Inference speed estimate
    model.eval()
    dummy = torch.randn(1, channels, img_size, img_size).to(device)
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            model(dummy)
        # Benchmark
        start = time.perf_counter()
        n_runs = 100
        for _ in range(n_runs):
            model(dummy)
        elapsed = time.perf_counter() - start
        ms_per_frame = (elapsed / n_runs) * 1000
        print(f"  Inference time : {ms_per_frame:.2f} ms/frame ({device})")
        print(f"  Max throughput : {1000 / ms_per_frame:.0f} FPS ({device})")

    print("=" * 45)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU device      : {torch.cuda.get_device_name(device)}")
    print(f"Using device    : {device}\n")

    # Validate data directory
    if not os.path.isdir(args.data):
        print(f"ERROR: Data directory does not exist: {args.data}")
        print("Run the preprocessing script first:")
        print("  python ddd_preprocessing.py --source /path/to/DDD --output ./processed_dataset")
        sys.exit(1)

    for split in ["train", "val"]:
        split_path = os.path.join(args.data, split)
        if not os.path.isdir(split_path):
            print(f"ERROR: Missing '{split}/' directory in {args.data}")
            print("Run ddd_preprocessing.py first.")
            sys.exit(1)

    # Build data pipeline
    print("Loading datasets...")
    image_datasets, dataloaders = build_dataloaders(
        args.data, args.img_size, args.channels, args.batch_size
    )
    dataset_sizes = {split: len(ds) for split, ds in image_datasets.items()}
    class_names = image_datasets["train"].classes

    print(f"  Training   : {dataset_sizes['train']} images")
    print(f"  Validation : {dataset_sizes['val']} images")
    print(f"  Classes    : {class_names}")

    # Build model
    model = DrowsinessNet(
        input_channels=args.channels,
        num_classes=len(class_names),
        img_size=args.img_size,
    ).to(device)

    # Print model summary + benchmark
    print_model_summary(model, args.img_size, args.channels, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    print("\n" + "=" * 45)
    print("  TRAINING")
    print("=" * 45)
    start = time.time()

    best_weights, best_acc, history = train(
        model, dataloaders, dataset_sizes, device,
        criterion, optimizer, args.epochs, args.output_dir,
    )

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best validation accuracy: {best_acc:.4f}")

    # Save training log and plots
    print("\n" + "=" * 45)
    print("  SAVING ARTIFACTS")
    print("=" * 45)
    save_training_log(history, args.output_dir)
    plot_training_curves(history, args.output_dir)

    # Export
    print("\n" + "=" * 45)
    print("  ONNX EXPORT")
    print("=" * 45)
    try:
        export_onnx(model, best_weights, device, args.channels, args.img_size, args.output_dir)
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")
        sys.exit(1)

    print("\nDone! Next steps:")
    print("  python evaluate.py                  # evaluate on validation set")
    print("  python ocv.py                       # run real-time detection")
    print("  python ocv.py --save-video demo.avi # record a demo session")


if __name__ == "__main__":
    main()
