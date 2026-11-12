#!/usr/bin/env python3
"""
Evaluate the trained ONNX model on the validation (or test) split.

Reports overall accuracy, per-class precision/recall/F1, confusion matrix,
and optionally saves a confusion matrix heatmap plot.

Uses only OpenCV + NumPy — no PyTorch needed.

Usage:
    python evaluate.py --data ./processed_dataset/val --model drowsiness_model.onnx
    python evaluate.py --data ./processed_dataset/val --save-plot
    python evaluate.py --data ./processed_dataset/val --save-misclassified 20
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np


# Must match training configuration
INPUT_IMG_SIZE = 80
STATE_LABELS = ["0_alert", "1_drowsy"]
DISPLAY_LABELS = ["Alert", "Drowsy"]


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(script_dir, "drowsiness_model.onnx")
    default_data = os.path.join(script_dir, "processed_dataset", "val")

    parser = argparse.ArgumentParser(description="Evaluate the drowsiness ONNX model on a dataset split.")
    parser.add_argument("--data", default=default_data,
                        help="Path to the evaluation split (must contain 0_alert/ and 1_drowsy/ sub-dirs).")
    parser.add_argument("--model", default=default_model,
                        help="Path to the ONNX model.")
    parser.add_argument("--save-plot", action="store_true",
                        help="Save a confusion matrix heatmap as confusion_matrix.png.")
    parser.add_argument("--save-misclassified", type=int, default=0, metavar="N",
                        help="Save a grid of N misclassified images to misclassified.png.")
    return parser.parse_args()


def softmax(logits):
    """Compute softmax probabilities from raw logits."""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def load_images(data_dir):
    """Load images and labels from an ImageFolder-like directory structure."""
    images = []
    labels = []

    for class_idx, class_name in enumerate(STATE_LABELS):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"WARNING: Class directory not found: {class_dir}")
            continue

        files = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        for fname in files:
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (INPUT_IMG_SIZE, INPUT_IMG_SIZE))
            images.append((fpath, img))
            labels.append(class_idx)

    return images, labels


def predict(model, img):
    """Run a single grayscale image through the ONNX model. Returns (class_id, confidence)."""
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1.0 / 127.5,
        size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
        mean=(127.5,),
        swapRB=False,
        crop=False,
    )
    model.setInput(blob)
    preds = model.forward()
    probs = softmax(preds[0])
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    return class_id, confidence


def print_classification_report(y_true, y_pred):
    """Print precision, recall, F1, and confusion matrix. Returns (accuracy, confusion_matrix)."""
    num_classes = len(DISPLAY_LABELS)

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    print("\n" + "=" * 50)
    print("  CONFUSION MATRIX")
    print("=" * 50)
    print(f"\n  {'Predicted →':>20s}")
    header = f"  {'Actual ↓':>12s}" + "".join(f"  {DISPLAY_LABELS[j]:>8s}" for j in range(num_classes))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(num_classes):
        row = f"  {DISPLAY_LABELS[i]:>12s}" + "".join(f"  {cm[i][j]:>8d}" for j in range(num_classes))
        print(row)

    print("\n" + "=" * 50)
    print("  CLASSIFICATION REPORT")
    print("=" * 50)
    print(f"  {'Class':>12s}  {'Precision':>9s}  {'Recall':>9s}  {'F1-Score':>9s}  {'Support':>8s}")
    print("  " + "-" * 53)

    total_correct = 0
    total_samples = 0
    per_class = []

    for i in range(num_classes):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(num_classes)) - tp
        fn = sum(cm[i]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(sum(cm[i]))

        total_correct += tp
        total_samples += support

        per_class.append({"precision": precision, "recall": recall, "f1": f1, "support": support})
        print(f"  {DISPLAY_LABELS[i]:>12s}  {precision:>9.4f}  {recall:>9.4f}  {f1:>9.4f}  {support:>8d}")

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # Macro averages
    macro_p = np.mean([c["precision"] for c in per_class])
    macro_r = np.mean([c["recall"] for c in per_class])
    macro_f1 = np.mean([c["f1"] for c in per_class])

    print("  " + "-" * 53)
    print(f"  {'Accuracy':>12s}  {'':>9s}  {'':>9s}  {accuracy:>9.4f}  {total_samples:>8d}")
    print(f"  {'Macro Avg':>12s}  {macro_p:>9.4f}  {macro_r:>9.4f}  {macro_f1:>9.4f}  {total_samples:>8d}")
    print()

    return accuracy, cm


def plot_confusion_matrix(cm, save_dir):
    """Save a confusion matrix heatmap using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not installed — skipping plot. Install: pip install matplotlib")
        return None

    fig, ax = plt.subplots(figsize=(6, 5))

    # Normalize for color intensity
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Labels
    ax.set_xticks(range(len(DISPLAY_LABELS)))
    ax.set_yticks(range(len(DISPLAY_LABELS)))
    ax.set_xticklabels(DISPLAY_LABELS, fontsize=12)
    ax.set_yticklabels(DISPLAY_LABELS, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    # Cell annotations: count + percentage
    for i in range(len(DISPLAY_LABELS)):
        for j in range(len(DISPLAY_LABELS)):
            count = cm[i][j]
            pct = cm_norm[i][j] * 100
            color = "white" if cm_norm[i][j] > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix plot saved → {plot_path}")
    return plot_path


def save_misclassified_grid(misclassified, n, save_dir):
    """Save a grid of misclassified images for error analysis."""
    if not misclassified:
        print("  No misclassified images to save.")
        return None

    samples = misclassified[:n]
    cols = min(5, len(samples))
    rows = (len(samples) + cols - 1) // cols

    cell_size = 100
    padding = 30
    grid_w = cols * cell_size
    grid_h = rows * (cell_size + padding)
    grid = np.ones((grid_h, grid_w), dtype=np.uint8) * 240  # Light gray background

    for idx, (fpath, img, true_label, pred_label, conf) in enumerate(samples):
        row = idx // cols
        col = idx % cols
        x = col * cell_size
        y = row * (cell_size + padding)

        # Resize image to cell
        resized = cv2.resize(img, (cell_size, cell_size))
        grid[y:y + cell_size, x:x + cell_size] = resized

        # Label: True -> Pred
        label = f"T:{DISPLAY_LABELS[true_label][:1]} P:{DISPLAY_LABELS[pred_label][:1]} {conf:.0%}"
        # Draw on a color version for colored text
        # (we'll convert the whole grid to color at the end)

    # Convert to color for colored annotations
    grid_color = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)

    for idx, (fpath, img, true_label, pred_label, conf) in enumerate(samples):
        row = idx // cols
        col = idx % cols
        x = col * cell_size
        y = row * (cell_size + padding)

        text = f"T:{DISPLAY_LABELS[true_label][:3]} P:{DISPLAY_LABELS[pred_label][:3]}"
        cv2.putText(grid_color, text, (x + 2, y + cell_size + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)
        cv2.putText(grid_color, f"{conf:.0%}", (x + 2, y + cell_size + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

    path = os.path.join(save_dir, "misclassified.png")
    cv2.imwrite(path, grid_color)
    print(f"  ✓ Misclassified samples saved → {path} ({len(samples)} images)")
    return path


def main():
    args = parse_args()

    # Validate inputs
    if not os.path.isfile(args.model):
        print(f"ERROR: Model not found: {args.model}")
        print("Train the model first:  python pyt2.py")
        sys.exit(1)

    if not os.path.isdir(args.data):
        print(f"ERROR: Data directory not found: {args.data}")
        print("Run preprocessing first:  python ddd_preprocessing.py --source /path/to/DDD")
        sys.exit(1)

    # Load model
    try:
        model = cv2.dnn.readNetFromONNX(args.model)
        print(f"✓ Loaded model: {args.model}")
    except cv2.error as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)

    # Load data
    print(f"Loading images from: {args.data}")
    images, y_true = load_images(args.data)
    print(f"  Total images: {len(images)}")

    if len(images) == 0:
        print("ERROR: No images found.")
        sys.exit(1)

    for i, label in enumerate(STATE_LABELS):
        count = sum(1 for lbl in y_true if lbl == i)
        print(f"  {DISPLAY_LABELS[i]:>8s}: {count}")

    # Predict
    print("\nRunning inference...")
    y_pred = []
    confidences = []
    misclassified = []
    start_time = time.perf_counter()

    for idx, (fpath, img) in enumerate(images):
        pred, conf = predict(model, img)
        y_pred.append(pred)
        confidences.append(conf)

        # Track misclassified
        if pred != y_true[idx]:
            misclassified.append((fpath, img, y_true[idx], pred, conf))

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(images)} images...")

    elapsed = time.perf_counter() - start_time

    # Inference speed report
    print(f"\n  Inference completed in {elapsed:.2f}s")
    print(f"  Speed: {len(images) / elapsed:.1f} images/sec ({elapsed / len(images) * 1000:.2f} ms/image)")

    # Classification report
    accuracy, cm = print_classification_report(y_true, y_pred)
    print(f"Overall accuracy: {accuracy:.2%}")
    print(f"Misclassified: {len(misclassified)} / {len(images)} images")

    # Confidence statistics
    conf_arr = np.array(confidences)
    print(f"\nConfidence statistics:")
    print(f"  Mean  : {conf_arr.mean():.4f}")
    print(f"  Std   : {conf_arr.std():.4f}")
    print(f"  Min   : {conf_arr.min():.4f}")
    print(f"  Max   : {conf_arr.max():.4f}")

    # Optional: save plots
    save_dir = os.path.dirname(args.data) if os.path.basename(args.data) in ("val", "test", "train") \
        else os.path.dirname(os.path.abspath(args.model))

    if args.save_plot:
        plot_confusion_matrix(cm, save_dir)

    if args.save_misclassified > 0:
        # Sort by lowest confidence (most uncertain misclassifications)
        misclassified.sort(key=lambda x: x[4])
        save_misclassified_grid(misclassified, args.save_misclassified, save_dir)


if __name__ == "__main__":
    main()
