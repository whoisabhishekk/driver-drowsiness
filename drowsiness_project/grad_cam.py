#!/usr/bin/env python3
"""
Grad-CAM Visualization — DrowsinessNet

Generates Gradient-weighted Class Activation Maps to visualize which facial
regions the model focuses on when classifying alert vs. drowsy.

This is essential for:
    - Verifying the model looks at meaningful features (eyes, mouth)
    - Debugging false positives/negatives
    - Building trust in the model's decisions
    - Presentations and reports

Usage:
    python grad_cam.py --image face.jpg
    python grad_cam.py --image face.jpg --model-pth drowsiness_model.pth --output grad_cam_output.png
    python grad_cam.py --data ./processed_dataset/val --grid 4x4
"""

import os
import sys
import argparse
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
except ImportError:
    print("ERROR: PyTorch is required for Grad-CAM. Install: pip install torch torchvision")
    sys.exit(1)

import cv2


# ──────────────────────────────────────────────
# Model (must match pyt2.py exactly)
# ──────────────────────────────────────────────

import torch.nn as nn


class DrowsinessNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, img_size=80):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
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
# Grad-CAM
# ──────────────────────────────────────────────

class GradCAM:
    """Grad-CAM for the last convolutional layer of DrowsinessNet."""

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Hook the last conv layer (last layer in conv_layers before fc)
        # conv_layers[-1] is MaxPool2d, conv_layers[-2] is BatchNorm2d,
        # conv_layers[-3] is ReLU, conv_layers[-4] is Conv2d
        # We hook the output of the last BatchNorm2d (index -2)
        target_layer = model.conv_layers[-2]  # BatchNorm2d(64)
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, C, H, W) tensor
            target_class: class index to explain (None = predicted class)

        Returns:
            heatmap: (H, W) numpy array in [0, 1]
            predicted_class: int
            confidence: float
        """
        self.model.zero_grad()

        # Forward
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        # Grad-CAM: weight activations by gradient-averaged importance
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global average pool
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, confidence


def apply_heatmap(image_gray, heatmap, alpha=0.5):
    """Overlay Grad-CAM heatmap on the original grayscale image.

    Returns:
        overlay: (H, W, 3) BGR uint8 image
    """
    h, w = image_gray.shape[:2]

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to color heatmap
    heatmap_color = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )

    # Convert grayscale to BGR
    if len(image_gray.shape) == 2:
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image_gray

    # Overlay
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_pth = os.path.join(script_dir, "drowsiness_model.pth")

    parser = argparse.ArgumentParser(description="Grad-CAM visualization for DrowsinessNet.")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single face image to visualize.")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to a dataset split (e.g. ./processed_dataset/val) for grid visualization.")
    parser.add_argument("--model-pth", default=default_pth,
                        help="Path to the .pth model checkpoint.")
    parser.add_argument("--output", default=None,
                        help="Output image path (default: grad_cam_output.png or grad_cam_grid.png).")
    parser.add_argument("--grid", default="4x4", metavar="ROWSxCOLS",
                        help="Grid size for dataset visualization (default: 4x4).")
    return parser.parse_args()


def preprocess_image(image_path):
    """Load and preprocess a single image for the model."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Could not read image: {image_path}")
        sys.exit(1)

    img_resized = cv2.resize(img, (80, 80))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    tensor = transform(img_resized).unsqueeze(0)  # (1, 1, 80, 80)
    return img_resized, tensor


def visualize_single(model_path, image_path, output_path):
    """Generate and save Grad-CAM for a single image."""
    model = DrowsinessNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    grad_cam = GradCAM(model)

    img, tensor = preprocess_image(image_path)
    heatmap, pred_class, confidence = grad_cam.generate(tensor)

    labels = ["Alert", "Drowsy"]

    # Create a side-by-side comparison
    overlay = apply_heatmap(img, heatmap, alpha=0.5)
    original_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Resize for better visibility
    scale = 3
    original_big = cv2.resize(original_bgr, (80 * scale, 80 * scale), interpolation=cv2.INTER_NEAREST)
    overlay_big = cv2.resize(overlay, (80 * scale, 80 * scale), interpolation=cv2.INTER_LINEAR)

    # Add labels
    label_text = f"{labels[pred_class]} ({confidence:.0%})"
    cv2.putText(original_big, "Original", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay_big, f"Grad-CAM: {label_text}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    combined = np.hstack([original_big, overlay_big])

    cv2.imwrite(output_path, combined)
    print(f"✓ Grad-CAM saved → {output_path}")
    print(f"  Prediction: {labels[pred_class]} (confidence: {confidence:.2%})")


def visualize_grid(model_path, data_dir, output_path, grid_size):
    """Generate a grid of Grad-CAM visualizations from a dataset split."""
    model = DrowsinessNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    grad_cam = GradCAM(model)
    labels_list = ["Alert", "Drowsy"]
    class_dirs = ["0_alert", "1_drowsy"]

    rows, cols = grid_size
    n_total = rows * cols
    n_per_class = n_total // 2

    # Collect sample images from each class
    samples = []
    for class_idx, class_name in enumerate(class_dirs):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"WARNING: {class_dir} not found, skipping.")
            continue

        files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        # Pick evenly spaced samples
        indices = np.linspace(0, len(files) - 1, n_per_class, dtype=int)
        for idx in indices:
            samples.append((os.path.join(class_dir, files[idx]), class_idx))

    if not samples:
        print("ERROR: No images found.")
        sys.exit(1)

    cell_size = 100
    header_h = 22
    grid_img = np.ones((rows * (cell_size + header_h), cols * cell_size, 3), dtype=np.uint8) * 40

    for i, (fpath, true_class) in enumerate(samples[:n_total]):
        if i >= rows * cols:
            break

        row = i // cols
        col = i % cols

        img, tensor = preprocess_image(fpath)

        # Need to re-create GradCAM for each image since hooks accumulate
        model_fresh = DrowsinessNet()
        model_fresh.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        gc = GradCAM(model_fresh)

        heatmap, pred_class, confidence = gc.generate(tensor)
        overlay = apply_heatmap(img, heatmap, alpha=0.5)
        overlay_resized = cv2.resize(overlay, (cell_size, cell_size))

        x = col * cell_size
        y = row * (cell_size + header_h)

        grid_img[y + header_h:y + header_h + cell_size, x:x + cell_size] = overlay_resized

        # Header label
        true_str = labels_list[true_class][:3]
        pred_str = labels_list[pred_class][:3]
        correct = "✓" if true_class == pred_class else "✗"
        color = (0, 255, 0) if true_class == pred_class else (0, 0, 255)
        header_text = f"{correct} T:{true_str} P:{pred_str} {confidence:.0%}"
        cv2.putText(grid_img, header_text, (x + 2, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    cv2.imwrite(output_path, grid_img)
    print(f"✓ Grad-CAM grid saved → {output_path} ({len(samples[:n_total])} images)")


def main():
    args = parse_args()

    if not os.path.isfile(args.model_pth):
        print(f"ERROR: Model checkpoint not found: {args.model_pth}")
        print("Grad-CAM requires the .pth file (not .onnx). Train the model first:")
        print("  python pyt2.py --data ./processed_dataset")
        sys.exit(1)

    if args.image:
        output = args.output or "grad_cam_output.png"
        visualize_single(args.model_pth, args.image, output)

    elif args.data:
        output = args.output or "grad_cam_grid.png"
        try:
            r, c = map(int, args.grid.split("x"))
        except ValueError:
            print(f"ERROR: Invalid grid format '{args.grid}'. Use ROWSxCOLS, e.g. 4x4")
            sys.exit(1)
        visualize_grid(args.model_pth, args.data, output, (r, c))

    else:
        print("ERROR: Provide --image for single visualization or --data for grid visualization.")
        print("\nExamples:")
        print("  python grad_cam.py --image face.jpg")
        print("  python grad_cam.py --data ./processed_dataset/val --grid 4x4")
        sys.exit(1)


if __name__ == "__main__":
    main()
