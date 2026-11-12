# Driver Drowsiness Detection System

Real-time driver drowsiness detection from a webcam feed. A compact CNN classifies the driver's face as **alert** or **drowsy** every frame; a temporal debounce counter suppresses single-frame noise and raises an audible alert only when drowsiness persists.

The model is trained in PyTorch, exported to **ONNX**, and served at inference time through **OpenCV's DNN backend** — so deployment needs only `opencv-python` and `numpy`, with no PyTorch/TensorFlow runtime.

**Stack:** Python · PyTorch · ONNX · OpenCV (DNN + Haar cascades) · NumPy · Matplotlib · tqdm

---

## Pipeline at a glance

```
Driver Drowsiness Dataset (DDD)   ← download_dataset.py
        │
        │  ddd_preprocessing.py       → per-class 80/20 train/val split (seed 42)
        ▼
processed_dataset/{train,val}/{0_alert,1_drowsy}
        │
        │  pyt2.py                    → DrowsinessNet (3-block CNN), 15 epochs, Adam
        ▼                               → training_curves.png + training_log.csv
drowsiness_model.pth  ──export──►  drowsiness_model.onnx
        │
        ├── evaluate.py               → accuracy, precision/recall/F1, confusion matrix
        ├── grad_cam.py               → model interpretability heatmaps
        ├── benchmark.py              → latency & throughput profiling
        └── ocv.py                    → webcam → Haar face detect → ONNX → alarm
            │                           → FPS counter, confidence %, video recording
            ▼
        Alert / Drowsy(n) / !!! DROWSY !!!  + audible alarm
```

---

## Repository layout

| Path | Purpose |
| --- | --- |
| `drowsiness_project/download_dataset.py` | Download the DDD dataset from Kaggle |
| `drowsiness_project/ddd_preprocessing.py` | Split raw DDD into train/val `ImageFolder` tree |
| `drowsiness_project/pyt2.py` | Model definition, training, checkpointing, ONNX export, training curves |
| `drowsiness_project/ocv.py` | Real-time inference: FPS, confidence %, debounce, alarm, video recording |
| `drowsiness_project/evaluate.py` | Offline eval: confusion matrix, precision/recall/F1, misclassified samples |
| `drowsiness_project/grad_cam.py` | Grad-CAM visualization — what the model looks at |
| `drowsiness_project/benchmark.py` | Latency & throughput profiling (detection, preprocessing, classification) |
| `drowsiness_project/drowsiness_model.onnx` | Pre-trained model, ONNX opset 11 (~3.4 MB) |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT License |

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For **inference only** (no training):

```bash
pip install opencv-python numpy
```

### 2. Run real-time detection (pre-trained model included)

```bash
cd drowsiness_project
python ocv.py
```

Press **q** to quit. The session summary prints on exit.

### 3. Record a demo video

```bash
python ocv.py --save-video demo.avi
```

---

## Full training pipeline

### Step 1: Download the dataset

```bash
python drowsiness_project/download_dataset.py
# Or download manually from:
# https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd
```

### Step 2: Preprocess (split into train/val)

```bash
python drowsiness_project/ddd_preprocessing.py --source /path/to/DDD
```

### Step 3: Train the model

```bash
python drowsiness_project/pyt2.py --data drowsiness_project/processed_dataset
```

Produces:
- `drowsiness_model.pth` — best PyTorch checkpoint
- `drowsiness_model.onnx` — ONNX export for deployment
- `training_curves.png` — loss & accuracy plots
- `training_log.csv` — per-epoch metrics

### Step 4: Evaluate

```bash
python drowsiness_project/evaluate.py --save-plot --save-misclassified 20
```

Produces:
- Classification report (precision, recall, F1)
- `confusion_matrix.png` — heatmap plot
- `misclassified.png` — grid of most uncertain errors

### Step 5: Interpret the model (Grad-CAM)

```bash
# Single image
python drowsiness_project/grad_cam.py --image face.jpg

# Grid from dataset
python drowsiness_project/grad_cam.py --data drowsiness_project/processed_dataset/val --grid 4x4
```

### Step 6: Benchmark performance

```bash
python drowsiness_project/benchmark.py --runs 500
```

---

## Model architecture — `DrowsinessNet`

Deliberately small, so it stays real-time on a CPU-only edge device.

| Stage | Layers | Output |
| --- | --- | --- |
| Input | 1×80×80 grayscale | — |
| Conv block 1 | Conv2d(1→16, 3×3, pad 1) → ReLU → BatchNorm → MaxPool 2×2 | 16×40×40 |
| Conv block 2 | Conv2d(16→32, 3×3, pad 1) → ReLU → BatchNorm → MaxPool 2×2 | 32×20×20 |
| Conv block 3 | Conv2d(32→64, 3×3, pad 1) → ReLU → BatchNorm → MaxPool 2×2 | 64×10×10 |
| Head | Flatten(6400) → Linear(128) → ReLU → Dropout(0.5) → Linear(2) | 2 logits |

---

## Real-time inference HUD

The `ocv.py` window displays:

| Element | Description |
| --- | --- |
| **Status panel** | Dark semi-transparent bar at the top |
| **Status text** | Current state (Alert / Drowsy / !!! DROWSY !!!) |
| **Confidence bar** | Visual confidence percentage with fill |
| **FPS counter** | Sliding-window average frames per second |
| **Drowsy counter** | `Counter: n/threshold` — progress toward alarm |
| **Bounding box** | Green (alert), yellow (drowsy), red (alarm) |
| **Face label** | Per-face prediction with confidence % |

### Temporal debouncing

| Counter state | Display | Colour |
| --- | --- | --- |
| Alert / no detection | `Alert` / `No Face Detected` | green / white |
| `1 <= n < threshold` | `Drowsy (n)` | yellow |
| `n >= threshold` | `!!! DROWSY !!!` + alarm | red |

### Cross-platform alarm

| OS | Method |
| --- | --- |
| Windows | `winsound.Beep(1500, 500)` |
| macOS | Generated WAV played via `afplay` |
| Linux | Generated WAV played via `paplay` / `aplay` / `ffplay` |
| Fallback | Terminal bell `\a` |

---

## Augmentation (train split only)

- `RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15))`
- `RandomHorizontalFlip()`
- `ColorJitter(brightness=0.3, contrast=0.3)`
- `Normalize(mean=[0.5], std=[0.5])` → inputs in `[-1, 1]`

---

## Raspberry Pi notes

- ONNX + OpenCV-DNN = no PyTorch on the Pi.
- Grayscale 80×80 + ~3.4 MB model → CPU-only real-time feasible.
- Drop `haarcascade_frontalface_default.xml` next to `ocv.py` if `cv2.data.haarcascades` is unavailable.
- Use `--camera` flag for non-default camera indices or GStreamer pipelines.
- Run `python benchmark.py` to verify real-time feasibility on your hardware.

---

## Tuning

| Setting | File | Effect |
| --- | --- | --- |
| `--threshold` | `ocv.py` | Frames before alarm. Higher = fewer false alarms. Default 20 ≈ 0.7s at 30 fps. |
| `--epochs`, `--lr`, `--batch-size` | `pyt2.py` | Training hyperparameters. |
| `--img-size` | `pyt2.py` | Input resolution. Must match `ocv.py` `INPUT_IMG_SIZE`. |
| `--val-ratio` | `ddd_preprocessing.py` | Train/val proportion. |
