#!/usr/bin/env python3
"""
Inference Benchmark — DrowsinessNet

Measures end-to-end latency and throughput of the full inference pipeline
(face detection + preprocessing + ONNX classification) to verify real-time
feasibility on the target hardware.

Usage:
    python benchmark.py
    python benchmark.py --model drowsiness_model.onnx --runs 500
"""

import os
import sys
import time
import argparse
import statistics

import cv2
import numpy as np


INPUT_IMG_SIZE = 80


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(script_dir, "drowsiness_model.onnx")

    parser = argparse.ArgumentParser(description="Benchmark the drowsiness detection pipeline.")
    parser.add_argument("--model", default=default_model, help="Path to the ONNX model.")
    parser.add_argument("--runs", type=int, default=200, help="Number of inference runs (default: 200).")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup runs (default: 20).")
    return parser.parse_args()


def generate_synthetic_face(size=200):
    """Generate a synthetic grayscale face-like image for benchmarking."""
    img = np.random.randint(80, 200, (size, size), dtype=np.uint8)
    # Add some structure so Haar cascade has something to work with
    cv2.circle(img, (size // 3, size // 3), size // 8, 50, -1)  # left eye
    cv2.circle(img, (2 * size // 3, size // 3), size // 8, 50, -1)  # right eye
    cv2.ellipse(img, (size // 2, 2 * size // 3), (size // 4, size // 6), 0, 0, 180, 50, -1)  # mouth
    return img


def benchmark_classification_only(model, n_runs, n_warmup):
    """Benchmark just the ONNX forward pass."""
    # Synthetic input matching training pre-processing
    fake_face = np.random.randint(0, 255, (INPUT_IMG_SIZE, INPUT_IMG_SIZE), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(
        fake_face,
        scalefactor=1.0 / 127.5,
        size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
        mean=(127.5,),
        swapRB=False,
        crop=False,
    )

    # Warmup
    for _ in range(n_warmup):
        model.setInput(blob)
        model.forward()

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.setInput(blob)
        model.forward()
        times.append((time.perf_counter() - t0) * 1000)

    return times


def benchmark_face_detection(n_runs, n_warmup):
    """Benchmark Haar cascade face detection."""
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Use a 640x480 synthetic frame
    frame = np.random.randint(50, 200, (480, 640), dtype=np.uint8)

    # Warmup
    for _ in range(n_warmup):
        face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        times.append((time.perf_counter() - t0) * 1000)

    return times


def benchmark_full_pipeline(model, n_runs, n_warmup):
    """Benchmark the complete pipeline: detection + preprocess + classify."""
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Create a synthetic 640x480 BGR frame with a face-like region
    frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pre-generate a face ROI (since Haar won't find faces in noise)
    face_roi = np.random.randint(0, 255, (INPUT_IMG_SIZE, INPUT_IMG_SIZE), dtype=np.uint8)

    # Warmup
    for _ in range(n_warmup):
        face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        blob = cv2.dnn.blobFromImage(face_roi, 1.0 / 127.5, (INPUT_IMG_SIZE, INPUT_IMG_SIZE),
                                     (127.5,), False, False)
        model.setInput(blob)
        model.forward()

    # Benchmark
    times_detect = []
    times_preprocess = []
    times_classify = []

    for _ in range(n_runs):
        # Detection
        t0 = time.perf_counter()
        face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        t1 = time.perf_counter()

        # Preprocessing
        resized = cv2.resize(face_roi, (INPUT_IMG_SIZE, INPUT_IMG_SIZE))
        blob = cv2.dnn.blobFromImage(resized, 1.0 / 127.5, (INPUT_IMG_SIZE, INPUT_IMG_SIZE),
                                     (127.5,), False, False)
        t2 = time.perf_counter()

        # Classification
        model.setInput(blob)
        model.forward()
        t3 = time.perf_counter()

        times_detect.append((t1 - t0) * 1000)
        times_preprocess.append((t2 - t1) * 1000)
        times_classify.append((t3 - t2) * 1000)

    return times_detect, times_preprocess, times_classify


def print_stats(name, times_ms):
    """Print timing statistics for a benchmark."""
    mean = statistics.mean(times_ms)
    std = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    p50 = statistics.median(times_ms)
    p95 = sorted(times_ms)[int(0.95 * len(times_ms))]
    p99 = sorted(times_ms)[int(0.99 * len(times_ms))]
    fps = 1000.0 / mean if mean > 0 else float("inf")

    print(f"  {name:>20s}:  mean={mean:7.2f}ms  std={std:6.2f}ms  "
          f"p50={p50:7.2f}ms  p95={p95:7.2f}ms  p99={p99:7.2f}ms  → {fps:.0f} FPS")


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    model = cv2.dnn.readNetFromONNX(args.model)
    print(f"✓ Loaded model: {args.model}")
    print(f"  Runs: {args.runs}  |  Warmup: {args.warmup}")
    print(f"  OpenCV: {cv2.__version__}")

    # 1. Classification only
    print("\n" + "=" * 80)
    print("  BENCHMARK: ONNX Classification Only")
    print("=" * 80)
    cls_times = benchmark_classification_only(model, args.runs, args.warmup)
    print_stats("Classification", cls_times)

    # 2. Face detection only
    print("\n" + "=" * 80)
    print("  BENCHMARK: Haar Cascade Face Detection")
    print("=" * 80)
    try:
        det_times = benchmark_face_detection(args.runs, args.warmup)
        print_stats("Face Detection", det_times)
    except Exception as e:
        print(f"  Skipped (Haar cascade not available): {e}")

    # 3. Full pipeline
    print("\n" + "=" * 80)
    print("  BENCHMARK: Full Pipeline (detect + preprocess + classify)")
    print("=" * 80)
    try:
        td, tp, tc = benchmark_full_pipeline(model, args.runs, args.warmup)
        print_stats("Detection", td)
        print_stats("Preprocessing", tp)
        print_stats("Classification", tc)

        total = [d + p + c for d, p, c in zip(td, tp, tc)]
        print_stats("TOTAL", total)

        # Breakdown
        mean_total = statistics.mean(total)
        print(f"\n  Pipeline breakdown (mean):")
        print(f"    Detection     : {statistics.mean(td):6.2f}ms  ({statistics.mean(td)/mean_total*100:.0f}%)")
        print(f"    Preprocessing : {statistics.mean(tp):6.2f}ms  ({statistics.mean(tp)/mean_total*100:.0f}%)")
        print(f"    Classification: {statistics.mean(tc):6.2f}ms  ({statistics.mean(tc)/mean_total*100:.0f}%)")
    except Exception as e:
        print(f"  Skipped: {e}")

    # Verdict
    total_mean = statistics.mean(cls_times)
    print("\n" + "=" * 80)
    if total_mean < 33.3:  # 30 FPS
        print(f"  ✅ REAL-TIME CAPABLE: {1000/total_mean:.0f} FPS (classification alone)")
    elif total_mean < 100:
        print(f"  ⚠ MARGINAL: {1000/total_mean:.0f} FPS — may need optimization for 30 FPS target")
    else:
        print(f"  ❌ TOO SLOW: {1000/total_mean:.0f} FPS — consider model pruning or hardware acceleration")
    print("=" * 80)


if __name__ == "__main__":
    main()
