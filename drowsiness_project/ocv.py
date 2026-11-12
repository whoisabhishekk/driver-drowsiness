#!/usr/bin/env python3
"""
Real-time Driver Drowsiness Detection — Inference

Runs the trained ONNX model on a live webcam feed using OpenCV's DNN backend.
No PyTorch/TensorFlow required at inference time.

Pipeline per frame:
    1. Capture frame → grayscale
    2. Haar cascade face detection → pick largest face
    3. Pre-process ROI (resize → blobFromImage with matching normalization)
    4. ONNX forward pass → softmax → alert / drowsy + confidence %
    5. Temporal debouncing → alarm only on sustained drowsiness

Features:
    - Live FPS counter
    - Confidence percentage overlay
    - Video recording (--save-video output.avi)
    - Configurable alarm threshold

Usage:
    python ocv.py
    python ocv.py --model drowsiness_model.onnx --camera 0 --threshold 20
    python ocv.py --save-video session.avi
"""

import os
import sys
import math
import wave
import struct
import time
import tempfile
import platform
import subprocess
import argparse
import threading
from collections import deque

import cv2
import numpy as np


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

INPUT_IMG_SIZE = 80       # Must match training
INPUT_CHANNELS = 1        # Grayscale
STATE_LABELS = ["0_alert", "1_drowsy"]
DISPLAY_LABELS = ["Alert", "Drowsy"]


# ──────────────────────────────────────────────
# Cross-platform alarm
# ──────────────────────────────────────────────

def _generate_beep_wav(path, frequency=1500, duration_s=0.5, sample_rate=44100):
    """Generate a simple sine-wave beep WAV file using only the stdlib."""
    n_samples = int(sample_rate * duration_s)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            sample = int(32767 * 0.8 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            wf.writeframes(struct.pack("<h", sample))


# Lazily created temp WAV for the alarm sound
_alarm_wav_path = None


def _get_alarm_wav():
    """Return path to a temporary beep WAV, generating it on first call."""
    global _alarm_wav_path
    if _alarm_wav_path is None or not os.path.exists(_alarm_wav_path):
        fd, _alarm_wav_path = tempfile.mkstemp(suffix=".wav", prefix="drowsy_alarm_")
        os.close(fd)
        _generate_beep_wav(_alarm_wav_path)
    return _alarm_wav_path


def play_alarm():
    """Play the alarm sound in a background thread (non-blocking)."""

    def _play():
        system = platform.system()
        try:
            if system == "Windows":
                import winsound
                winsound.Beep(1500, 500)
            elif system == "Darwin":  # macOS
                wav = _get_alarm_wav()
                subprocess.run(["afplay", wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:  # Linux and others
                wav = _get_alarm_wav()
                for player in ["paplay", "aplay", "ffplay -nodisp -autoexit"]:
                    cmd = player.split() + [wav]
                    try:
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
                        return
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        continue
                print("\a", end="", flush=True)
        except Exception:
            print("\a", end="", flush=True)

    threading.Thread(target=_play, daemon=True).start()


# ──────────────────────────────────────────────
# Haar cascade locator
# ──────────────────────────────────────────────

def find_haar_cascade():
    """Search common paths for the frontal-face Haar cascade XML."""
    filename = "haarcascade_frontalface_default.xml"
    search_paths = [
        cv2.data.haarcascades if hasattr(cv2, "data") else "",
        "/usr/share/opencv4/haarcascades/",
        "/usr/local/share/opencv4/haarcascades/",
        "/usr/share/opencv/haarcascades/",
        os.path.dirname(os.path.abspath(__file__)),
        "",
    ]
    for base in search_paths:
        full = os.path.join(base, filename)
        if os.path.isfile(full):
            return full
    return None


# ──────────────────────────────────────────────
# Softmax
# ──────────────────────────────────────────────

def softmax(logits):
    """Compute softmax probabilities from raw logits."""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


# ──────────────────────────────────────────────
# FPS tracker
# ──────────────────────────────────────────────

class FPSTracker:
    """Sliding-window FPS counter for smooth display."""

    def __init__(self, window_size=30):
        self._timestamps = deque(maxlen=window_size)

    def tick(self):
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self):
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


# ──────────────────────────────────────────────
# HUD drawing helpers
# ──────────────────────────────────────────────

def draw_hud_background(frame, x, y, w, h, color=(0, 0, 0), alpha=0.6):
    """Draw a semi-transparent rectangle for text background."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_status_panel(frame, status, color, fps, confidence, drowsy_counter, threshold):
    """Draw a professional status panel at the top of the frame."""
    h, w = frame.shape[:2]
    panel_h = 70

    # Semi-transparent dark background
    draw_hud_background(frame, 0, 0, w, panel_h, (30, 30, 30), 0.7)

    # Status text (left)
    cv2.putText(frame, f"Status: {status}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Confidence bar
    if confidence is not None:
        conf_pct = int(confidence * 100)
        bar_x, bar_y, bar_w, bar_h = 10, 42, 200, 16
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        # Filled portion
        fill_w = int(bar_w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
        # Percentage text
        cv2.putText(frame, f"{conf_pct}%", (bar_x + bar_w + 8, bar_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # FPS (right)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 140, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Drowsy counter (right, below FPS)
    if drowsy_counter > 0:
        counter_text = f"Counter: {drowsy_counter}/{threshold}"
        counter_color = (0, 255, 255) if drowsy_counter < threshold else (0, 0, 255)
        cv2.putText(frame, counter_text, (w - 200, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, counter_color, 1)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(script_dir, "drowsiness_model.onnx")

    parser = argparse.ArgumentParser(description="Real-time drowsiness detection from webcam.")
    parser.add_argument("--model", default=default_model,
                        help="Path to the ONNX model file.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0).")
    parser.add_argument("--threshold", type=int, default=20,
                        help="Consecutive drowsy frames before alarm (default: 20, ~0.7 s at 30 fps).")
    parser.add_argument("--save-video", type=str, default=None, metavar="PATH",
                        help="Save annotated output to a video file (e.g. session.avi).")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # --- Load ONNX model ---
    if not os.path.isfile(args.model):
        print(f"ERROR: ONNX model not found: {args.model}")
        print("Train the model first:  python pyt2.py --data ./processed_dataset")
        sys.exit(1)

    try:
        model = cv2.dnn.readNetFromONNX(args.model)
        print(f"✓ Loaded ONNX model: {args.model}")
    except cv2.error as e:
        print(f"ERROR loading ONNX model: {e}")
        sys.exit(1)

    # --- Load Haar cascade ---
    cascade_path = find_haar_cascade()
    if cascade_path is None:
        print("ERROR: Could not find haarcascade_frontalface_default.xml")
        print("Download it from:")
        print("  https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
              "haarcascade_frontalface_default.xml")
        print("and place it next to this script.")
        sys.exit(1)

    face_cascade = cv2.CascadeClassifier(cascade_path)
    print(f"✓ Loaded Haar cascade: {cascade_path}")

    # --- Open camera ---
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}.")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera {args.camera} opened ({frame_w}×{frame_h})")

    # --- Video writer (optional) ---
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (frame_w, frame_h))
        if writer.isOpened():
            print(f"✓ Recording to: {args.save_video}")
        else:
            print(f"WARNING: Could not open video writer for {args.save_video}")
            writer = None

    # --- State ---
    drowsy_counter = 0
    alarm_on = False
    DROWSY_ALARM_THRESHOLD = args.threshold
    fps_tracker = FPSTracker(window_size=30)
    frame_count = 0
    session_start = time.time()

    # Session statistics
    stats = {
        "total_frames": 0,
        "face_detected_frames": 0,
        "alert_frames": 0,
        "drowsy_frames": 0,
        "alarm_triggers": 0,
    }

    print(f"\nRunning real-time detection (alarm threshold = {DROWSY_ALARM_THRESHOLD} frames)")
    print("Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        fps_tracker.tick()
        stats["total_frames"] += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Face detection ---
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
        )

        final_label = "Alert"
        final_color = (0, 255, 0)  # Green
        confidence = None

        if len(faces) > 0:
            stats["face_detected_frames"] += 1

            # Pick the largest face by bounding-box area
            (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])

            # --- Pre-process ROI ---
            face_roi = gray[y : y + h, x : x + w]
            try:
                resized_roi = cv2.resize(face_roi, (INPUT_IMG_SIZE, INPUT_IMG_SIZE))
            except cv2.error as e:
                print(f"Resize error: {e} — skipping frame.")
                continue

            # blobFromImage normalization — must reproduce training transforms:
            #   PyTorch: Normalize(mean=[0.5], std=[0.5])  on [0,1] tensor
            #            => (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1.0
            #   OpenCV:  (pixel - mean) * scalefactor
            #            => (pixel - 127.5) * (1/127.5)   ← same result
            blob = cv2.dnn.blobFromImage(
                resized_roi,
                scalefactor=1.0 / 127.5,
                size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
                mean=(127.5,),
                swapRB=False,
                crop=False,
            )

            # --- Classify ---
            model.setInput(blob)
            preds = model.forward()
            probs = softmax(preds[0])
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            label_name = STATE_LABELS[class_id]

            # --- Debouncing ---
            if label_name == "1_drowsy":
                drowsy_counter += 1
                stats["drowsy_frames"] += 1

                if drowsy_counter >= DROWSY_ALARM_THRESHOLD:
                    final_label = "!!! DROWSY !!!"
                    final_color = (0, 0, 255)  # Red

                    if not alarm_on:
                        play_alarm()
                        alarm_on = True
                        stats["alarm_triggers"] += 1
                else:
                    final_label = f"Drowsy ({drowsy_counter})"
                    final_color = (0, 255, 255)  # Yellow
                    alarm_on = False
            else:
                drowsy_counter = 0
                alarm_on = False
                stats["alert_frames"] += 1
                final_label = "Alert"
                final_color = (0, 255, 0)  # Green

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), final_color, 2)

            # Face label with confidence
            face_text = f"{DISPLAY_LABELS[class_id]} {confidence:.0%}"
            cv2.putText(frame, face_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, final_color, 2)
        else:
            # No face detected — reset
            drowsy_counter = 0
            alarm_on = False
            final_label = "No Face Detected"
            final_color = (200, 200, 200)  # Light gray

        # --- HUD ---
        draw_status_panel(
            frame, final_label, final_color,
            fps_tracker.fps, confidence,
            drowsy_counter, DROWSY_ALARM_THRESHOLD,
        )

        # --- Record ---
        if writer is not None:
            writer.write(frame)

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --- Cleanup ---
    session_duration = time.time() - session_start
    print("\nCleaning up...")
    cap.release()
    if writer is not None:
        writer.release()
        print(f"✓ Video saved: {args.save_video}")
    cv2.destroyAllWindows()

    # Remove temp alarm WAV
    if _alarm_wav_path and os.path.exists(_alarm_wav_path):
        try:
            os.remove(_alarm_wav_path)
        except OSError:
            pass

    # --- Session summary ---
    print("\n" + "=" * 45)
    print("  SESSION SUMMARY")
    print("=" * 45)
    print(f"  Duration         : {session_duration:.1f}s")
    print(f"  Total frames     : {stats['total_frames']}")
    print(f"  Avg FPS          : {stats['total_frames'] / max(session_duration, 0.001):.1f}")
    print(f"  Face detected    : {stats['face_detected_frames']} "
          f"({100 * stats['face_detected_frames'] / max(stats['total_frames'], 1):.1f}%)")
    print(f"  Alert frames     : {stats['alert_frames']}")
    print(f"  Drowsy frames    : {stats['drowsy_frames']}")
    print(f"  Alarm triggers   : {stats['alarm_triggers']}")
    print("=" * 45)
    print("Done.")


if __name__ == "__main__":
    main()
