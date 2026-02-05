#!/usr/bin/env python3
"""
train_exercise_classifier.py

Train a lightweight "exercise classifier" on YOLOv8-pose keypoints extracted from videos.

This produces:
  - exercise_float.tflite  (optional)
  - exercise_int8.tflite   (FULL integer quantized; suitable for MCU)
  - metadata JSON (classes, window length, normalization, quant params)

Recommended deployment on Nuvoton M55M1:
  1) Run YOLOv8n-pose (INT8, Vela compiled) on the NPU to get 17 keypoints (x,y,conf).
  2) Run THIS model on the keypoint vector/window (INT8, Vela compiled) to classify exercise.

Dataset layout (same as your yolodnnmodel.py):
  data_dir/
    pushups/
      vid1.mp4
    squats/
      vid2.mp4

Key ideas:
  - Extract 17 keypoints per sampled frame using Ultralytics YOLOv8 pose
  - Normalize x,y to [0,1] by frame width/height
  - Build temporal windows of length T (default 16 frames) with stride S (default 4)
  - Train a tiny MLP (Dense/ReLU) on windows: Flatten(T*51) -> Dense -> Dense -> Softmax
  - Export FULL INT8 TFLite using a representative dataset

Notes:
  - For best on-device fidelity, optionally simulate your board camera by resizing frames
    BEFORE pose extraction (e.g., --simulate-camera 256 to downsample/upsample to 256x256).
  - We split by VIDEO (not by frame) to avoid leakage.

Dependencies:
  pip install ultralytics opencv-python numpy scikit-learn tensorflow

Example:
  python train_exercise_classifier.py \\
    --data-dir /home/ed/aifitnessmirror/data/videos \\
    --yolo yolov8n-pose.pt \\
    --imgsz 256 \\
    --simulate-camera 256 \\
    --sample-rate 2 \\
    --window 16 --stride 4 \\
    --epochs 30 \\
    --out-dir /home/ed/aifitnessmirror/output/exercise_model
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# TensorFlow is used for easy INT8 TFLite export.
import tensorflow as tf


NUM_KEYPOINTS = 17
KEYPOINT_FEATURES = 3
FEATURE_DIM = NUM_KEYPOINTS * KEYPOINT_FEATURES  # 51


def load_training_data(data_dir: Path) -> Dict[str, object]:
    """
    Expected structure:
      data_dir/
        classA/
          *.mp4
        classB/
          *.mp4
    Returns:
      { 'videos': {class: [Path...]}, 'classes': [class...] }
    """
    data = {"videos": {}, "classes": []}
    if not data_dir.exists():
        raise FileNotFoundError(f"Training data directory not found: {data_dir}")

    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    for class_folder in sorted(data_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        cls = class_folder.name
        vids = [p for p in sorted(class_folder.iterdir()) if p.suffix.lower() in video_exts]
        if not vids:
            continue
        data["classes"].append(cls)
        data["videos"][cls] = vids
        print(f"Class '{cls}': {len(vids)} videos")
    return data


def maybe_resize_frame(frame: np.ndarray, simulate_camera: Optional[int]) -> np.ndarray:
    """
    If simulate_camera is set (e.g. 256), we resize frames to simulate a low-res camera.
    This helps match the jitter/noise you'll see on-device.
    """
    if not simulate_camera:
        return frame
    s = int(simulate_camera)
    if s <= 0:
        return frame
    return cv2.resize(frame, (s, s), interpolation=cv2.INTER_AREA)


def extract_keypoints_from_video(
    video_path: Path,
    yolo_model: YOLO,
    sample_rate: int = 5,
    pose_confidence: float = 0.3,
    yolo_device: Optional[str] = None,
    imgsz: int = 256,
    simulate_camera: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Returns a list of per-frame keypoint vectors of shape (51,) float32:
      [x1, y1, c1, x2, y2, c2, ...] where x,y are normalized to [0,1] in the frame space.

    NOTE: Uses ONLY the first detected person.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to load video: {video_path}")
        return []

    seq: List[np.ndarray] = []
    frame_idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_rate == 0:
            frame_proc = maybe_resize_frame(frame, simulate_camera)
            h, w = frame_proc.shape[:2]

            results = yolo_model.predict(
                frame_proc,
                verbose=False,
                device=yolo_device,
                imgsz=imgsz,
            )
            if len(results) > 0 and results[0].keypoints is not None:
                kdata = results[0].keypoints.data
                if len(kdata) > 0:
                    kpts = kdata[0].detach().cpu().numpy().astype(np.float32)  # (17,3) in pixels
                    if float(np.mean(kpts[:, 2])) >= pose_confidence:
                        kpts[:, 0] /= max(w, 1)
                        kpts[:, 1] /= max(h, 1)
                        seq.append(kpts.flatten())
        frame_idx += 1

    cap.release()
    return seq


def build_windows(
    keypoints_seq: List[np.ndarray],
    window: int,
    stride: int,
) -> List[np.ndarray]:
    """
    Turn a sequence of (51,) vectors into windows of shape (window, 51).
    """
    if window <= 0:
        raise ValueError("window must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    n = len(keypoints_seq)
    if n < window:
        return []
    out: List[np.ndarray] = []
    for start in range(0, n - window + 1, stride):
        w = np.stack(keypoints_seq[start : start + window], axis=0).astype(np.float32)  # (T,51)
        out.append(w)
    return out


def prepare_dataset(
    training_data: Dict[str, object],
    yolo_model: YOLO,
    sample_rate: int,
    pose_confidence: float,
    imgsz: int,
    yolo_device: Optional[str],
    window: int,
    stride: int,
    simulate_camera: Optional[int],
    max_videos_per_class: Optional[int] = None,
) -> Tuple[List[str], List[Dict[str, object]]]:
    """
    Extract windows per video. Returns:
      classes: list[str]
      per_video: list of dicts:
        { 'video': str, 'class': str, 'windows': List[np.ndarray] }
    """
    classes = list(training_data["classes"])
    per_video: List[Dict[str, object]] = []

    for cls in classes:
        vids: List[Path] = training_data["videos"][cls]  # type: ignore
        if max_videos_per_class:
            vids = vids[: max_videos_per_class]

        for vp in vids:
            seq = extract_keypoints_from_video(
                vp,
                yolo_model=yolo_model,
                sample_rate=sample_rate,
                pose_confidence=pose_confidence,
                yolo_device=yolo_device,
                imgsz=imgsz,
                simulate_camera=simulate_camera,
            )
            wins = build_windows(seq, window=window, stride=stride)
            if wins:
                per_video.append({"video": str(vp), "class": cls, "windows": wins})
            else:
                print(f"[skip] {vp} (not enough pose frames: {len(seq)} for window={window})")

        print(f"Done class '{cls}'")

    return classes, per_video


def split_by_video(
    per_video: List[Dict[str, object]],
    classes: List[str],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Split by video, roughly stratified by class.
    """
    rng = random.Random(seed)
    by_class: Dict[str, List[Dict[str, object]]] = {c: [] for c in classes}
    for item in per_video:
        by_class[item["class"]].append(item)

    train_items: List[Dict[str, object]] = []
    val_items: List[Dict[str, object]] = []
    for c in classes:
        items = by_class[c]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio)) if len(items) >= 2 else 0
        val_items.extend(items[:n_val])
        train_items.extend(items[n_val:])
    return train_items, val_items


def flatten_windows(items: List[Dict[str, object]], class_to_idx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert per-video window lists into big arrays:
      X: (N, T, 51)
      y: (N,)
    """
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for it in items:
        cls = str(it["class"])
        idx = class_to_idx[cls]
        wins: List[np.ndarray] = it["windows"]  # type: ignore
        for w in wins:
            X_list.append(w.astype(np.float32))
            y_list.append(idx)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def build_mlp_window_model(T: int, D: int, num_classes: int) -> tf.keras.Model:
    """
    MCU-friendly model: Flatten(T*D) -> Dense/ReLU -> Dense/ReLU -> Dense/Softmax
    """
    inputs = tf.keras.Input(shape=(T, D), name="keypoints")
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


@dataclass
class TrainCfg:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 8


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainCfg,
) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, cfg.patience // 2)
        ),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return model


def convert_to_int8_tflite(
    model: tf.keras.Model,
    rep_data: np.ndarray,
    tflite_out: Path,
    num_calib: int = 500,
) -> Tuple[Path, Dict[str, object]]:
    """
    Full integer quantization with representative dataset.
    Writes `tflite_out` and returns (path, quant_info).
    """
    tflite_out.parent.mkdir(parents=True, exist_ok=True)

    rep_data = rep_data.astype(np.float32)
    n = min(num_calib, rep_data.shape[0])

    def representative_dataset():
        for i in range(n):
            yield [rep_data[i : i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    tflite_out.write_bytes(tflite_model)

    # Read quant params for convenience
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    quant_info = {
        "input": {
            "name": inp.get("name"),
            "shape": list(inp.get("shape", [])),
            "dtype": str(inp.get("dtype")),
            "scale": float(inp.get("quantization_parameters", {}).get("scales", [0.0])[0])
            if inp.get("quantization_parameters") else float(inp.get("quantization", (0.0, 0))[0]),
            "zero_point": int(inp.get("quantization_parameters", {}).get("zero_points", [0])[0])
            if inp.get("quantization_parameters") else int(inp.get("quantization", (0.0, 0))[1]),
        },
        "output": {
            "name": out.get("name"),
            "shape": list(out.get("shape", [])),
            "dtype": str(out.get("dtype")),
            "scale": float(out.get("quantization_parameters", {}).get("scales", [0.0])[0])
            if out.get("quantization_parameters") else float(out.get("quantization", (0.0, 0))[0]),
            "zero_point": int(out.get("quantization_parameters", {}).get("zero_points", [0])[0])
            if out.get("quantization_parameters") else int(out.get("quantization", (0.0, 0))[1]),
        },
    }
    return tflite_out, quant_info


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train keypoint-only exercise classifier and export INT8 TFLite")
    p.add_argument("--data-dir", type=Path, required=True, help="Root with class subfolders of videos")
    p.add_argument("--yolo", type=str, default="yolov8n-pose.pt", help="YOLOv8 pose weights/name")
    p.add_argument("--yolo-device", type=str, default=None, help="Ultralytics device string (e.g. '0', 'cpu')")
    p.add_argument("--imgsz", type=int, default=256, help="YOLO inference size (try 256/192/160 for speed)")
    p.add_argument("--simulate-camera", type=int, default=None, help="Resize frames to NxN before YOLO (e.g. 256)")
    p.add_argument("--sample-rate", type=int, default=2, help="Process every Nth frame")
    p.add_argument("--pose-confidence", type=float, default=0.3, help="Avg keypoint confidence threshold")
    p.add_argument("--window", type=int, default=16, help="Window length in frames (T)")
    p.add_argument("--stride", type=int, default=4, help="Stride between windows")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (split by video)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--max-videos-per-class", type=int, default=None, help="Limit videos per class (debug)")

    # training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)

    # outputs
    p.add_argument("--out-dir", type=Path, required=True, help="Output folder for model + metadata")
    p.add_argument("--name", type=str, default="exercise", help="Base name for outputs")
    p.add_argument("--calib-samples", type=int, default=500, help="Representative samples for INT8 quant")
    return p


def main():
    args = build_argparser().parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO pose model: {args.yolo}")
    yolo = YOLO(args.yolo)

    training_data = load_training_data(args.data_dir)
    classes: List[str] = training_data["classes"]  # type: ignore
    if not classes:
        raise SystemExit("No classes found. Add class subfolders with videos.")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print("\nExtracting keypoints + building windows...")
    classes, per_video = prepare_dataset(
        training_data,
        yolo_model=yolo,
        sample_rate=args.sample_rate,
        pose_confidence=args.pose_confidence,
        imgsz=args.imgsz,
        yolo_device=args.yolo_device,
        window=args.window,
        stride=args.stride,
        simulate_camera=args.simulate_camera,
        max_videos_per_class=args.max_videos_per_class,
    )
    if not per_video:
        raise SystemExit("No usable videos/windows. Try lowering --window or --pose-confidence, or add more data.")

    train_items, val_items = split_by_video(per_video, classes, val_ratio=args.val_ratio, seed=args.seed)
    if not train_items or not val_items:
        # fallback: simple random split of videos
        print("Warning: not enough videos per class for stratified split. Falling back to random split.")
        rng = random.Random(args.seed)
        rng.shuffle(per_video)
        n_val = max(1, int(len(per_video) * args.val_ratio))
        val_items = per_video[:n_val]
        train_items = per_video[n_val:]

    X_train, y_train = flatten_windows(train_items, class_to_idx)
    X_val, y_val = flatten_windows(val_items, class_to_idx)

    print("\nDataset summary:")
    print(f"  Classes: {classes}")
    print(f"  Train videos: {len(train_items)}, Val videos: {len(val_items)}")
    print(f"  Train windows: {X_train.shape[0]}, Val windows: {X_val.shape[0]}")
    print(f"  Input shape: {X_train.shape[1:]} (T={args.window}, D={FEATURE_DIM})")

    print("\nBuilding model...")
    model = build_mlp_window_model(args.window, FEATURE_DIM, num_classes=len(classes))
    model.summary()

    print("\nTraining...")
    cfg = TrainCfg(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience)
    model = train_model(model, X_train, y_train, X_val, y_val, cfg)

    # Save float model as TFLite too (optional but handy for debugging)
    float_tflite_path = args.out_dir / f"{args.name}_float.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_tflite_path.write_bytes(converter.convert())

    # Save INT8 model
    int8_path = args.out_dir / f"{args.name}_int8.tflite"
    int8_path, quant_info = convert_to_int8_tflite(
        model, rep_data=X_train, tflite_out=int8_path, num_calib=args.calib_samples
    )

    print("\nDone.")
    print(f"Float TFLite: {float_tflite_path}")
    print(f"INT8 TFLite:  {int8_path}")
    print(f"Metadata:     {meta_path}")
    print("\nNext step for M55M1:")
    print("  Run Vela on the INT8 model (accelerator config depends on your Ethos-U):")
    print(f"    vela {int8_path} --accelerator-config ethos-u55-256")
    print("  Then copy the Vela output TFLite to your SD card (e.g., EXERCISE_vela.tflite) and load it in Keil.")


if __name__ == "__main__":
    # Make TF less noisy
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
