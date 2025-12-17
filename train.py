#!/usr/bin/env python3
"""Train YOLO classification models on ship imagery dataset."""
import os
import warnings
import logging

# Suppress GPU/NVML warnings and disable auto-install (CPU-only setup)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["YOLO_AUTOINSTALL"] = "false"  # Prevent auto-installing nvidia-ml-py
warnings.filterwarnings("ignore", message=".*pynvml.*")
warnings.filterwarnings("ignore", message=".*NVML.*")


class NVMLFilter(logging.Filter):
    """Filter out GPU/NVML related log messages."""

    def filter(self, record):
        msg = record.getMessage().lower()
        return not any(x in msg for x in ["pynvml", "nvml", "idle gpu"])


logging.getLogger("ultralytics").addFilter(NVMLFilter())

import sys
import shutil
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "satellite-imagery-of-ships"
DATASET_DIR_SPLIT = DATA_DIR / "satellite-imagery-of-ships_split"
RUNS_DIR = PROJECT_ROOT / "runs"

# Training defaults
DEFAULT_IMGSZ = 96
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_TRAIN_SPLIT = 0.8


def cleanup_runs(runs_dir=None):
    """Remove previous training runs directory."""
    runs_dir = runs_dir or RUNS_DIR
    if runs_dir.exists():
        print(f"[cleanup] Removing previous runs directory at: {runs_dir}")
        shutil.rmtree(runs_dir)


def train_classifier(
    model_variant="yolo11n-cls.pt",
    data_dir=None,
    epochs=None,
    patience=None,
    imgsz=None,
    batch=None,
    split=None,
    device=-1,
    cleanup=False,
):
    """
    Train a YOLO classification model.

    Args:
        model_variant: YOLO model to use (e.g., yolo11n-cls.pt, yolo11x-cls.pt)
        data_dir: Path to dataset directory
        epochs: Number of training epochs
        patience: Early stopping patience
        imgsz: Input image size
        batch: Batch size
        split: Train/validation split ratio
        device: Device to use (-1 for CPU, 0+ for GPU)
        cleanup: Whether to remove previous runs before training

    Returns:
        Trained YOLO model
    """
    # Use pre-split directory if it exists (avoids "split=train not found" warning)
    if data_dir:
        data_dir = Path(data_dir)
    elif DATASET_DIR_SPLIT.exists() and (DATASET_DIR_SPLIT / "train").exists():
        data_dir = DATASET_DIR_SPLIT
    else:
        data_dir = DATASET_DIR
    epochs = epochs or DEFAULT_EPOCHS
    patience = patience or DEFAULT_PATIENCE
    imgsz = imgsz or DEFAULT_IMGSZ
    batch = batch or DEFAULT_BATCH_SIZE
    split = split or DEFAULT_TRAIN_SPLIT

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"Data directory has no class subdirectories: {data_dir}")

    if cleanup:
        cleanup_runs()

    model = YOLO(model_variant)

    model.train(
        data=str(data_dir),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        task="classify",
        split=split,
        device=device,
    )

    return model


def validate_model(model):
    """Run validation on a trained model."""
    return model.val()


def export_model(model, format="onnx"):
    """Export model to specified format."""
    try:
        return model.export(format=format)
    except ModuleNotFoundError as e:
        print(f"Skipping {format.upper()} export: {e}")
        print("Install with: pip install onnx")
        return None


if __name__ == "__main__":
    # Get model variant from command line (default: n for nano)
    variant = sys.argv[1] if len(sys.argv) > 1 else "n"
    model_file = f"yolo11{variant}-cls.pt"

    print(f"Training with model: {model_file}")
    model = train_classifier(model_variant=model_file)
    validate_model(model)
    exported = export_model(model)
    if exported:
        print("Training complete. Model exported to ONNX.")
    else:
        print("Training complete.")
