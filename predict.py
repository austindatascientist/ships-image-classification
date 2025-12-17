#!/usr/bin/env python3
"""Run inference on images using trained ONNX models."""
import sys
import argparse
import warnings
import csv
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).parent
RUNS_DIR = PROJECT_ROOT / "runs" / "classify"
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "satellite-imagery-of-ships"
SHIPS_DIR = DATASET_DIR / "ship"

CLASS_NAMES = ["no_ship", "ship"]

# Session cache
_session_cache = {}


def find_best_model():
    """Find the best trained model based on top-1 accuracy."""
    if not RUNS_DIR.exists():
        return None

    best_accuracy = 0.0
    best_model_path = None

    for results_file in RUNS_DIR.rglob("results.csv"):
        run_dir = results_file.parent
        weights_dir = run_dir / "weights"
        best_onnx = weights_dir / "best.onnx"

        if not best_onnx.exists():
            continue

        try:
            with open(results_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    continue

                max_acc = 0.0
                for row in rows:
                    acc_str = row.get("metrics/accuracy_top1") or row.get("accuracy_top1", "0")
                    try:
                        acc = float(acc_str.strip())
                        max_acc = max(max_acc, acc)
                    except (ValueError, AttributeError):
                        continue

                if max_acc > best_accuracy:
                    best_accuracy = max_acc
                    best_model_path = best_onnx

        except Exception:
            continue

    return best_model_path


def softmax(x, axis=-1):
    """Compute softmax values for array x along specified axis."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def load_onnx_session(model_path):
    """Load an ONNX model and return session info."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_input = sess.get_inputs()[0]
    input_name = model_input.name

    shape = model_input.shape
    try:
        h = int(shape[2]) if isinstance(shape[2], int) else 80
        w = int(shape[3]) if isinstance(shape[3], int) else 80
    except (IndexError, TypeError):
        h, w = 80, 80

    return sess, input_name, (h, w)


def preprocess_image(path, size):
    """Load and preprocess an image for ONNX model inference."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        pil_img = Image.open(path).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"Cannot read image: {path}") from e

    resized = pil_img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = arr.transpose(2, 0, 1)[None, ...]
    return tensor, pil_img


def run_inference(sess, input_name, tensor, class_names=None):
    """Run inference on a preprocessed image tensor."""
    if class_names is None:
        class_names = CLASS_NAMES

    outputs = sess.run(None, {input_name: tensor})[0]
    logits = outputs[0] if outputs.ndim > 1 else outputs
    probs = softmax(logits)

    pred_idx = int(np.argmax(probs))
    pred_conf = float(probs[pred_idx])
    pred_label = class_names[pred_idx]

    return pred_label, pred_conf, probs


def _get_session(model_path):
    """Get or create cached ONNX session."""
    key = str(model_path)
    if key not in _session_cache:
        _session_cache[key] = load_onnx_session(model_path)
    return _session_cache[key]


def predict_image(image_path, model_path=None):
    """
    Run prediction on a single image.

    Args:
        image_path: Path to the image file
        model_path: Path to ONNX model (defaults to best model)

    Returns:
        pred_label, pred_conf, probs
    """
    if model_path is None:
        model_path = find_best_model()
        if model_path is None:
            raise FileNotFoundError("No trained model found. Run 'make yolo' first.")

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    sess, input_name, (h, w) = _get_session(model_path)
    tensor, _ = preprocess_image(image_path, h)
    return run_inference(sess, input_name, tensor)


def predict_and_display(image_path, model_path=None, show_plot=False):
    """Run prediction and display results."""
    if model_path is None:
        model_path = find_best_model()
        if model_path is None:
            raise FileNotFoundError("No trained model found. Run 'make yolo' first.")

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Using model: {model_path}")
    print(f"Running inference on: {image_path}")

    sess, input_name, (h, w) = _get_session(model_path)
    print(f"Model expects input size: {h}x{w}")

    tensor, pil_img = preprocess_image(image_path, h)
    pred_label, pred_conf, probs = run_inference(sess, input_name, tensor)

    print("=" * 50)
    print(f"Image:        {image_path}")
    print(f"Prediction:   {pred_label}")
    print(f"Confidence:   {pred_conf * 100:.2f}%")
    print("Class probabilities:")
    for name, p in zip(CLASS_NAMES, probs):
        print(f"  {name:<12} {p * 100:.2f}%")
    print("=" * 50)

    if show_plot:
        import matplotlib.pyplot as plt
        plt.imshow(pil_img)
        plt.axis("off")
        plt.title(f"{pred_label} ({pred_conf * 100:.2f}% confidence)")
        plt.show()

    return pred_label, pred_conf, probs


def main():
    parser = argparse.ArgumentParser(description="Run ship classification prediction")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("-m", "--model", help="Path to ONNX model (default: best model)")
    parser.add_argument("--show", action="store_true", help="Show matplotlib plot")

    args = parser.parse_args()

    if args.image is None:
        # Use sample image if available
        sample = SHIPS_DIR / "ship_000000.png"
        if sample.exists():
            args.image = sample
        else:
            parser.error("No image specified and no sample image found")

    predict_and_display(args.image, model_path=args.model, show_plot=args.show)


if __name__ == "__main__":
    main()
