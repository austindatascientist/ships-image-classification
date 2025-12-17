#!/usr/bin/env python3
"""Find and print the best trained model based on accuracy."""
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RUNS_DIR = PROJECT_ROOT / "runs" / "classify"


def find_best_model():
    """
    Find the best trained model based on top-1 accuracy.

    Returns:
        Tuple of (best_accuracy, best_model_path, best_run_name) or (None, None, None) if no models found
    """
    if not RUNS_DIR.exists():
        print(f"No runs directory found at: {RUNS_DIR}")
        return None, None, None

    best_accuracy = 0.0
    best_model_path = None
    best_run_name = None

    # Find all results.csv files
    for results_file in RUNS_DIR.rglob("results.csv"):
        run_dir = results_file.parent
        weights_dir = run_dir / "weights"
        best_onnx = weights_dir / "best.onnx"
        best_pt = weights_dir / "best.pt"

        # Check if model exists
        model_path = None
        if best_onnx.exists():
            model_path = best_onnx
        elif best_pt.exists():
            model_path = best_pt
        else:
            continue

        # Read accuracy from results.csv
        try:
            with open(results_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    continue

                # Get the best accuracy from the run (max of metrics/accuracy_top1)
                max_acc = 0.0
                for row in rows:
                    # Handle both possible column name formats
                    acc_str = row.get("metrics/accuracy_top1") or row.get("accuracy_top1", "0")
                    try:
                        acc = float(acc_str.strip())
                        max_acc = max(max_acc, acc)
                    except (ValueError, AttributeError):
                        continue

                if max_acc > best_accuracy:
                    best_accuracy = max_acc
                    best_model_path = model_path
                    best_run_name = run_dir.name

        except Exception as e:
            print(f"Warning: Could not read {results_file}: {e}")
            continue

    return best_accuracy, best_model_path, best_run_name


def main():
    """Print information about the best model."""
    accuracy, model_path, run_name = find_best_model()

    if model_path is None:
        print("No trained models found.")
        print(f"Run 'make yolo' to train a model first.")
        return 1

    print("=" * 50)
    print("BEST MODEL")
    print("=" * 50)
    print(f"Run:      {run_name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Path:     {model_path}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())
