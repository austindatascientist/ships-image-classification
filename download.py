#!/usr/bin/env python3
"""Download datasets from Kaggle."""
import shutil
from pathlib import Path
import kagglehub
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DATASET = "apollo2506/satellite-imagery-of-ships"


def download_dataset(kaggle_path=DEFAULT_DATASET, target_dir=None):
    """
    Download a dataset from Kaggle.

    Args:
        kaggle_path: Kaggle dataset path (e.g., 'apollo2506/satellite-imagery-of-ships')
        target_dir: Target directory for the dataset (defaults to DATA_DIR)

    Returns:
        Path to the downloaded dataset
    """
    if not kaggle_path or "/" not in kaggle_path:
        raise ValueError(
            f"Invalid Kaggle path: '{kaggle_path}'. "
            "Expected format: 'username/dataset-name'"
        )

    target_dir = Path(target_dir) if target_dir else DATA_DIR

    print(f"Downloading dataset: {kaggle_path}")
    path = kagglehub.dataset_download(kaggle_path)
    print(f"Downloaded to: {path}")

    target_dir.mkdir(parents=True, exist_ok=True)

    # Collect all files to copy
    source_path = Path(path)
    files = list(source_path.rglob("*"))
    files = [f for f in files if f.is_file()]

    # Copy with progress bar
    with tqdm(files, desc="Copying files", unit="file") as pbar:
        for src_file in pbar:
            rel_path = src_file.relative_to(source_path)
            dst_file = target_dir / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)

    print(f"Dataset copied to: {target_dir}")
    return target_dir


if __name__ == "__main__":
    download_dataset()
