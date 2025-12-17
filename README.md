# Ships Image Classification

A YOLO11-based system for classifying ships in satellite imagery.

## Features

- Download satellite ship imagery dataset(s) via Kaggle API
- Train YOLO11 classification models (nano to extra-large variants)
- Export trained models to ONNX format for portable inference
- Run predictions on individual images with confidence scores
- Automatic best model selection based on accuracy

## Development Environment

This project was developed and tested in a WSL Debian environment.

## Getting Started

Follow these steps to train a ship classifier.

### Step 1: Setup Environment

Clone the repository and install dependencies:

```bash
git clone https://github.com/austindatascientist/ships-image-classification.git
cd ships-image-classification
make setup
```

### Step 2: Download the Dataset

Downloads the [Satellite Imagery of Ships](https://www.kaggle.com/datasets/apollo2506/satellite-imagery-of-ships) dataset from Kaggle:

```bash
make download
```

If Kaggle API credentials are not configured, you'll be prompted to set them up with instructions.

The dataset contains 80x80 pixel satellite/aerial images in two categories:
- `ship/` - Images containing ships
- `no-ship/` - Images without ships

### Step 3: Train a Model

Train a YOLO11 classification model (automatically downloads pretrained weights if needed):

```bash
make yolo
```

Training outputs are saved to `runs/classify/` including model weights and metrics.

To find your best trained model:

```bash
make best
```

Example output:
```
==================================================
BEST MODEL
==================================================
Run:      train
Accuracy: 99.85%
Path:     runs/classify/train/weights/best.onnx
==================================================
```

### Step 4: Run Predictions

Run inference on an image (automatically uses the best model):

```bash
make predict IMAGE=data/satellite-imagery-of-ships/ship/ship_000000.png
```

Example output:
```
Using model: runs/classify/train/weights/best.onnx
Running inference on: data/satellite-imagery-of-ships/ship/ship_000000.png
Model expects input size: 80x80
==================================================
Image:        data/satellite-imagery-of-ships/ship/ship_000000.png
Prediction:   ship
Confidence:   97.82%
Class probabilities:
  no_ship      2.18%
  ship         97.82%
==================================================
```

Or specify a custom model:

```bash
make predict IMAGE=path/to/image.png MODEL=path/to/model.onnx
```

## Model Variants

YOLO11 classification models available:

| Model | Size | Parameters | Make Command |
|-------|------|------------|--------------|
| yolo11n-cls.pt | Nano | ~2.7M | `make yolo` or `make yolo VARIANT=n` |
| yolo11s-cls.pt | Small | ~9.5M | `make yolo VARIANT=s` |
| yolo11m-cls.pt | Medium | ~23M | `make yolo VARIANT=m` |
| yolo11l-cls.pt | Large | ~28M | `make yolo VARIANT=l` |
| yolo11x-cls.pt | Extra-Large | ~59M | `make yolo VARIANT=x` |

On my computer, the nano model trained 8x faster and achieved about 0.05% higher accuracy than the X model because the larger model's approximately 59M parameters were excessive for this binary classification task, leading to overfitting.

## Cleanup

To remove the virtual environment, training runs, and cache files:

```bash
make clean
```

## Author

Austin Pacheco-Timmerman
