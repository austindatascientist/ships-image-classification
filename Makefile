# Ships Image Classification Makefile
# ===================================

SHELL := /bin/bash
PYTHON := python3
VENV := venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

# YOLO model variants
VARIANT ?= n
MODEL_FILE := yolo11$(VARIANT)-cls.pt
MODEL_URL := https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11$(VARIANT)-cls.pt

# Directories
DATA_DIR := data
RUNS_DIR := runs

.PHONY: help setup download yolo best predict clean

# Default target
help:
	@echo "Ships Image Classification"
	@echo "=========================="
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  setup     Create venv and install deps (uses uv if available, ~10x faster)"
	@echo ""
	@echo "Data:"
	@echo "  download  Download satellite ship imagery dataset from Kaggle"
	@echo ""
	@echo "Training:"
	@echo "  yolo      Train YOLO model (default: nano)"
	@echo "            Use VARIANT=n/s/m/l/x for different model sizes"
	@echo ""
	@echo "Inference:"
	@echo "  best      Find and print the best trained model"
	@echo "  predict   Run prediction on an image"
	@echo "            Example: make predict IMAGE=path/to/image.png"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean     Remove venv, runs, and cached files"

# Create virtual environment and install dependencies
# Installs uv if not present for faster package installs
setup:
	@if ! command -v uv &> /dev/null; then \
		echo "Installing uv for faster package installs..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo ""; \
		echo "uv installed. Restart your shell or run: source ~/.local/bin/env"; \
		echo "Then run 'make setup' again."; \
		exit 0; \
	fi
	@echo "Using uv for fast setup..."
	uv venv $(VENV)
	uv pip install --python $(VENV_PYTHON) -r requirements.txt
	@echo ""
	@echo "Setup complete. Launching activated shell..."
	@exec bash -c "source $(VENV)/bin/activate && exec bash"

$(VENV)/bin/activate:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi

# Download dataset (prompts for Kaggle credentials only if not already configured)
download: $(VENV)/bin/activate
	@# Check if data already exists
	@if [ -d "$(DATA_DIR)/ship" ] && [ -n "$$(ls -A $(DATA_DIR)/ship 2>/dev/null)" ]; then \
		echo ""; \
		echo "Image files found in $(DATA_DIR)/"; \
		read -p "Proceed with downloading? [y/N]: " confirm; \
		if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
			echo "Download cancelled."; \
			exit 0; \
		fi; \
	fi
	@# Check for Kaggle credentials - env vars first, then kaggle.json, then prompt
	@if [ -n "$$KAGGLE_USERNAME" ] && [ -n "$$KAGGLE_KEY" ]; then \
		echo "Using Kaggle credentials from environment variables (user: $$KAGGLE_USERNAME)"; \
	elif [ -f ~/.kaggle/kaggle.json ]; then \
		username=$$(grep -o '"username":"[^"]*"' ~/.kaggle/kaggle.json | cut -d'"' -f4); \
		key=$$(grep -o '"key":"[^"]*"' ~/.kaggle/kaggle.json | cut -d'"' -f4); \
		if [ -n "$$username" ] && [ -n "$$key" ]; then \
			echo "Found Kaggle credentials for user: $$username"; \
		else \
			echo ""; \
			echo "=== Kaggle API Setup ==="; \
			echo ""; \
			echo "Existing credentials file is incomplete."; \
			echo "You can also set KAGGLE_USERNAME and KAGGLE_KEY environment variables."; \
			echo ""; \
			read -p "Kaggle username: " username; \
			read -p "Kaggle API key: " apikey; \
			if [ -z "$$username" ] || [ -z "$$apikey" ]; then \
				echo "Skipping credentials setup (empty input)."; \
			else \
				echo "{\"username\":\"$$username\",\"key\":\"$$apikey\"}" > ~/.kaggle/kaggle.json; \
				chmod 600 ~/.kaggle/kaggle.json; \
				echo "Kaggle credentials saved."; \
			fi; \
		fi; \
	else \
		echo ""; \
		echo "=== Kaggle API Setup ==="; \
		echo ""; \
		echo "To download datasets, you need Kaggle API credentials."; \
		echo ""; \
		echo "Option 1: Set environment variables"; \
		echo "  export KAGGLE_USERNAME=your_username"; \
		echo "  export KAGGLE_KEY=your_api_key"; \
		echo ""; \
		echo "Option 2: Create credentials file"; \
		echo "  1. Go to https://www.kaggle.com/settings"; \
		echo "  2. Click 'Create New Token' in the API section"; \
		echo "  3. Move downloaded kaggle.json to ~/.kaggle/"; \
		echo ""; \
		echo "Option 3: Enter credentials now:"; \
		echo ""; \
		read -p "Kaggle username (or press Enter to skip): " username; \
		read -p "Kaggle API key: " apikey; \
		if [ -z "$$username" ] || [ -z "$$apikey" ]; then \
			echo "Skipping credentials setup (empty input)."; \
		else \
			mkdir -p ~/.kaggle; \
			echo "{\"username\":\"$$username\",\"key\":\"$$apikey\"}" > ~/.kaggle/kaggle.json; \
			chmod 600 ~/.kaggle/kaggle.json; \
			echo ""; \
			echo "Kaggle credentials saved to ~/.kaggle/kaggle.json"; \
		fi; \
		echo ""; \
	fi
	@echo "Downloading satellite ship imagery dataset..."
	$(VENV_PYTHON) download.py

# Download YOLO model if not present, then train
yolo: $(VENV)/bin/activate
	@echo "Checking for model file: $(MODEL_FILE)"
	@if [ ! -f "$(MODEL_FILE)" ]; then \
		echo "Downloading $(MODEL_FILE)..."; \
		curl -L -o $(MODEL_FILE) $(MODEL_URL); \
		echo "Downloaded $(MODEL_FILE)"; \
	else \
		echo "Model file $(MODEL_FILE) already exists."; \
	fi
	@echo ""
	@echo "Starting training with $(MODEL_FILE)..."
	$(VENV_PYTHON) train.py $(VARIANT)

# Find and print best model
best: $(VENV)/bin/activate
	@$(VENV_PYTHON) best.py

# Run prediction
# Usage: make predict IMAGE=path/to/image.png [MODEL=path/to/model.onnx]
predict: $(VENV)/bin/activate
ifndef IMAGE
	@echo "Error: IMAGE not specified"
	@echo "Usage: make predict IMAGE=path/to/image.png"
	@echo "       make predict IMAGE=path/to/image.png MODEL=path/to/model.onnx"
	@exit 1
endif
ifdef MODEL
	$(VENV_PYTHON) predict.py "$(IMAGE)" -m "$(MODEL)"
else
	$(VENV_PYTHON) predict.py "$(IMAGE)"
endif

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf $(RUNS_DIR)
	rm -rf __pycache__ *.pyc
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned."
