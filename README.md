# Spatial Transcriptomics Machine Learning Pipeline

A comprehensive machine learning pipeline for predicting spatial gene expression from histology images in spatial transcriptomics research.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
  - [Model Architectures](#model-architectures)
  - [Training Capabilities](#training-capabilities)
  - [Experiment Management](#experiment-management)
  - [Evaluation & Analysis](#evaluation--analysis)
- [Architecture](#architecture)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Quick Install](#quick-install)
  - [Full Installation with Extras](#full-installation-with-extras)
  - [Docker Installation (Alternative)](#docker-installation-alternative)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
  - [Example Configuration (YAML)](#example-configuration-yaml)
  - [Configuration Schema](#configuration-schema)
- [Examples](#examples)
  - [1. Basic Training Script](#1-basic-training-script)
  - [2. Quantum Model Example](#2-quantum-model-example)
- [Contributing](#contributing)
  - [Getting Started](#getting-started)
  - [Development Guidelines](#development-guidelines)
  - [Running Tests](#running-tests)
  - [Code Quality](#code-quality)
- [License](#license)
- [Citation](#citation)

## Overview

This pipeline provides a flexible framework for analyzing spatial transcriptomics data by predicting gene expression patterns from histology images. It combines classical deep learning approaches with cutting-edge quantum machine learning techniques to advance our understanding of spatial gene expression.

The system is designed to be modular, extensible, and production-ready, supporting various model architectures, training strategies, and evaluation methods.

## Features

### Model Architectures

- **Classical Models:** EfficientNet-based architectures, Auxiliary Network models  
- **Quantum Models:** Amplitude embedding quantum machine learning models  
- **Extensible Design:** Easy to add new model architectures  

### Training Capabilities

- **Multiple Training Strategies:** Supervised learning, transfer learning  
- **Advanced Training Features:** Early stopping, learning rate scheduling, gradient clipping  
- **Cross-Validation:** Built-in k-fold cross-validation support  
- **Callbacks:** Model checkpointing, learning rate reduction, plotting  

### Experiment Management

- **Comprehensive Tracking:** Metric logging, configuration management, result visualization  
- **Reproducibility:** Random seed control, detailed logging  
- **Flexible Configuration:** YAML/JSON-based configuration system  

### Evaluation & Analysis

- **Rich Metrics:** MSE, MAE, RMSE, correlation coefficients  
- **Visualization:** Training curves, correlation heatmaps, prediction plots  
- **Export Options:** CSV, JSON, model checkpoints  

## Architecture

```text
spatial-transcriptomics-pipeline/
â”œâ”€â”€ configs/                 # Configuration files
â”œâ”€â”€ data/                    # Data storage (not included in repo)
â”œâ”€â”€ logs/                    # Log files
â”œâ”€â”€ results/                 # Experiment results
â”œâ”€â”€ src/                     # Source code
â”‚   â”œâ”€â”€ core/               # Core utilities and registry
â”‚   â”œâ”€â”€ data/               # Data loading and preprocessing
â”‚   â”œâ”€â”€ models/             # Model definitions and factories
â”‚   â”‚   â”œâ”€â”€ classical/      # Classical deep learning models
â”‚   â”‚   â””â”€â”€ quantum/        # Quantum machine learning models
â”‚   â”œâ”€â”€ training/           # Training components and utilities
â”‚   â””â”€â”€ utils/              # Utility functions and helpers
â”œâ”€â”€ tests/                   # Unit and integration tests
â””â”€â”€ README.md
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager
- (Optional) CUDA-compatible GPU for accelerated training

### Quick Install

```bash
# Clone the repository
git clone https://github.com/spatial-transcriptomics/research-pipeline.git
cd research-pipeline

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Full Installation with Extras

```bash
# Install with development tools
pip install -e .[dev]

# Install with quantum computing support
pip install -e .[quantum]

# Install everything
pip install -e .[dev,quantum,docs]
```

### Docker Installation (Alternative)

```bash
# Build Docker image
docker build -t spatial-transcriptomics-pipeline .

# Run container
docker run -it --gpus all spatial-transcriptomics-pipeline
```

# Build and Run Instructions

## Building the Docker Image

```bash
# Build the image
docker build -t spatial-transcriptomics-pipeline .

# Or using docker-compose
docker-compose build
```

## Running with Docker

```bash
# Run basic help command
docker run --rm spatial-transcriptomics-pipeline

# Run training with GPU support
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/results:/app/results \
  spatial-transcriptomics-pipeline \
  --config configs/example_config.yaml --mode train

# Run cross-validation
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/results:/app/results \
  spatial-transcriptomics-pipeline \
  --config configs/example_config.yaml --mode cross_validate
```

## Using Docker Compose

```bash
# Run CPU version
docker-compose run spatial-pipeline-cpu \
  --config configs/example_config.yaml --mode train

# Run GPU version
docker-compose run spatial-pipeline-gpu \
  --config configs/example_config.yaml --mode train

# Start Jupyter Lab
docker-compose up jupyter
```

## Interactive Development Environment

```bash
# Run interactive shell
docker run -it --rm --gpus all \
  -v $(pwd):/app \
  spatial-transcriptomics-pipeline \
  bash

# Inside container, you can run Python directly
# python src/main.py --config configs/example_config.yaml
```

## Docker Image Variants

For different use cases, you might want to create specialized Dockerfiles:

### Minimal CPU Version (Dockerfile.cpu)

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p data logs results configs

# Entry point
ENTRYPOINT ["python", "src/main.py"]
CMD ["--help"]
```

## Usage

### Command Line Interface

```bash
# Basic training
spatial-train --config configs/example_config.yaml --mode train

# Cross-validation
spatial-train --config configs/example_config.yaml --mode cross_validate

# Verbose output
spatial-train --config configs/example_config.yaml --mode train --verbose
```

### Python API

```python
from src.models.factory import ModelFactory
from src.training.factory import TrainerFactory
from src.utils.result_tracker import initialize_tracker

# Initialize experiment tracking
tracker = initialize_tracker("my_experiment")

# Create model
model = ModelFactory.create_model("classical_efficientnet", model_config)

# Create trainer
trainer = TrainerFactory.create_trainer("supervised_trainer", model, trainer_config)

# Train model
results = trainer.train()

# Evaluate model
evaluation = trainer.evaluate()
```

## Project Structure

```text
src/
â”œâ”€â”€ core/
â”‚   â”œâ”€â”€ factory_registry.py     # Global component registry
â”‚   â””â”€â”€ __init__.py
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ dataset.py              # Spatial transcriptomics dataset
â”‚   â”œâ”€â”€ preprocessing.py        # Data preprocessing utilities
â”‚   â””â”€â”€ __init__.py
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ factory.py              # Model factory
â”‚   â”œâ”€â”€ base_model.py           # Abstract base model
â”‚   â”œâ”€â”€ classical/
â”‚   â”‚   â”œâ”€â”€ efficientnet_model.py
â”‚   â”‚   â”œâ”€â”€ auxnet_model.py
â”‚   â”‚   â””â”€â”€ __init__.py
â”‚   â”œâ”€â”€ quantum/
â”‚   â”‚   â”œâ”€â”€ amplitude_embedding_qml.py
â”‚   â”‚   â”œâ”€â”€ quantum_layers.py
â”‚   â”‚   â””â”€â”€ __init__.py
â”‚   â””â”€â”€ __init__.py
â”œâ”€â”€ training/
â”‚   â”œâ”€â”€ factory.py              # Trainer factory
â”‚   â”œâ”€â”€ base_trainer.py         # Abstract base trainer
â”‚   â”œâ”€â”€ supervised_trainer.py   # Supervised trainer
â”‚   â”œâ”€â”€ callbacks.py            # Training callbacks
â”‚   â”œâ”€â”€ cross_validator.py      # Cross-validation framework
â”‚   â””â”€â”€ __init__.py
â””â”€â”€ utils/
    â”œâ”€â”€ result_tracker.py       # Experiment tracking
    â””â”€â”€ __init__.py
```

## Configuration

### Example Configuration (YAML)

```yaml
# experiment settings
experiment_name: "spatial_transcriptomics_experiment"
random_seed: 42

# model configuration
model:
  name: "classical_efficientnet"
  architecture: "efficientnet_b0"
  input_channels: 3
  input_height: 224
  input_width: 224
  output_genes: 250
  pretrained: true
  dropout_rate: 0.2

# training configuration
trainer:
  name: "supervised_trainer"
  training:
    epochs: 100
    batch_size: 32
    learning_rate: 0.001
    optimizer: "adam"
    loss_function: "mse"
    weight_decay: 0.0001
    gradient_clip: 1.0
    validation_split: 0.2
    early_stopping_patience: 20
  execution:
    cuda_enabled: true
  results:
    base_dir: "results/"

# data configuration
data:
  train_path: "data/train/"
  val_path: "data/validation/"
  test_path: "data/test/"
  num_workers: 4

# cross-validation settings
cross_validation:
  folds: 5
  type: "kfold"
  shuffle: true
  random_state: 42
```

### Configuration Schema

The pipeline supports extensive configuration customization:

- **Model Parameters:** Architecture-specific settings, input/output dimensions  
- **Training Parameters:** Optimizer settings, loss functions, regularization  
- **Data Parameters:** Paths, preprocessing settings, augmentation  
- **System Parameters:** Device settings, resource allocation  
- **Experiment Parameters:** Random seeds, naming conventions, logging  

## Examples

### 1. Basic Training Script

```python
#!/usr/bin/env python3
"""
Example training script
"""

import yaml
from src.models.factory import ModelFactory
from src.training.factory import TrainerFactory
from src.utils.result_tracker import initialize_tracker

def main():
    # Load configuration
    with open("configs/basic_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize tracking
    tracker = initialize_tracker(config["experiment_name"])
    tracker.log_config(config)

    # Create and train model
    model = ModelFactory.create_model(
        config["model"]["name"], 
        config["model"]
    )

    trainer = TrainerFactory.create_trainer(
        config["trainer"]["name"],
        model,
        config["trainer"]
    )

    # Execute training
    results = trainer.train()

    # Log final results
    tracker.log_metrics({
        "final_loss": results["best_val_loss"],
        "training_time": results["training_time"]
    })

    print(f"Training completed with loss: {results['best_val_loss']}")

if __name__ == "__main__":
    main()
```

### 2. Quantum Model Example

```python
# Requires quantum computing dependencies
try:
    from src.models.factory import ModelFactory

    # Create quantum model
    quantum_config = {
        "name": "quantum_amplitude_embedding",
        "n_qubits": 8,
        "n_genes": 100,
        "layers": 3
    }

    model = ModelFactory.create_model(
        "quantum_amplitude_embedding",
        quantum_config
    )

    print("Quantum model created successfully")
except ImportError:
    print("Quantum computing dependencies not available")
```

## Contributing

We welcome contributions to enhance the pipeline! Here's how you can help:

### Getting Started

1. Fork the repository  
2. Create a feature branch: `git checkout -b feature/amazing-feature`  
3. Commit your changes: `git commit -m 'Add amazing feature'`  
4. Push to the branch: `git push origin feature/amazing-feature`  
5. Open a pull request  

### Development Guidelines

- Follow PEP 8 coding standards
- Write unit tests for new functionality
- Document public APIs
- Use type hints
- Maintain backward compatibility

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py
```

### Code Quality

```bash
# Format code
black src/

# Check style
flake8 src/

# Type checking
mypy src/
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Citation

If you use this pipeline in your research, please cite our work:

```bibtex
@article{spatial_transcriptomics_pipeline2023,
  title={A Flexible Machine Learning Pipeline for Spatial Transcriptomics Analysis},
  author={Research Team, Spatial Transcriptomics},
  journal={Bioinformatics},
  year={2023},
  doi={XXXX/XXXX.XXXX}
}
```

