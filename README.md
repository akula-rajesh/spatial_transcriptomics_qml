# Spatial Transcriptomics QML Pipeline

A research pipeline for predicting spatial gene expression from H&E histology images, comparing **classical EfficientNet-based** models against **quantum machine learning (QML)** models. The project implements the BRSTNet paper methodology and extends it with PennyLane quantum circuits.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [GPU Support](#gpu-support)
- [Data Setup](#data-setup)
- [Configuration System](#configuration-system)
- [Running the Pipeline](#running-the-pipeline)
- [Scripts Reference](#scripts-reference)
- [Model Reference](#model-reference)
- [Experiment Results](#experiment-results)
- [Docker](#docker)
- [Adding New Components](#adding-new-components)
- [Troubleshooting](#troubleshooting)

---

## Overview

The pipeline predicts gene expression values at spatial spots on a tissue slide from the corresponding H&E image patch. It implements a **leave-one-patient-out** evaluation strategy:

- **23 patients total** — 22 train, 1 test (configurable via `data.test_patient`)
- **Gene prediction**: Top 250 genes by mean expression (configurable)
- **Auxiliary task**: Optionally predict remaining genes (`aux_ratio`)
- **Metrics**: aMAE, aRMSE, aCC (average per-gene Pearson correlation coefficient)

---

## Project Structure

```
spatial_transcriptomics_qml/
├── src/
│   ├── main.py                               # Entry point — CLI
│   ├── core/
│   │   ├── factory_registry.py               # Component registry (MODEL, TRAINER, DATA_PIPELINE)
│   │   └── pipeline_orchestrator.py          # Orchestrates all pipeline steps
│   ├── data_pipeline/
│   │   ├── base_pipeline.py                  # Abstract base for all pipeline components
│   │   ├── factory.py                        # Data pipeline factory registrations
│   │   ├── TEMPLATE_custom_data_loader.py    # Template for new loaders
│   │   ├── TEMPLATE_custom_data_processor.py # Template for new processors
│   │   └── brstnet_data_pipeline/
│   │       ├── spatial_downloader.py         # Downloads dataset from Mendeley
│   │       ├── spatial_file_organizer.py     # Organises raw files into subtype/patient dirs
│   │       ├── spatial_gene_processor.py     # Filters & packages gene counts as NPZ
│   │       ├── spatial_stain_normalizer.py   # Macenko stain normalisation
│   │       └── spatial_patch_extractor.py    # Crops & caches image patches
│   ├── models/
│   │   ├── factory.py                        # Model factory registrations
│   │   ├── base_model.py                     # Abstract base model
│   │   ├── classical/
│   │   │   ├── efficientnet_model.py         # EfficientNet-B4 + AuxNet (primary classical)
│   │   │   └── auxnet_model.py               # Standalone dual-head AuxNet
│   │   └── quantum/
│   │       ├── amplitude_embedding_qml.py    # Amplitude-embedding VQC
│   │       ├── efficientnet_quantum_head.py  # EfficientNet + re-uploading quantum head
│   │       ├── qnn_gene_predictor.py         # Full QNN v1 (PennyLane)
│   │       └── qnn_gene_predictor_v2.py      # Full QNN v2 (gradient-fixed, 3-phase)
│   ├── training/
│   │   ├── supervised_trainer.py             # Main trainer (fit/validate/evaluate loops)
│   │   ├── data_generator.py                 # SpatialDataset + DataLoader builders
│   │   ├── metrics.py                        # aCC, aMAE, aRMSE implementations
│   │   ├── cross_validator.py                # K-fold cross-validation
│   │   ├── callbacks.py                      # Training callbacks
│   │   └── factory.py                        # Trainer factory registrations
│   └── utils/
│       ├── config_manager.py                 # YAML config loader with dot-key access
│       ├── result_tracker.py                 # Experiment directory + JSON persistence
│       ├── device_utils.py                   # CUDA / MPS / CPU device detection
│       ├── directory_utils.py                # Auto-creates logs/, results/, data/ on startup
│       ├── logger.py                         # Logging setup helpers
│       └── visualization.py                  # Plot helpers
├── config/
│   ├── pipeline_config.yaml                           # Default (classical, resume-enabled)
│   ├── pipeline_config_classical_efficientnet.yaml
│   ├── pipeline_config_quantum_amplitude_embedding.yaml
│   ├── pipeline_config_efficientnet_quantum_head.yaml
│   ├── pipeline_config_qnn_gene_predictor.yaml
│   ├── pipeline_config_qnn_gene_predictor_v2.yaml
│   ├── pipeline_config_qnn_v2_phase0.yaml             # Phase 0: classical warmup
│   ├── pipeline_config_qnn_v2_phase1.yaml             # Phase 1: quantum training
│   ├── pipeline_config_qnn_v2_phase2.yaml             # Phase 2: joint fine-tune
│   └── model_configs/
│       ├── classical_efficientnet.yaml
│       ├── quantum_amplitude_embedding.yaml
│       ├── efficientnet_quantum_head.yaml
│       ├── qnn_gene_predictor.yaml
│       └── qnn_gene_predictor_v2.yaml
├── scripts/
│   ├── swap_test_patient.py       # Rearranges train/test split by patient ID
│   ├── visualize_training.py      # Generates HTML + PNG training reports
│   └── diagnose_pipeline.py       # Full pre-training diagnostic (12 checks)
├── data/
│   ├── train/counts/<subtype>/<patient>/*.npz
│   ├── train/images/<subtype>/<patient>/<window>/*.jpg
│   ├── test/counts/<subtype>/<patient>/*.npz
│   └── test/images/<subtype>/<patient>/<window>/*.jpg
├── results/                        # Auto-created per-experiment directories
├── logs/                           # Auto-created: logs/main.log
├── docs/
│   ├── PROJECT_UNDERSTANDING.md    # Deep technical reference
│   └── (this README is at root)
├── requirements.txt
├── setup.py
├── Dockerfile
└── docker-compose.yml
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.9 – 3.11 | 3.11 recommended |
| PyTorch | ≥ 2.0 | With torchvision |
| PennyLane | ≥ 0.36 | Quantum models only |
| OpenCV | ≥ 4.5 | Stain normalisation |
| Pillow | ≥ 9.0 | Image loading |
| NumPy | ≥ 1.21 | |
| scikit-learn | ≥ 1.0 | PCA for QNN v2 local loss |
| matplotlib | ≥ 3.4 | Visualisation scripts |
| tqdm | ≥ 4.62 | Progress bars |
| PyYAML | ≥ 6.0 | Config loading |
| requests | ≥ 2.25 | Data download |

**GPU (optional but strongly recommended):**
- NVIDIA CUDA ≥ 11.7
- Apple Silicon: macOS 12.3+ with PyTorch ≥ 2.0 for MPS

---

## Installation

### 1 — Clone

```bash
git clone https://github.com/your-org/spatial_transcriptomics_qml.git
cd spatial_transcriptomics_qml
```

### 2 — Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate         # macOS / Linux
# venv\Scripts\activate          # Windows
```

### 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4 — Install Package (editable mode)

```bash
pip install -e .
```

### 5 — Quantum Support (optional)

```bash
# Core PennyLane
pip install pennylane

# Lightning backend — enables backprop diff_method (strongly recommended for QNN v2)
pip install pennylane-lightning
```

> Without `pennylane-lightning`, QNN v2 uses a custom `torch.autograd.Function` with finite-difference input gradients. With it, full autograd backprop is used — faster and more accurate.

---

## GPU Support

The pipeline **auto-detects** the best available device at runtime. No configuration is needed unless you want to force a specific device.

| Priority | Device | Condition |
|---|---|---|
| 1 | CUDA | `torch.cuda.is_available()` |
| 2 | MPS | Apple Silicon + PyTorch ≥ 2.0 |
| 3 | CPU | Fallback |

### Enable / Disable via Config

```yaml
execution:
  gpu_enabled: true      # Master switch
  cuda_enabled: true     # Allow NVIDIA CUDA
  mps_enabled: true      # Allow Apple MPS
```

### MPS Notes (Apple Silicon)

- All tensors are kept as `float32` — MPS does **not** support `float64`.
- Quantum circuit outputs are cast to `float32` before moving to the MPS device.
- The pipeline handles this automatically.

### CUDA Notes

- `pin_memory=True` is set automatically when CUDA is available.
- Multi-GPU (`DataParallel`) is not enabled by default.

---

## Data Setup

### Option A — Automatic Download

Set `pipeline.download_data: true` in your config:

```yaml
pipeline:
  download_data: true

download:
  url: "https://data.mendeley.com/public-api/zip/29ntw7sh4r/download/5"
  remove_zip: true
  verify_ssl: false   # set false if behind a corporate proxy
```

Then run the full pipeline:

```bash
python src/main.py --config config/pipeline_config_classical_efficientnet.yaml --mode train
```

### Option B — Manual Setup

Place your data following this exact structure:

```
data/
├── train/
│   ├── counts/
│   │   ├── HER2_luminal/
│   │   │   ├── BC23287/
│   │   │   │   ├── C1_10_25.npz
│   │   │   │   └── ...
│   │   │   └── BC23901/...
│   │   ├── HER2_non_luminal/...
│   │   ├── Luminal_A/...
│   │   ├── Luminal_B/...
│   │   └── TNBC/...
│   └── images/
│       ├── HER2_luminal/
│       │   ├── BC23287/
│       │   │   └── 299/
│       │   │       ├── C1_10_25.jpg
│       │   │       └── ...
│       │   └── ...
│       └── ...
└── test/
    ├── counts/HER2_luminal/BC23450/*.npz
    └── images/HER2_luminal/BC23450/299/*.jpg
```

### NPZ Spot File Format

Each `.npz` file is one spatial spot:

| Key | Shape | Description |
|---|---|---|
| `count` | `(total_genes,)` | Raw gene counts |
| `pixel` | `(2,)` | Pixel coordinates `[x, y]` |
| `patient` | `(1,)` | Patient ID string |
| `section` | `(1,)` | Section ID string |
| `index` | `(2,)` | Spot grid coordinates `[row, col]` |

### Required Metadata Files

Place these in `data/processed/count_filtered/`:

| File | Description |
|---|---|
| `gene.pkl` | Python list of all gene ENSG names |
| `subtype.pkl` | Dict: `patient_id → cancer_subtype` |
| `mean_expression.npy` | Array `(total_genes,)` of per-gene mean expression |

---

## Configuration System

All configuration is YAML. The `ConfigManager` supports **dot-key access**: `config.get('training.epochs')`.

### Key Configuration Sections

```yaml
# ── Pipeline steps ────────────────────────────────────────────────────
pipeline:
  download_data: false       # Run SpatialDownloader
  process_data: false        # Run processing steps in order
  train_model: true          # Run training loop
  evaluate_model: true       # Run evaluation after training
  compare_results: false     # Compare multiple model results

  data_loader: "spatial_downloader"
  processing_steps:
    - name: "spatial_file_organizer"
      enabled: true
    - name: "spatial_gene_processor"
      enabled: true
    - name: "spatial_stain_normalizer"
      enabled: true
    - name: "spatial_patch_extractor"
      enabled: true

# ── Data paths ────────────────────────────────────────────────────────
data:
  test_patient: "BC23450"             # Patient held out for testing
  train_counts_dir: "data/train/counts/"
  train_images_dir: "data/train/images/"
  test_counts_dir:  "data/test/counts/"
  test_images_dir:  "data/test/images/"

# ── Training hyperparameters ─────────────────────────────────────────
training:
  epochs: 30
  lr: 0.001
  optimizer: "sgd"                    # "sgd" | "adam"
  momentum: 0.9
  weight_decay: 1e-6
  scheduler: "cosine"
  cosine_t_max: 5
  batch_size: 32
  num_workers: 8
  early_stopping_patience: 20
  pred_root: "results/predictions/"
  resume_path: null                   # Path to .pth checkpoint or null

# ── Model settings ────────────────────────────────────────────────────
model:
  gene_filter: 250                    # Top genes to predict
  aux_ratio: 1.0                      # 0 = no aux head; 1.0 = all remaining genes
  aux_weight: 1.0                     # Weight of aux loss term
  pretrained: true
  finetuning: "ftall"                 # "ftfc" | "ftconv" | "ftall"
  total_genes: null                   # Auto-resolved from dataset at runtime

# ── Model selector ────────────────────────────────────────────────────
models:
  active_model: "classical_efficientnet"
  save_best_only: true

# ── Preprocessing ─────────────────────────────────────────────────────
preprocessing:
  window_size: 299                    # Patch extraction window (paper default)

# ── Device ────────────────────────────────────────────────────────────
execution:
  gpu_enabled: true
  cuda_enabled: true
  mps_enabled: true
```

### Config Files per Model

| Config File | `active_model` value |
|---|---|
| `pipeline_config_classical_efficientnet.yaml` | `classical_efficientnet` |
| `pipeline_config_quantum_amplitude_embedding.yaml` | `quantum_amplitude_embedding` |
| `pipeline_config_efficientnet_quantum_head.yaml` | `efficientnet_quantum_head` |
| `pipeline_config_qnn_gene_predictor.yaml` | `qnn_gene_predictor` |
| `pipeline_config_qnn_gene_predictor_v2.yaml` | `qnn_gene_predictor_v2` |
| `pipeline_config_qnn_v2_phase0.yaml` | `qnn_gene_predictor_v2` (Phase 0) |
| `pipeline_config_qnn_v2_phase1.yaml` | `qnn_gene_predictor_v2` (Phase 1) |
| `pipeline_config_qnn_v2_phase2.yaml` | `qnn_gene_predictor_v2` (Phase 2) |

---

## Running the Pipeline

### Basic Training

```bash
python src/main.py --config config/pipeline_config_classical_efficientnet.yaml --mode train
```

### Resume Training from Checkpoint

```bash
# Via CLI (takes priority over config file setting)
python src/main.py \
  --config config/pipeline_config_classical_efficientnet.yaml \
  --mode train \
  --resume results/experiment_20260328_221235/20260328_221235/models/classical_efficientnet_final.pth
```

Or via `pipeline_config.yaml`:
```yaml
training:
  resume_path: "results/experiment_20260328_221235/20260328_221235/models/classical_efficientnet_final.pth"
```

### Train Each Model

```bash
# Classical EfficientNet-B4 (recommended starting point)
python src/main.py --config config/pipeline_config_classical_efficientnet.yaml --mode train

# Quantum Amplitude Embedding
python src/main.py --config config/pipeline_config_quantum_amplitude_embedding.yaml --mode train

# EfficientNet + Quantum Head
python src/main.py --config config/pipeline_config_efficientnet_quantum_head.yaml --mode train

# QNN Gene Predictor v1
python src/main.py --config config/pipeline_config_qnn_gene_predictor.yaml --mode train

# QNN Gene Predictor v2 — Phase 0 (classical warmup)
python src/main.py --config config/pipeline_config_qnn_v2_phase0.yaml --mode train

# QNN Gene Predictor v2 — Phase 1 (resume from Phase 0)
python src/main.py \
  --config config/pipeline_config_qnn_v2_phase1.yaml \
  --resume results/<phase0_run>/models/qnn_gene_predictor_v2_final.pth \
  --mode train

# QNN Gene Predictor v2 — Phase 2 (resume from Phase 1)
python src/main.py \
  --config config/pipeline_config_qnn_v2_phase2.yaml \
  --resume results/<phase1_run>/models/qnn_gene_predictor_v2_final.pth \
  --mode train
```

### CLI Arguments

| Argument | Short | Required | Description |
|---|---|---|---|
| `--config` | `-c` | **Yes** | Path to pipeline config YAML |
| `--mode` | `-m` | No | `train` (default) or `cross_validate` |
| `--resume` | `-r` | No | Path to `.pth` checkpoint to resume training from |
| `--verbose` | `-v` | No | Enable DEBUG-level logging |

---

## Scripts Reference

### `scripts/swap_test_patient.py`

Swaps the held-out test patient. Moves all counts + images for the new test patient from `data/train/` to `data/test/`, and the old test patient back to `data/train/`. Preserves the full `<subtype>/<patient>/<window>/` directory structure.

```bash
# Always preview first — no files are moved
python scripts/swap_test_patient.py --new-test BC23287 --dry-run

# Perform the swap (files moved, no config changed)
python scripts/swap_test_patient.py --new-test BC23287

# Swap + update one pipeline config
python scripts/swap_test_patient.py --new-test BC23287 \
    --config config/pipeline_config_classical_efficientnet.yaml

# Swap + update ALL pipeline_config*.yaml files
python scripts/swap_test_patient.py --new-test BC23287 --update-all-configs
```

| Argument | Description |
|---|---|
| `--new-test PATIENT_ID` | Patient to move to `data/test/` (must exist in `data/train/`) |
| `--config CONFIG_YAML` | Single config to update `data.test_patient` in |
| `--update-all-configs` | Update all `pipeline_config*.yaml` files |
| `--dry-run` | Preview only — no files touched |

**Known patient IDs:** `BC23209`, `BC23268`, `BC23269`, `BC23270`, `BC23272`, `BC23277`, `BC23287`, `BC23288`, `BC23377`, `BC23450`, `BC23506`, `BC23508`, `BC23567`, `BC23803`, `BC23810`, `BC23895`, `BC23901`, `BC23903`, `BC23944`, `BC24044`, `BC24105`, `BC24220`, `BC24223`

---

### `scripts/visualize_training.py`

Generates a combined HTML + PNG training report from one or more `results.json` files. Merges multiple consecutive runs (with resume) into a single continuous curve.

```bash
# Single run
python scripts/visualize_training.py \
    results/experiment_20260328_192449/20260328_192449/results.json

# Multiple runs merged (run1 → resume → run2 → ...)
python scripts/visualize_training.py \
    results/experiment_20260328_192449/20260328_192449/results.json \
    results/experiment_20260329_083530/20260329_083530/results.json

# Pass run directories (results.json located automatically)
python scripts/visualize_training.py \
    results/experiment_20260328_192449/20260328_192449 \
    results/experiment_20260329_083530/20260329_083530

# Custom output directory
python scripts/visualize_training.py run1/results.json run2/results.json \
    --output reports/classical_combined

# Show interactive matplotlib window
python scripts/visualize_training.py run1/results.json --show
```

**Outputs:**
- `training_report.html` — Full HTML report with embedded plots and metric tables
- `training_curves.png` — Train/val loss curves with best epoch marker
- `metrics_curves.png` — aMAE, aRMSE, aCC per epoch

---

### `scripts/diagnose_pipeline.py`

Runs 12 pre-training diagnostic checks. Use this before committing to a full multi-hour training run.

```bash
# Standard
python scripts/diagnose_pipeline.py --config config/pipeline_config.yaml

# Verbose output
python scripts/diagnose_pipeline.py --config config/pipeline_config.yaml --verbose

# Skip the 3-epoch training dry-run (faster)
python scripts/diagnose_pipeline.py --config config/pipeline_config.yaml --skip-training

# Attempt to auto-fix common issues
python scripts/diagnose_pipeline.py --config config/pipeline_config.yaml --fix
```

**Checks performed:**

| # | Check | What it validates |
|---|---|---|
| 1 | Environment | Python version, all required packages importable |
| 2 | Data directories | `data/train/`, `data/test/` populated |
| 3 | Metadata files | `gene.pkl`, `subtype.pkl`, `mean_expression.npy` |
| 4 | NPZ spot files | Required keys in a random sample of spots |
| 5 | Image loading | RGB, `uint8`, expected shape |
| 6 | Image normalisation | Detects grayscale channel collapse |
| 7 | Gene count processing | log1p transform, shape, value range |
| 8 | DataLoader batches | 8-tuple `(X, y, aux, coord, idx, patient, section, pixel)` |
| 9 | Model forward pass | Output shapes match `gene_filter` / `aux_nums` |
| 10 | Loss computation | MSELoss runs without error |
| 11 | Mini-training dry-run | 3 real training epochs complete successfully |
| 12 | Config validation | All required YAML keys present |

---

## Model Reference

### Classical EfficientNet (`classical_efficientnet`)

```
Image (B, 3, 224, 224)
  → EfficientNet-B4 backbone         (B, 1792)
  → main_head  Linear(1792 → 250)    (B, 250)    main gene predictions
  → aux_head   Linear(1792 → 5966)   (B, 5966)   auxiliary genes (optional)
```

- **Optimizer:** SGD, lr=1e-3, momentum=0.9, wd=1e-6
- **Scheduler:** CosineAnnealingLR (T_max=5)
- **Loss:** MSELoss (main + aux_weight × aux)
- **Fine-tuning:** `ftfc` | `ftconv` | `ftall`

---

### Quantum Amplitude Embedding (`quantum_amplitude_embedding`)

```
Image (B, 3, 224, 224)
  → EfficientNet-B4                  (B, 1792)
  → Linear(1792 → 2^n_qubits)       (B, 2^n_qubits)
  → AngleEmbedding + StronglyEntanglingLayers
  → PauliZ measurements              (B, n_qubits)
  → main_head  Linear(n_qubits → 250)
```

- Default: `n_qubits=3`, `n_layers=2`
- Fallback to classical pass-through if PennyLane not installed

---

### EfficientNet + Quantum Head (`efficientnet_quantum_head`)

```
Image (B, 3, 224, 224)
  → EfficientNet-B4                   (B, 1792)
  → ClassicalToQuantumBridge          (B, n_layers, n_qubits=16)
  → QuantumHead (data re-uploading)   (B, 16)
  → QuantumExpansionHead MLP          (B, 256)
  → main_head  Linear(256 → 250)
  → aux_head   Linear(256 → aux_nums)
```

- Expansion head solves the rank-16 bottleneck (`16 → 256 → 250`)
- `n_qubits=16`, `n_layers=4`

---

### QNN Gene Predictor v1 (`qnn_gene_predictor`)

```
Image (B, 3, 224, 224)
  → EfficientNet-B4      (B, 1792)
  → FeatureReducer MLP   (B, n_qubits)
  → QNNLayer (PennyLane) (B, n_qubits)   variational circuit
  → ClassicalDecoder MLP (B, decode_dim)
  → main_head  Linear(→ 250)
  → aux_head   Linear(→ aux_nums)
```

- **Optimizer:** Adam (better than SGD for small quantum gradients)
- Default: `n_qubits=8`, `n_layers=3`, `reduce_dim=32`, `decode_dim=512`

---

### QNN Gene Predictor v2 (`qnn_gene_predictor_v2`)

Addresses barren plateau and gradient vanishing with 7 targeted fixes:

| Fix | v1 problem | v2 solution |
|---|---|---|
| Gradient flow | `x.detach()` killed gradients | `torch.autograd.Function` + finite diff |
| Weight init | Uniform `[-π,π]` → barren plateau | Near-zero (`σ=0.01`) |
| Optimizer | SGD | Adam |
| Qubit count | `n_qubits=8` | `n_qubits=4` (16× better gradients) |
| Training phases | Single pass | Phase 0 → 1 → 2 with checkpoint handoff |
| Local quantum loss | None | PCA projection supervision on QNN output |
| Backend | `default.qubit` | `lightning.qubit` with backprop (if installed) |

**Three-phase training:**

| Phase | Config | What trains | Epochs |
|---|---|---|---|
| 0 — Classical warmup | `pipeline_config_qnn_v2_phase0.yaml` | FeatureReducer + Decoder + Heads; QNN frozen | 20 |
| 1 — Quantum training | `pipeline_config_qnn_v2_phase1.yaml` | QNN weights only; backbone frozen | 15 |
| 2 — Joint fine-tuning | `pipeline_config_qnn_v2_phase2.yaml` | All layers at small lr | 10 |

---

## Experiment Results

Each run creates a timestamped directory:

```
results/
└── experiment_20260328_192449/
    └── 20260328_192449/
        ├── results.json        ← live-updated training history (every epoch)
        ├── metrics.json        ← flattened metric log
        ├── metadata.json       ← experiment metadata + config snapshot
        ├── config.json         ← full pipeline config used for this run
        └── models/
            └── classical_efficientnet_final.pth   ← best checkpoint
```

### `results.json` Structure

```json
{
  "status": "completed",
  "current_epoch": 30,
  "best_epoch": 22,
  "best_val_loss": 1.3998,
  "train_losses": [2.14, 1.89, ...],
  "val_losses":   [2.01, 1.75, ...],
  "train_metrics": [
    {"loss": 2.14, "amae": 0.91, "armse": 1.31, "correlation_coefficient": 0.12},
    ...
  ],
  "val_metrics": [
    {"loss": 2.01, "amae": 0.87, "armse": 1.28, "correlation_coefficient": 0.14},
    ...
  ]
}
```

> `results.json` is **written after every epoch** — it is available even if training crashes.

### Prediction Files

Saved to `results/predictions/predictions_epoch_<N>.npz` when `evaluate_model: true`.

Contains arrays: `predictions`, `counts`, `patient`, `section`.

---

## Docker

### Build

```bash
docker build -t spatial-transcriptomics-qml .
# or
docker-compose build
```

### Run Training

```bash
# CPU
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/config:/app/config \
  spatial-transcriptomics-qml \
  python src/main.py --config config/pipeline_config_classical_efficientnet.yaml --mode train

# NVIDIA GPU
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  spatial-transcriptomics-qml \
  python src/main.py --config config/pipeline_config_classical_efficientnet.yaml --mode train
```

### Docker Compose

```bash
docker-compose run spatial-pipeline-cpu \
  python src/main.py --config config/pipeline_config.yaml --mode train

docker-compose run spatial-pipeline-gpu \
  python src/main.py --config config/pipeline_config.yaml --mode train
```

---

## Adding New Components

### New Model

1. Create `src/models/<classical|quantum>/my_model.py` subclassing `nn.Module`.
2. Implement `set_aux_head(aux_nums)` if you use an aux head.
3. Register in `src/models/factory.py`:

```python
from src.models.quantum.my_model import MyModel

def _create_my_model(config):
    return MyModel(config)

register_factory(ComponentType.MODEL, 'my_model', _create_my_model)
```

4. Create `config/model_configs/my_model.yaml`.
5. Create `config/pipeline_config_my_model.yaml` with `models.active_model: "my_model"`.

### New Data Pipeline Step

1. Create `src/data_pipeline/brstnet_data_pipeline/my_step.py` subclassing `BaseDataPipeline`.
2. Implement `execute(self) -> bool`.
3. The `__init__.py` auto-discovers and registers it via the `@register_data_pipeline` decorator.
4. Add to your config:

```yaml
pipeline:
  process_data: true
  processing_steps:
    - name: "my_step"
      enabled: true
```

---

## Troubleshooting

### `FileNotFoundError: logs/main.log`
Run from the project root. The `logs/` directory is auto-created by `directory_utils.py` on startup.

```bash
cd /path/to/spatial_transcriptomics_qml
python src/main.py ...
```

### `size mismatch for aux_head` on resume
The checkpoint was saved with `aux_nums=5966` (6216 total − 250 main = 5966) but the model rebuilt with a different default. The pipeline auto-fixes this in `_load_checkpoint()` by reading the checkpoint's `aux_head.weight` shape and calling `model.set_aux_head(ckpt_aux_nums)` before loading.

### MPS `Cannot convert to float64`
MPS does not support `float64`. All quantum model outputs are cast to `float32` before device transfer. Ensure you have the latest `amplitude_embedding_qml.py`.

### `Got unexpected field names: ['weights']`
Occurs when `efficientnet_pytorch` package conflicts with `torchvision`. The models now use `torchvision.models.efficientnet_b4()` as the primary backbone — remove `efficientnet_pytorch` or ensure the import fallback path is used.

### Very slow start (~30s) before first epoch
`compute_image_normalization()` loads all 4895 training patches to compute per-channel mean/std. This matches the paper's methodology and cannot be skipped without changing dataset behaviour.

### `element 0 of tensors does not require grad`
Seen in QNN v2 Phase 1 if quantum outputs are detached. Install `pennylane-lightning` for backprop diff_method, or ensure Phase 0 classical warmup has been completed before running Phase 1.

---

## Logs

```bash
tail -f logs/main.log                     # Follow live-updated log
grep "ERROR" logs/main.log                # Find errors
grep "aCC" logs/main.log                  # Find correlation values
grep "✓ New best" logs/main.log           # Find best epoch saves
grep "Checkpoint loaded" logs/main.log    # Confirm resume worked
```

---

*Last updated: April 2026*
