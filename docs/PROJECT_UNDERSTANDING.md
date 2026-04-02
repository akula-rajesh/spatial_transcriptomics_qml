# Project Understanding — Spatial Transcriptomics QML Pipeline

> Deep technical reference for all source code, design decisions, data formats, and training strategy.  
> Last updated: **April 2026**

---

## Table of Contents

1. [Research Goal](#1-research-goal)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Entry Point — `src/main.py`](#3-entry-point--srcmainpy)
4. [Core Layer](#4-core-layer)
   - [Factory Registry](#41-factory-registry--srccorefactory_registrypy)
   - [Pipeline Orchestrator](#42-pipeline-orchestrator--srccorepipeline_orchestratorpy)
5. [Data Pipeline Layer](#5-data-pipeline-layer)
   - [Base Pipeline](#51-base-pipeline)
   - [SpatialDownloader](#52-spatialdownloader)
   - [SpatialFileOrganizer](#53-spatialfileorganizer)
   - [SpatialGeneProcessor](#54-spatialgenprocessor)
   - [SpatialStainNormalizer](#55-spatialstainnormalizer)
   - [SpatialPatchExtractor](#56-spatialpatchextractor)
   - [Auto-Registration System](#57-auto-registration-system)
6. [Models Layer](#6-models-layer)
   - [Classical EfficientNet](#61-classical-efficientnet-classical_efficientnet)
   - [Quantum Amplitude Embedding](#62-quantum-amplitude-embedding-quantum_amplitude_embedding)
   - [EfficientNet + Quantum Head](#63-efficientnet--quantum-head-efficientnet_quantum_head)
   - [QNN Gene Predictor v1](#64-qnn-gene-predictor-v1-qnn_gene_predictor)
   - [QNN Gene Predictor v2](#65-qnn-gene-predictor-v2-qnn_gene_predictor_v2)
   - [Model Factory](#66-model-factory)
7. [Training Layer](#7-training-layer)
   - [SpatialDataset](#71-spatialdataset--srctrainingdata_generatorpy)
   - [SupervisedTrainer](#72-supervisedtrainer--srctrainingsupervised_trainerpy)
   - [Metrics](#73-metrics--srctrainingmetricspy)
   - [Callbacks](#74-callbacks)
   - [CrossValidator](#75-crossvalidator)
8. [Utils Layer](#8-utils-layer)
9. [Configuration System](#9-configuration-system)
10. [Experiment Tracking](#10-experiment-tracking)
11. [Data Flow — End to End](#11-data-flow--end-to-end)
12. [Key Design Decisions](#12-key-design-decisions)
13. [Known Issues and Fixes History](#13-known-issues-and-fixes-history)
14. [Quantum vs Classical — Technical Comparison](#14-quantum-vs-classical--technical-comparison)

---

## 1. Research Goal

Predict **spatial gene expression** at tissue spots from the corresponding H&E histology image patch. This is a regression problem: given a 224×224 pixel patch cropped at a spatial transcriptomics spot, predict a vector of `gene_filter=250` log-normalised gene expression values.

**Dataset:** Breast cancer spatial transcriptomics (23 patients, 5 cancer subtypes).  
**Evaluation strategy:** Leave-one-patient-out (22 train, 1 test).  
**Primary metrics:** aMAE, aRMSE, aCC (average per-gene Pearson correlation).

The project compares classical deep learning (EfficientNet-B4) against multiple quantum machine learning architectures to investigate whether quantum circuits provide any advantage.

---

## 2. High-Level Architecture

```
CLI (src/main.py)
  │
  ▼
PipelineOrchestrator                    ← controls execution flow
  ├── Step 1: _download_data()          ← SpatialDownloader
  ├── Step 2: _process_data()           ← sequential processing steps
  │     ├── SpatialFileOrganizer
  │     ├── SpatialGeneProcessor
  │     ├── SpatialStainNormalizer
  │     └── SpatialPatchExtractor
  ├── Step 3: _train_model()            ← creates model + trainer
  │     ├── FactoryRegistry → Model
  │     ├── FactoryRegistry → SupervisedTrainer
  │     │     ├── SpatialDataset (train + test)
  │     │     ├── fit() loop (epochs)
  │     │     ├── validate() loop
  │     │     └── early stopping
  │     └── result_tracker.save_model()
  └── Step 4: _evaluate_model()         ← reuses trained trainer
        └── evaluate() → predictions NPZ
```

All components are registered in a central **FactoryRegistry** and created on-demand by name string from the YAML config. This means adding a new model or data step requires no changes to the orchestrator.

---

## 3. Entry Point — `src/main.py`

**Purpose:** Parses CLI args, sets up logging, delegates to pipeline orchestrator.

```python
python src/main.py --config config/pipeline_config.yaml --mode train
python src/main.py --config config/pipeline_config.yaml --mode train --resume path/to/checkpoint.pth
python src/main.py --config config/pipeline_config.yaml --mode cross_validate
```

**Key behaviour:**
- Calls `ensure_project_structure()` before logging setup — guarantees `logs/`, `results/`, `data/` directories exist before `FileHandler` tries to open `logs/main.log`.
- Strips `config/` prefix from the config path before passing to `ConfigManager` (which always looks in the `config/` directory).
- `resume_path` from `--resume` CLI flag takes **priority** over `training.resume_path` in the YAML config.

---

## 4. Core Layer

### 4.1 Factory Registry — `src/core/factory_registry.py`

Central registry that maps string names to factory functions for three component types:

| ComponentType | Examples |
|---|---|
| `DATA_PIPELINE` | `spatial_downloader`, `spatial_gene_processor` |
| `MODEL` | `classical_efficientnet`, `qnn_gene_predictor_v2` |
| `TRAINER` | `supervised_trainer` |

```python
# Register a factory
registry.register_factory(ComponentType.MODEL, 'my_model', _create_my_model_fn)

# Create an instance
model = registry.create_instance(ComponentType.MODEL, 'my_model', config=cfg)
```

The helper `create_component(type, name, **kwargs)` wraps the registry for convenience.

All three package-level `__init__.py` files (`src.data_pipeline`, `src.models`, `src.training`) are imported at the top of `pipeline_orchestrator.py` purely for their side-effects: their `factory.py` modules call `register_factory()` at import time.

---

### 4.2 Pipeline Orchestrator — `src/core/pipeline_orchestrator.py`

**Purpose:** Translates the `pipeline:` section of the YAML config into sequential execution of named components.

**Initialization:**
1. Loads YAML config via `ConfigManager`
2. Resolves `resume_path` (CLI > config file > None)
3. Creates `ResultTracker` (creates experiment directory, writes `config.json`)
4. Stores experiment ID and directory path

**`run_pipeline()` steps:**

| Step | Config key | Action |
|---|---|---|
| 1 | `pipeline.download_data` | Calls `_download_data()` |
| 2 | `pipeline.process_data` | Calls `_process_data()` (each step by name) |
| 3 | `pipeline.train_model` | Calls `_train_model()` |
| 4 | `pipeline.evaluate_model` | Calls `_evaluate_model()` (reuses trainer) |
| 5 | `pipeline.compare_results` | Calls `_compare_results()` |

On any exception: saves partial results to `results.json`, marks status `failed`, re-raises.

**`_train_model()` important details:**
- Sets `total_genes = None` — model starts with `aux_nums=0`. The trainer calls `set_aux_head(dataset.aux_nums)` once the dataset is loaded.
- If `resume_path` is set, calls `_load_checkpoint(model, path)` before creating the trainer.
- Stores the trained trainer as `self._trained_trainer` so `_evaluate_model()` can reuse it without creating a new untrained model.

**`_load_checkpoint()` — aux_head size mismatch fix:**

```
Checkpoint saved with aux_nums=5966  (from real dataset: 6216 − 250 = 5966)
New model built with aux_nums=0      (total_genes=None → aux head not yet installed)

_load_checkpoint detects: state_dict has 'aux_head.weight' shape [5966, 1792]
→ calls model.set_aux_head(5966)     (installs correct-size Linear on device)
→ load_state_dict succeeds           (shapes now match)
```

This prevents the `RuntimeError: size mismatch for aux_head` that occurred on every resume before this fix.

---

## 5. Data Pipeline Layer

### 5.1 Base Pipeline

`src/data_pipeline/base_pipeline.py` — `BaseDataPipeline(ABC)` defines:
- `execute(self) -> bool` — abstract method each step must implement
- `get_config_value(key, default)` — dot-key config lookup
- `log_info()`, `log_warning()`, `log_error()` — standardised logging
- `_ensure_directory_exists(path)` — creates dir + returns Path

---

### 5.2 SpatialDownloader

**File:** `src/data_pipeline/brstnet_data_pipeline/spatial_downloader.py`  
**Registered as:** `spatial_downloader`

Downloads the dataset ZIP from Mendeley via HTTP streaming with a `tqdm` progress bar. Extracts to `data/input/`. Optionally removes the ZIP file after extraction.

**Config keys used:**
```yaml
download:
  url: "https://data.mendeley.com/..."
  remove_zip: true
  chunk_size: 1024
  max_retries: 3
  timeout: 300
data:
  input_dir: "data/input/"
  verify_ssl: false
```

---

### 5.3 SpatialFileOrganizer

**File:** `src/data_pipeline/brstnet_data_pipeline/spatial_file_organizer.py`  
**Registered as:** `spatial_file_organizer`

Scans `data/input/` and reorganises raw patient files into a structured hierarchy by cancer subtype:

```
data/stbc/<subtype>/<patient>/
  ├── HE/                  ← H&E whole-slide image
  ├── count_matrix/        ← gene count TSV files
  └── spot_files/          ← spot coordinate files
```

Builds and saves `subtype.pkl` mapping `patient_id → subtype`.

---

### 5.4 SpatialGeneProcessor

**File:** `src/data_pipeline/brstnet_data_pipeline/spatial_gene_processor.py`  
**Registered as:** `spatial_gene_processor`

The most complex processing step. Converts raw TSV gene count matrices into per-spot NPZ archives.

**Processing stages:**

1. **Global gene list** — collects all unique gene names across all patients
2. **Per-spot filters:**
   - Boundary check — patch must fit within image
   - Spot-ID check — spot must appear in count matrix
   - Quality filter — total reads ≥ `quality_threshold` (default 1000)
3. **Sparsity filter** — gene expressed in ≥ `sparsity_threshold` (10%) of spots
4. **Save** — per-spot NPZ files + `gene.pkl` + `mean_expression.npy`

**Output NPZ structure per spot:**
```python
{
  'count':   np.float32,  # shape (total_filtered_genes,)
  'pixel':   np.int32,    # shape (2,) — [x, y]
  'patient': np.str_,     # shape (1,)
  'section': np.str_,     # shape (1,)
  'index':   np.int32,    # shape (2,) — [row, col]
}
```

---

### 5.5 SpatialStainNormalizer

**File:** `src/data_pipeline/brstnet_data_pipeline/spatial_stain_normalizer.py`  
**Registered as:** `spatial_stain_normalizer`

Applies Macenko stain normalisation to all H&E whole-slide images. Reads from `data/stbc/`, writes normalised images to `data/stained/`. Preserves the same `<subtype>/<patient>/` directory structure.

Uses OpenCV for image I/O and numpy for the SVD-based Macenko algorithm.

---

### 5.6 SpatialPatchExtractor

**File:** `src/data_pipeline/brstnet_data_pipeline/spatial_patch_extractor.py`  
**Registered as:** `spatial_patch_extractor`

Crops fixed-size square patches centred on each spatial spot and saves them as JPEG files. Uses a PyTorch `DataLoader` with the internal `_PatchDataset` class to parallelise extraction across workers.

**Input:** Stain-normalised whole-slide images from `data/stained/`  
**Output:** `data/train/images/<subtype>/<patient>/<window>/<section>_<x>_<y>.jpg`

The `window` subdirectory name equals the `preprocessing.window_size` value (default: `299`).

---

### 5.7 Auto-Registration System

`src/data_pipeline/brstnet_data_pipeline/__init__.py` auto-discovers all `BaseDataPipeline` subclasses in the package and registers them in the factory. No manual factory registration is needed for new data pipeline steps — just create the file with the correct base class.

---

## 6. Models Layer

### 6.1 Classical EfficientNet (`classical_efficientnet`)

**File:** `src/models/classical/efficientnet_model.py`

**Architecture:**
```
Image (B, 3, 224, 224)
  → EfficientNet-B4 backbone (torchvision)   → (B, 1792)
  → main_head  nn.Linear(1792 → gene_filter) → (B, 250)
  → aux_head   nn.Linear(1792 → aux_nums)    → (B, aux_nums)   [optional]
```

The backbone's original classifier is replaced with `AuxNet` (dual-head) or a single `nn.Linear` (when `aux_ratio=0`).

**Key methods:**
- `set_aux_head(aux_nums)` — installs or replaces the aux head and moves it to device. Called by trainer once dataset gene count is known.
- `_apply_finetuning(strategy)`:
  - `ftfc` — freeze all, unfreeze FC layers (children ≥ 7)
  - `ftconv` — freeze all, unfreeze last 15+ blocks + FC
  - `ftall` — all parameters trainable

**Backbone loading strategy:**
1. Try `efficientnet_pytorch.EfficientNet.from_pretrained("efficientnet-b4")` without `weights` arg
2. Fall back to `torchvision.models.efficientnet_b4(weights=pretrained_weights)` if step 1 fails

This two-path strategy was added after `efficientnet_pytorch`'s `from_pretrained` started rejecting the `weights` kwarg passed by torchvision-aware code.

**Device placement:** The model calls `self.to(self._device)` at the end of `__init__`. The `_device` is determined by `device_utils.get_optimal_device()` which checks CUDA → MPS → CPU in priority order.

---

### 6.2 Quantum Amplitude Embedding (`quantum_amplitude_embedding`)

**File:** `src/models/quantum/amplitude_embedding_qml.py`

**Architecture:**
```
Image (B, 3, 224, 224)
  → EfficientNet-B4 backbone        → (B, 1792)
  → nn.Linear(1792 → 2^n_qubits)   → (B, 2^n_qubits)   dim reduction
  → Quantum Circuit (PennyLane)     → (B, n_qubits)      VQC outputs
  → main_head  Linear(n_qubits → 250)
  → aux_head   Linear(n_qubits → aux_nums)
```

**Quantum circuit:**
- `AngleEmbedding` — encodes `2^n_qubits` classical features as qubit rotations
- `StronglyEntanglingLayers(weights)` — variational ansatz
- `PauliZ` measurements on all qubits → `n_qubits` outputs in `[-1, 1]`
- Weight shape: `(n_layers, n_qubits, 3)` — the trailing `3` is fixed by PennyLane (Rx, Ry, Rz per qubit)

**MPS compatibility fix:** Quantum output is cast to `float32` via `.float()` before moving to device. Without this, MPS rejects `float64` tensors with `TypeError`.

**Fallback:** If PennyLane is not installed, the quantum layer is replaced with `nn.Linear(n_qubits → n_qubits)` (pass-through) so the rest of the pipeline still works.

---

### 6.3 EfficientNet + Quantum Head (`efficientnet_quantum_head`)

**File:** `src/models/quantum/efficientnet_quantum_head.py`

**Architecture:**
```
Image (B, 3, 224, 224)
  → EfficientNet-B4                       → (B, 1792)
  → ClassicalToQuantumBridge (MLP)        → (B, n_layers, n_qubits)
  → QuantumHead (data re-uploading VQC)   → (B, n_qubits=16)
  → QuantumExpansionHead (MLP)            → (B, expansion_dim=256)
  → main_head  Linear(256 → 250)
  → aux_head   Linear(256 → aux_nums)
```

**Key design — expansion head:** A quantum circuit with 16 qubits produces only 16 measurements. A single `Linear(16 → 250)` has rank ≤ 16 — mathematically underdetermined. The expansion head (`Linear(16 → 256) + GELU + Linear(256 → 250)`) breaks this bottleneck via the non-linearity between the two linear layers.

**Data re-uploading:** The `ClassicalToQuantumBridge` outputs fresh angle parameters for each circuit layer — the same classical features are re-encoded at each variational layer, giving universal approximation capability.

---

### 6.4 QNN Gene Predictor v1 (`qnn_gene_predictor`)

**File:** `src/models/quantum/qnn_gene_predictor.py`

**Architecture:**
```
Image (B, 3, 224, 224)
  → EfficientNet-B4              → (B, 1792)
  → FeatureReducer MLP           → (B, reduce_dim=32)
  → Linear(32 → n_qubits=8)     → (B, 8)
  → QNNLayer (PennyLane VQC)    → (B, 8)
  → ClassicalDecoder MLP         → (B, decode_dim=512)
  → main_head Linear(512 → 250)
  → aux_head  Linear(512 → aux_nums)
```

**Why the MLP wrapper around the QNN?**  
The quantum circuit alone outputs only `n_qubits=8` values. Predicting 250 genes from 8 values would be severely underdetermined. The `ClassicalDecoder` MLP (8 → 512 → 512) expands the quantum output before the gene prediction heads.

**Known limitation in v1:** `x.detach().cpu().float()` was used before the quantum circuit to prevent PennyLane crashes. This **detached the input from the autograd graph**, meaning the `FeatureReducer` and `EfficientNet` backbone received **no gradient signal**. This was identified as the primary reason v1 failed to learn. Fixed in v2.

---

### 6.5 QNN Gene Predictor v2 (`qnn_gene_predictor_v2`)

**File:** `src/models/quantum/qnn_gene_predictor_v2.py`

v2 targets 7 specific failure modes identified in v1:

**Fix 1 — Gradient flow via `torch.autograd.Function`**

```python
class QuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, circuit_fn, n_qubits):
        ctx.save_for_backward(inputs, weights)
        ctx.circuit_fn = circuit_fn
        ctx.n_qubits = n_qubits
        return circuit_fn(inputs, weights)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        # Input gradient via finite differences
        dx = finite_diff_input_grad(ctx.circuit_fn, inputs, weights, grad_output)
        # Weight gradient via parameter-shift rule
        dw = parameter_shift_grad(ctx.circuit_fn, inputs, weights, grad_output)
        return dx, dw, None, None
```

This allows `FeatureReducer` to receive gradients from the quantum circuit.

**Fix 2 — Anti-barren-plateau weight initialization**

```python
# v1: uniform (-π, π) → barren plateau at n_qubits=8
# v2: near-zero noise → near-identity circuit at initialization
self.weights = nn.Parameter(torch.zeros(n_layers, n_qubits, 3) + 0.01 * torch.randn(...))
```

**Fix 3 — Adam optimizer** — better than SGD for tiny quantum gradients.

**Fix 4 — Fewer qubits (`n_qubits=4`)** — gradient magnitude scales as `1/2^n`. At n=8, gradients are ~1/256 of their n=4 value.

**Fix 5 — Three-phase training:**

| Phase | `training.phase` | Frozen | Trainable |
|---|---|---|---|
| 0 | `0` | QNN weights | FeatureReducer + ClassicalDecoder + Heads |
| 1 | `1` | Backbone + FeatureReducer + Decoder + Heads | QNN weights only |
| 2 | `2` | Nothing | All layers at lr=1e-5 |

**Fix 6 — Local quantum loss:**

```python
# PCA projects 250-dim gene target down to n_qubits dimensions
# This adds supervision directly on the QNN output
pca_target = self.pca_projector.transform(y_batch)  # (B, 4)
quantum_loss = mse_loss(q_output, pca_target)
total_loss = main_loss + lambda_q * quantum_loss
```

**Fix 7 — `lightning.qubit` backend:**

```python
if lightning_available:
    dev = qml.device("lightning.qubit", wires=n_qubits)
    @qml.qnode(dev, diff_method="backprop")  # full autograd graph
    def circuit(inputs, weights): ...
```

When `lightning.qubit` is available, the custom `QuantumFunction` is not needed — full autograd backprop works natively.

---

### 6.6 Model Factory

**File:** `src/models/factory.py`

Registers all models at import time:

```python
register_factory(ComponentType.MODEL, 'classical_efficientnet',        _create_efficientnet_model)
register_factory(ComponentType.MODEL, 'quantum_amplitude_embedding',   _create_quantum_amplitude_embedding_model)
register_factory(ComponentType.MODEL, 'efficientnet_quantum_head',     _create_efficientnet_quantum_head_model)
register_factory(ComponentType.MODEL, 'qnn_gene_predictor',            _create_qnn_gene_predictor_model)
register_factory(ComponentType.MODEL, 'qnn_gene_predictor_v2',         _create_qnn_gene_predictor_v2_model)
```

Each factory function reads the model-specific YAML from `config/model_configs/`, merges pipeline-level `model:` settings, then instantiates the class.

---

## 7. Training Layer

### 7.1 SpatialDataset — `src/training/data_generator.py`

`SpatialDataset(Dataset)` implements the reference paper's data loading exactly.

**Initialisation steps:**
1. Loads all `.npz` file paths from `count_root` (recursive glob)
2. Filters to `patient_list` if provided
3. Loads metadata: `gene.pkl`, `subtype.pkl`, `mean_expression.npy`
4. Selects top `gene_filter` genes by mean expression (`np.argsort(mean_expression)[::-1][:gene_filter]`)
5. Computes auxiliary gene indices (next `aux_ratio × remaining` genes by expression)

**`__getitem__` — returns 8-tuple:**

```python
# With aux_ratio > 0:
return X, y, aux, coord_tensor, index_tensor, patient, section, pixel_tensor

# With aux_ratio == 0:
return X, y, coord_tensor, index_tensor, patient, section, pixel_tensor
```

Where:
- `X`: transformed image patch `(3, 224, 224)` float32
- `y`: log1p(main gene counts), optionally normalised by training stats
- `aux`: log1p(aux gene counts), **not normalised** (matches paper)
- `coord`: spot grid `[row, col]`
- `pixel`: pixel coordinates `[x, y]`

**Image path resolution:**
```python
img_path = img_root / subtype[patient] / patient / str(window) / f"{section}_{coord[0]}_{coord[1]}.jpg"
```

**Normalisation strategy (matches paper):**
- `compute_image_normalization()` — computes per-channel mean/std from ALL training images
- `compute_dataset_normalization()` — computes per-gene log1p mean/std from ALL training spots
- Both are computed from training data only; the same stats are applied to test data

**`create_dataloaders(paths, config, test_patient)` — full setup:**
1. Build initial dataset with `T.ToTensor()` only (no normalisation)
2. Compute image mean/std from that dataset
3. Compute gene mean/std
4. Rebuild train dataset with augmentation + normalisation
5. Build test dataset with val transforms + **train normalisation stats**

---

### 7.2 SupervisedTrainer — `src/training/supervised_trainer.py`

The main training engine. Implements the paper's training strategy exactly.

**Optimizer:** `SGD(lr=1e-3, momentum=0.9, weight_decay=1e-6)` — or Adam if `optimizer: "adam"` in config.

**Scheduler:** `CosineAnnealingLR(T_max=5)` — matches paper default.

**Loss:** `MSELoss()` — on log1p-transformed gene counts.

**`train()` high-level flow:**
```
1. _prepare_data_loaders()     ← calls create_dataloaders, sets aux_head size
2. For each epoch:
   a. fit() train loop
   b. scheduler.step()
   c. validate() val loop
   d. track best model (by val loss)
   e. EarlyStopping check
   f. Save results.json
3. Return training history dict
```

**`_prepare_data_loaders()` — crucial step:**
- Calls `create_dataloaders()` to build `SpatialDataset` loaders
- Reads `dataset.aux_nums` from the loaded dataset (e.g. 5966)
- Calls `model.set_aux_head(dataset.aux_nums)` to install correct-size aux head
- This is where `total_genes` is finally resolved from real data

**`fit()` — one training epoch:**
```python
for batch in train_loader:
    X, y, aux, *_ = _unpack_batch(batch, aux_ratio, device)
    pred = model(X)           # (main_pred, aux_pred) or just main_pred
    main_loss = criterion(main_pred, y)
    aux_loss  = criterion(aux_pred, aux)    # if aux_ratio > 0
    loss = main_loss + aux_weight * aux_loss
    loss.backward()
    optimizer.step()
```

**`_unpack_batch(batch, aux_ratio, device)` — handles both 8-tuple and 7-tuple:**
```python
if aux_ratio > 0:
    X, y, aux, coord, idx, patient, section, pixel = batch
else:
    X, y, coord, idx, patient, section, pixel = batch
    aux = None
```

**Best model tracking:** `best_state` stores `{k: v.cpu().clone()}` of the full state dict whenever val loss improves. On `evaluate()`, best weights are restored before running the test pass.

**`evaluate()` — test set evaluation:**
1. Calls `_prepare_data_loaders()` with test patient to build test DataLoader
2. Restores best weights
3. Runs `evaluate()` function — collects all predictions, computes metrics
4. If `save_path` is set, saves predictions as `.npz`

**Prediction save path fix:** When `pred_root = "results/predictions/"` (a directory path), the filename is constructed as:
```python
save_path_obj.mkdir(parents=True, exist_ok=True)
npz_path = str(save_path_obj / f"predictions_epoch_{epoch}")
```
This prevents the bug where `results/predictions/_epoch_29.npz` was produced (empty filename stem).

---

### 7.3 Metrics — `src/training/metrics.py`

**`average_correlation_coefficient(y_pred, y_true)` — aCC:**

Per-gene Pearson r averaged across all genes:

```python
# For each gene j (axis=0 = across spots):
top_j    = sum_i((y_true[:,j] - mean_true_j) * (y_pred[:,j] - mean_pred_j))
bottom_j = sqrt(sum_i(y_true[:,j] - mean_true_j)^2 * sum_i(y_pred[:,j] - mean_pred_j)^2) + eps
r_j      = top_j / bottom_j

aCC = mean(r_j for j in range(d))
```

This differs from a **global** Pearson r (flattening all predictions into one vector). The per-gene formulation is correct per the paper — it measures whether the model learns the spatial expression pattern of each individual gene.

**`average_mae(y_pred, y_true)` — aMAE:** `mean(|y_pred - y_true|)` across all spots and genes.

**`average_rmse(y_pred, y_true)` — aRMSE:** `sqrt(mean((y_pred - y_true)^2))` across all spots and genes.

**`compute_all_metrics(y_pred, y_true, prefix="")` — convenience function:**
```python
return {
    f"{prefix}amae":                    average_mae(y_pred, y_true),
    f"{prefix}armse":                   average_rmse(y_pred, y_true),
    f"{prefix}correlation_coefficient": average_correlation_coefficient(y_pred, y_true),
}
```

Both numpy arrays and torch tensors are accepted (dispatched internally).

---

### 7.4 Callbacks

**File:** `src/training/callbacks.py`

`EarlyStopping`: Stops training when validation loss does not improve by `min_delta` for `patience` epochs.

`LRScheduler`: Wraps `ReduceLROnPlateau` — reduces LR by `factor` when val loss plateaus for `patience` epochs. (Separate from the CosineAnnealingLR used in training; this is available as an alternative.)

---

### 7.5 CrossValidator

**File:** `src/training/cross_validator.py`

Implements k-fold cross-validation using patient-level splits. Currently used with synthetic data for quick architecture testing; real data integration is planned.

---

## 8. Utils Layer

### `config_manager.py`
`ConfigManager` loads YAML files from the `config/` directory. Supports dot-key access for nested keys:

```python
manager.get('training.epochs')         # → 30
manager.get('model.gene_filter', 250)  # → 250 (with default)
```

`load_model_config(model_name)` loads `config/model_configs/<model_name>.yaml`.

### `result_tracker.py`
Creates and manages the experiment directory structure:

```
results/experiment_<timestamp>/
  <timestamp>/
    ├── results.json    ← updated every epoch
    ├── metrics.json    ← flattened metric log
    ├── metadata.json   ← experiment info
    ├── config.json     ← config snapshot
    └── models/         ← saved .pth checkpoints
```

Key methods:
- `log_metric(name, value, step=None)` — adds to metrics.json
- `log_config(config)` — saves config.json
- `save_model(state_dict, filename)` — pickles model state to `models/`
- `save_results(pipeline_results)` — writes final results.json
- `update_status('completed'|'failed')` — updates metadata.json

### `device_utils.py`
`get_optimal_device(config)` returns the best available `torch.device`:

```python
if config.get('execution.cuda_enabled') and torch.cuda.is_available():
    return torch.device('cuda')
elif config.get('execution.mps_enabled') and torch.backends.mps.is_available():
    return torch.device('mps')
else:
    return torch.device('cpu')
```

Also provides `get_available_devices()` for a full inventory of all detected devices.

### `directory_utils.py`
`ensure_project_structure(project_root)` — creates all required directories before logging is initialised:

```python
dirs = ['logs', 'results', 'data', 'data/train', 'data/test',
        'data/processed', 'data/input', 'data/stained', 'data/stbc']
for d in dirs:
    (project_root / d).mkdir(parents=True, exist_ok=True)
```

This is why `logs/main.log` is always available by the time `FileHandler` is created.

---

## 9. Configuration System

### Hierarchy

```
pipeline_config_*.yaml
  ├── pipeline.*          (which steps to run)
  ├── data.*              (paths, test_patient)
  ├── training.*          (hyperparameters, optimizer, scheduler)
  ├── model.*             (gene_filter, aux_ratio, finetuning)
  ├── models.*            (active_model name)
  ├── preprocessing.*     (window_size)
  ├── execution.*         (gpu_enabled, mps_enabled)
  └── results.*           (base_dir)

config/model_configs/<active_model>.yaml
  └── (model-specific overrides merged into model_config dict)
```

### Merge Strategy in `_train_model()`

```python
model_config = config_manager.load_model_config(active_model)   # model-specific YAML
pipeline_model = self.config.get('model', {})                    # pipeline-level model section
model_config.update(pipeline_model)                              # pipeline values WIN
```

This means you can override `gene_filter` or `finetuning` from the pipeline config without editing the model config.

---

## 10. Experiment Tracking

### Live `results.json` Update

`results.json` is written after **every epoch**, not just at the end. This means:
- If training crashes, the last completed epoch is still visible
- `visualize_training.py` can be run on an in-progress training run
- Multiple runs can be merged by the visualiser for continuous curves across resumes

### `results.json` training state fields

| Field | Type | Description |
|---|---|---|
| `status` | str | `running` → `completed` or `failed` |
| `current_epoch` | int | Last completed epoch (0-based) |
| `best_epoch` | int | Epoch with lowest val loss |
| `best_val_loss` | float | Lowest validation loss seen |
| `train_losses` | list[float] | Train MSE loss per epoch |
| `val_losses` | list[float] | Val MSE loss per epoch |
| `train_metrics` | list[dict] | aMAE, aRMSE, aCC per epoch |
| `val_metrics` | list[dict] | aMAE, aRMSE, aCC per epoch |

---

## 11. Data Flow — End to End

```
Raw dataset (Mendeley ZIP)
  │  SpatialDownloader
  ▼
data/input/<patient_dirs>/
  │  SpatialFileOrganizer
  ▼
data/stbc/<subtype>/<patient>/
  ├── HE/         (whole-slide H&E images)
  ├── count_matrix/
  └── spot_files/
  │  SpatialGeneProcessor
  ▼
data/processed/count_raw/<subtype>/<patient>/<spot>.npz   (all genes)
data/processed/count_filtered/
  ├── gene.pkl                  (global gene list)
  ├── subtype.pkl               (patient → subtype mapping)
  └── mean_expression.npy       (per-gene mean across all spots)
  │  SpatialStainNormalizer
  ▼
data/stained/<subtype>/<patient>/  (normalised whole-slide images)
  │  SpatialPatchExtractor
  ▼
data/train/counts/<subtype>/<patient>/<spot>.npz   (filtered genes only)
data/train/images/<subtype>/<patient>/299/<spot>.jpg
data/test/counts/<subtype>/<patient>/<spot>.npz
data/test/images/<subtype>/<patient>/299/<spot>.jpg
  │  SpatialDataset
  ▼
DataLoader 8-tuple batches: (X, y, aux, coord, idx, patient, section, pixel)
  X:   (B, 3, 224, 224)   images resized from 299→224, augmented, normalised
  y:   (B, 250)           log1p main gene counts, normalised by training stats
  aux: (B, 5966)          log1p aux gene counts, NOT normalised
  │  SupervisedTrainer
  ▼
Model forward pass → (main_pred, aux_pred)
  │  MSELoss
  ▼
loss = main_loss + aux_weight × aux_loss
  │  SGD / Adam + CosineAnnealingLR
  ▼
results/experiment_<id>/
  ├── results.json        ← written every epoch
  ├── models/*.pth        ← best checkpoint saved
  └── predictions/*.npz   ← test set predictions on evaluate
```

---

## 12. Key Design Decisions

### Leave-One-Patient-Out Evaluation
Spatial transcriptomics datasets are small (23 patients). Using a held-out patient rather than random spot-level splits avoids data leakage and measures true generalisation to unseen patients.

### log1p Gene Count Transformation
Gene count data is highly skewed (many zeros, few very high values). `log1p(count)` compresses the dynamic range and makes the MSE loss more meaningful across all expression levels.

### Dataset-Specific Image Normalisation
ImageNet statistics (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`) are not appropriate for H&E images which have very different colour distributions. The per-channel mean/std is computed from the training set H&E patches directly. This matches the paper's approach.

### Auxiliary Task (aux_ratio)
Predicting all remaining genes (beyond the top 250) as an auxiliary task regularises the backbone features. The aux loss uses the same MSE criterion but aux gene counts are **not normalised** — matching the paper's behaviour. Set `aux_ratio: 0` to disable.

### `total_genes = None` Convention
Models are built with `total_genes=None`, producing `aux_nums=0` (no aux head). The trainer installs the correct-size aux head via `model.set_aux_head(dataset.aux_nums)` once the dataset is loaded. This avoids all checkpoint size-mismatch errors.

### Resume Path Priority: CLI > Config File
The CLI `--resume` flag always overrides `training.resume_path` in the YAML. This allows using the same config file for multiple resume experiments without editing the file each time.

---

## 13. Known Issues and Fixes History

| Date | Error | Root Cause | Fix Applied |
|---|---|---|---|
| Mar 2026 | `IndentationError` in `setup.py` | Bad indentation in `setup.py` | Fixed indentation |
| Mar 2026 | `FileNotFoundError: logs/main.log` | `logs/` directory not created before `FileHandler` | Added `ensure_project_structure()` before logging setup |
| Mar 2026 | Config path `config/config/pipeline_config.yaml` | Double `config/` prefix | Strip `config/` prefix in `main.py` before passing to `ConfigManager` |
| Mar 2026 | `ResultTracker has no attribute create_experiment_directory` | Method renamed | Updated orchestrator to use current `ResultTracker` API |
| Mar 2026 | Download SSL verification failed | Corporate proxy | Added `verify_ssl: false` config option |
| Mar 2026 | Download showing raw percentage logs | `tqdm` not used | Replaced raw progress logs with `tqdm` progress bar |
| Mar 2026 | Training not using pipeline config epochs | `main.py` not using orchestrator | Rewired `main.py` to always use `PipelineOrchestrator` |
| Mar 2026 | `not enough values to unpack (expected 8, got 2)` | Synthetic DataLoader yielded `(X, y)` tuples | Fixed synthetic data generator to yield full 8-tuple |
| Mar 2026 | `aux_pred` is `None` when model returns tuple | `model(X)` returns `(main, aux)` but code checked `aux is not None` | Fixed `_unpack_batch` to handle both with/without aux correctly |
| Mar 2026 | `weight is on cpu but expected on mps` | `set_aux_head()` created Linear on CPU | Added `.to(self._device)` in `set_aux_head()` |
| Mar 2026 | Tensor size mismatch: `(1000)` vs `(250)` | torchvision EfficientNet default FC = 1000 classes | Replaced backbone FC with `AuxNet` during backbone build |
| Mar 2026 | `Got unexpected field names: ['weights']` | `efficientnet_pytorch.from_pretrained` rejected `weights` kwarg | Added two-path backbone loading; torchvision fallback |
| Mar 2026 | QML `Torch not compiled with CUDA enabled` | Quantum layer tried to use CUDA | Fixed device detection to use MPS when CUDA unavailable |
| Mar 2026 | `Cannot convert MPS Tensor to float64` | Quantum output was `float64` | Added `.float()` cast before `.to(device)` |
| Mar 2026 | `size mismatch for aux_head` on resume | Checkpoint `aux_nums=5966`, model rebuilt with `6000` | `_load_checkpoint` now reads checkpoint shape, calls `set_aux_head()` first |
| Mar 2026 | Predictions saved as `_epoch_29.npz` | `save_path` was a directory path with no filename stem | Fixed to construct filename explicitly inside the directory |
| Mar 2026 | `No such file or directory: results/predictions/` | `Path(save_path).parent.mkdir()` only created `results/` not `results/predictions/` | Fixed to call `.mkdir()` on the save_path directory itself |
| Apr 2026 | `element 0 of tensors does not require grad` | QNN v2 Phase 1: quantum output detached from graph | Fixed with `QuantumFunction` custom autograd; `lightning.qubit` backprop |

---

## 14. Quantum vs Classical — Technical Comparison

### Architecture Summary

| Model | Parameters | Quantum Circuit | Training Time |
|---|---|---|---|
| Classical EfficientNet | ~19M | None | ~35s/epoch |
| Quantum Amplitude Embedding | ~19M + VQC | 3 qubits, 2 layers | ~45s/epoch |
| EfficientNet + Quantum Head | ~19M + VQC | 16 qubits, 4 layers | ~90s/epoch |
| QNN Gene Predictor v1 | ~19M + VQC | 8 qubits, 3 layers | ~120s/epoch |
| QNN Gene Predictor v2 | ~19M + VQC | 4 qubits, 3 layers | ~90s/epoch |

### Why Classical Outperforms Quantum (Currently)

1. **Barren plateau** — gradient magnitude vanishes exponentially with n_qubits. At n=8, gradients are ~1/256 of classical values. QNN v2 mitigates this with n=4 and near-zero init.

2. **Simulation overhead** — PennyLane simulates quantum circuits on classical hardware. The simulator must track a `2^n_qubits`-dimensional state vector. Even at n=4, this adds significant computational overhead.

3. **Expressibility bottleneck** — a quantum circuit with n_qubits outputs exactly n measurements, regardless of circuit depth. The `ClassicalDecoder` is needed to expand these into 250 gene predictions, adding classical computation back into the pipeline.

4. **Gradient method** — PennyLane's default `parameter-shift` gradient requires 2 circuit evaluations per parameter. With n_layers=3, n_qubits=4, this is 2×3×4×3=72 circuit evaluations per backward pass vs 1 for classical backprop.

### Potential Advantages of Quantum (Theoretical)

- **Hilbert space dimension** — `2^n_qubits` dimensional Hilbert space may provide inductive bias for learning certain correlation patterns in gene expression data.
- **Entanglement** — `StronglyEntanglingLayers` creates full entanglement between all qubits; functions that require exponentially large classical tensors to represent may be compact in the quantum representation.
- **Quantum interference** — constructive/destructive interference between circuit paths allows the circuit to compute functions not efficiently computable by equivalent-depth classical networks.

### Current Experimental Status

Based on training runs completed in March 2026:
- Classical EfficientNet achieves stable convergence (val loss ~1.40 at epoch 30)
- QNN v2 Phase 0 (classical warmup) achieves similar loss (~1.42)
- QNN v2 Phase 1 (quantum training) shows slower convergence but gradients are non-zero
- QNN v2 Phase 2 (joint fine-tune) was not completed due to time constraints

The research hypothesis — that quantum circuits can learn spatial gene expression patterns more efficiently — remains open. Further experiments with larger qubit counts on real quantum hardware (not simulators) would be needed to draw definitive conclusions.

---

*Document last updated: April 2026*
