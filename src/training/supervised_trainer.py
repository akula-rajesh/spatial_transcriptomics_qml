"""
Trainer for spatial transcriptomics gene expression prediction.

Matches the reference paper's training strategy:
  - SGD optimizer (lr=1e-3, momentum=0.9, weight_decay=1e-6)
  - CosineAnnealingLR scheduler (T_max=5)
  - MSELoss criterion
  - Per-gene aCC / aMAE / aRMSE metrics (not global scalar)
  - Auxiliary task support with weighted loss
  - 8-tuple DataLoader unpacking: (X, y, aux, coord, index, patient, section, pixel)
  - Early stopping
  - Best-checkpoint tracking & restore
"""

import time
import logging
import math
import pathlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple

from src.training.metrics import (
    average_correlation_coefficient,
    average_mae,
    average_rmse,
    compute_all_metrics,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Early Stopping  (matches paper's EarlyStopping class)
# ======================================================================

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                logger.info("Early stopping triggered")
                self.early_stop = True


# ======================================================================
# Batch unpacking helper
# ======================================================================

def _unpack_pred(pred, aux_ratio: float):
    """
    Safely unpack model forward() output into (main_pred, aux_pred_or_None).

    Handles:
      - tuple (main, aux)  where aux may be None  ← torchvision fallback
      - tuple (main, aux)  where aux is a Tensor  ← AuxNet path
      - plain Tensor                               ← no-aux path
    """
    if isinstance(pred, (tuple, list)):
        main_pred = pred[0]
        aux_pred  = pred[1] if len(pred) > 1 else None
    else:
        main_pred = pred
        aux_pred  = None

    # If aux_ratio > 0 but the model didn't produce an aux output, disable aux for this batch
    if aux_ratio > 0 and (aux_pred is None or not isinstance(aux_pred, torch.Tensor)):
        aux_pred = None   # caller must handle None aux_pred gracefully

    return main_pred, aux_pred


def _unpack_batch(batch: tuple, aux_ratio: float, device: torch.device) -> Tuple:
    """
    Unpack a batch from any DataLoader into a normalised 8-element tuple.

    Supported formats (by length):
      8 — SpatialDataset + aux:    (X, y, aux, coord, index, patient, section, pixel)
      7 — SpatialDataset no aux:   (X, y, coord, index, patient, section, pixel)
      2 — Synthetic TensorDataset: (X, y)   ← fallback when real data is missing

    Always returns:
        (X, y, aux_or_None, coord, idx, patient, section, pixel)
    """
    n = len(batch)

    if n == 8:
        # Full SpatialDataset with auxiliary genes
        X, y, aux, coord, idx, patient, section, pixel = batch
        return X.to(device), y.to(device), aux.to(device), coord, idx, patient, section, pixel

    elif n == 7:
        # SpatialDataset without auxiliary genes
        X, y, coord, idx, patient, section, pixel = batch
        return X.to(device), y.to(device), None, coord, idx, patient, section, pixel

    elif n == 2:
        # Synthetic / TensorDataset fallback — only (images, counts)
        X, y = batch
        B = X.size(0)
        dummy_coord   = torch.zeros(B, 2, dtype=torch.long)
        dummy_idx     = torch.zeros(B, 1, dtype=torch.long)
        dummy_patient = ["synthetic"] * B
        dummy_section = ["synthetic"] * B
        dummy_pixel   = torch.zeros(B, 2, dtype=torch.long)

        if aux_ratio > 0:
            # aux target shape must match model's aux_fc output — use same as y for synthetic
            # (criterion only needs matching shapes; real aux size resolved at runtime)
            aux = torch.zeros_like(y)
            return X.to(device), y.to(device), aux.to(device), \
                   dummy_coord, dummy_idx, dummy_patient, dummy_section, dummy_pixel
        else:
            return X.to(device), y.to(device), None, \
                   dummy_coord, dummy_idx, dummy_patient, dummy_section, dummy_pixel

    else:
        raise ValueError(
            f"Unexpected batch length {n}. "
            "Expected 2 (synthetic), 7 (SpatialDataset no-aux), or 8 (SpatialDataset with-aux)."
        )


# ======================================================================
# Local quantum loss helper  (V2 support)
# ======================================================================

def _compute_local_quantum_loss(
    model: nn.Module,
    y: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Compute the local quantum loss for QNNGenePredictorV2.

    Called after forward() — the model stores its QNN output in _last_q_out.
    Returns None for all non-V2 models (zero overhead).

    Args:
        model : The model (any type — safely checks for V2 API)
        y     : (B, gene_filter) gene targets on device

    Returns:
        Scalar loss tensor, or None if model does not support it.
    """
    try:
        if hasattr(model, "compute_local_quantum_loss") and \
           hasattr(model, "_last_q_out") and \
           model._last_q_out is not None and \
           getattr(model, "use_local_loss", False):
            return model.compute_local_quantum_loss(model._last_q_out, y)
    except Exception:
        pass
    return None


# ======================================================================
# One-epoch functions  (fit / validate / evaluate)
# ======================================================================

def fit(
    model:        nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer:    torch.optim.Optimizer,
    criterion:    nn.Module,
    aux_ratio:    float,
    aux_weight:   float,
    device:       torch.device,
) -> Dict[str, float]:
    """
    One epoch of training.

    Matches the paper's fit() exactly:
      - Auxiliary task: loss = main_loss + aux_weight * aux_loss
      - All metrics computed over full epoch (not batch averages)
      - aCC = per-gene Pearson r averaged across genes
    """
    print('-' * 10)
    print('Training:')
    model.train()

    total_loss = total_main_loss = total_aux_loss = 0.0
    epoch_preds  = []
    epoch_counts = []
    aux_preds    = []
    aux_counts   = []

    for batch in train_loader:
        X, y, aux, *_ = _unpack_batch(batch, aux_ratio, device)

        optimizer.zero_grad()
        pred = model(X)
        main_pred, aux_pred = _unpack_pred(pred, aux_ratio)

        if aux_ratio > 0 and aux_pred is not None:
            # Align aux target size to model's aux output
            if aux is not None and aux.shape[1] != aux_pred.shape[1]:
                if aux.shape[1] < aux_pred.shape[1]:
                    pad = torch.zeros(aux.shape[0], aux_pred.shape[1] - aux.shape[1],
                                      device=aux.device, dtype=aux.dtype)
                    aux = torch.cat([aux, pad], dim=1)
                else:
                    aux = aux[:, :aux_pred.shape[1]]

            epoch_preds.append(main_pred.cpu().detach().numpy())
            epoch_counts.append(y.cpu().detach().numpy())
            aux_preds.append(aux_pred.cpu().detach().numpy())
            aux_counts.append(aux.cpu().detach().numpy())

            main_loss = criterion(main_pred, y)
            aux_loss  = criterion(aux_pred,  aux)
            loss      = main_loss + aux_weight * aux_loss

            total_loss      += loss.item()
            total_main_loss += main_loss.item()
            total_aux_loss  += aux_loss.item()
        else:
            # No aux output (single-head model or aux disabled)
            epoch_preds.append(main_pred.cpu().detach().numpy())
            epoch_counts.append(y.cpu().detach().numpy())

            loss        = criterion(main_pred, y)
            total_loss += loss.item()
            total_main_loss += loss.item()

        # ── Local quantum loss (V2 only) ────────────────────────────────
        # QNNGenePredictorV2 stores its QNN output in _last_q_out
        # and exposes compute_local_quantum_loss() for direct QNN supervision.
        # This provides a gradient signal when the main loss gradient is too small
        # (barren plateau conditions). No-ops for all other models.
        local_q_loss = _compute_local_quantum_loss(model, y)
        if local_q_loss is not None:
            loss = loss + local_q_loss

        loss.backward()
        optimizer.step()

    n = max(len(train_loader), 1)
    total_loss      /= n
    total_main_loss /= n
    total_aux_loss  /= n

    epoch_preds  = np.concatenate(epoch_preds)
    epoch_counts = np.concatenate(epoch_counts)

    if aux_ratio > 0 and aux_preds:
        aux_preds  = np.concatenate(aux_preds)
        aux_counts = np.concatenate(aux_counts)
        main_m = compute_all_metrics(epoch_preds, epoch_counts, prefix="")
        aux_m  = compute_all_metrics(aux_preds,   aux_counts,   prefix="aux_")

        print(f"Total: Loss={total_loss:.4f}")
        print(f"Main:  Loss={total_main_loss:.4f}  aMAE={main_m['amae']:.4f}  "
              f"aRMSE={main_m['armse']:.4f}  aCC={main_m['correlation_coefficient']:.4f}")
        print(f"Aux:   Loss={total_aux_loss:.4f}  aMAE={aux_m['aux_amae']:.4f}  "
              f"aRMSE={aux_m['aux_armse']:.4f}  aCC={aux_m['aux_correlation_coefficient']:.4f}")

        return {'loss': total_main_loss, **main_m, **aux_m}
    else:
        m = compute_all_metrics(epoch_preds, epoch_counts)
        print(f"Loss={total_loss:.4f}  aMAE={m['amae']:.4f}  "
              f"aRMSE={m['armse']:.4f}  aCC={m['correlation_coefficient']:.4f}")
        return {'loss': total_loss, **m}


def validate(
    model:      nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion:  nn.Module,
    aux_ratio:  float,
    aux_weight: float,
    device:     torch.device,
) -> Dict[str, float]:
    """
    One epoch of validation.  Matches the paper's validate() exactly.
    """
    print('Validate:')
    model.eval()

    total_loss = total_main_loss = total_aux_loss = 0.0
    epoch_preds  = []
    epoch_counts = []
    aux_preds    = []
    aux_counts   = []

    with torch.no_grad():
        for batch in val_loader:
            X, y, aux, *_ = _unpack_batch(batch, aux_ratio, device)
            pred = model(X)
            main_pred, aux_pred = _unpack_pred(pred, aux_ratio)

            if aux_ratio > 0 and aux_pred is not None:
                # Align aux target size to model's aux output
                if aux is not None and aux.shape[1] != aux_pred.shape[1]:
                    if aux.shape[1] < aux_pred.shape[1]:
                        pad = torch.zeros(aux.shape[0], aux_pred.shape[1] - aux.shape[1],
                                          device=aux.device, dtype=aux.dtype)
                        aux = torch.cat([aux, pad], dim=1)
                    else:
                        aux = aux[:, :aux_pred.shape[1]]

                epoch_preds.append(main_pred.cpu().numpy())
                epoch_counts.append(y.cpu().numpy())
                aux_preds.append(aux_pred.cpu().numpy())
                aux_counts.append(aux.cpu().numpy())

                main_loss = criterion(main_pred, y)
                aux_loss  = criterion(aux_pred,  aux)
                loss      = main_loss + aux_weight * aux_loss

                total_loss      += loss.item()
                total_main_loss += main_loss.item()
                total_aux_loss  += aux_loss.item()
            else:
                epoch_preds.append(main_pred.cpu().numpy())
                epoch_counts.append(y.cpu().numpy())

                loss        = criterion(main_pred, y)
                total_loss += loss.item()
                total_main_loss += loss.item()

    n = max(len(val_loader), 1)
    total_loss      /= n
    total_main_loss /= n
    total_aux_loss  /= n

    epoch_preds  = np.concatenate(epoch_preds)
    epoch_counts = np.concatenate(epoch_counts)

    if aux_ratio > 0 and aux_preds:
        aux_preds  = np.concatenate(aux_preds)
        aux_counts = np.concatenate(aux_counts)
        main_m = compute_all_metrics(epoch_preds, epoch_counts, prefix="")
        aux_m  = compute_all_metrics(aux_preds,   aux_counts,   prefix="aux_")

        print(f"Total: Loss={total_loss:.4f}")
        print(f"Main:  Loss={total_main_loss:.4f}  aMAE={main_m['amae']:.4f}  "
              f"aRMSE={main_m['armse']:.4f}  aCC={main_m['correlation_coefficient']:.4f}")
        print(f"Aux:   Loss={total_aux_loss:.4f}  aMAE={aux_m['aux_amae']:.4f}  "
              f"aRMSE={aux_m['aux_armse']:.4f}  aCC={aux_m['aux_correlation_coefficient']:.4f}")

        return {'loss': total_main_loss, **main_m, **aux_m}
    else:
        m = compute_all_metrics(epoch_preds, epoch_counts)
        print(f"Loss={total_loss:.4f}  aMAE={m['amae']:.4f}  "
              f"aRMSE={m['armse']:.4f}  aCC={m['correlation_coefficient']:.4f}")
        return {'loss': total_loss, **m}


def evaluate(
    model:       nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion:   nn.Module,
    aux_ratio:   float,
    aux_weight:  float,
    device:      torch.device,
    save_path:   Optional[str] = None,
    epoch:       Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full test/evaluation pass — mirrors validate() but collects metadata.

    Called by SupervisedTrainer._run_evaluate() and cross-validation.
    Optionally saves predictions to a compressed NPZ file.
    """
    print('Evaluate:')
    model.eval()

    total_loss = total_main_loss = total_aux_loss = 0.0
    epoch_preds  = []
    epoch_counts = []
    aux_preds    = []
    aux_counts   = []
    patients, sections, coords, pixels = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            X, y, aux, coord, idx, patient, section, pixel = _unpack_batch(
                batch, aux_ratio, device
            )
            pred = model(X)
            main_pred, aux_pred = _unpack_pred(pred, aux_ratio)

            if aux_ratio > 0 and aux_pred is not None:
                if aux is not None and aux.shape[1] != aux_pred.shape[1]:
                    if aux.shape[1] < aux_pred.shape[1]:
                        pad = torch.zeros(aux.shape[0], aux_pred.shape[1] - aux.shape[1],
                                          device=aux.device, dtype=aux.dtype)
                        aux = torch.cat([aux, pad], dim=1)
                    else:
                        aux = aux[:, :aux_pred.shape[1]]

                epoch_preds.append(main_pred.cpu().numpy())
                epoch_counts.append(y.cpu().numpy())
                aux_preds.append(aux_pred.cpu().numpy())
                aux_counts.append(aux.cpu().numpy())

                main_loss = criterion(main_pred, y)
                aux_loss  = criterion(aux_pred,  aux)
                loss      = main_loss + aux_weight * aux_loss
                total_loss += loss.item(); total_main_loss += main_loss.item(); total_aux_loss += aux_loss.item()
            else:
                epoch_preds.append(main_pred.cpu().numpy())
                epoch_counts.append(y.cpu().numpy())
                loss = criterion(main_pred, y)
                total_loss += loss.item(); total_main_loss += loss.item()

            patients  += list(patient)
            sections  += list(section)
            if isinstance(coord, torch.Tensor):
                coords.append(coord.numpy())
            if isinstance(pixel, torch.Tensor):
                pixels.append(pixel.numpy())

    n = max(len(test_loader), 1)
    total_loss /= n; total_main_loss /= n; total_aux_loss /= n

    epoch_preds  = np.concatenate(epoch_preds)
    epoch_counts = np.concatenate(epoch_counts)

    if aux_ratio > 0 and aux_preds:
        aux_preds  = np.concatenate(aux_preds)
        aux_counts = np.concatenate(aux_counts)
        main_m = compute_all_metrics(epoch_preds, epoch_counts, prefix="")
        aux_m  = compute_all_metrics(aux_preds,   aux_counts,   prefix="aux_")
        results = {'loss': total_main_loss, **main_m, **aux_m}
    else:
        results = {'loss': total_loss, **compute_all_metrics(epoch_preds, epoch_counts)}

    if save_path is not None:
        import pathlib
        save_path_obj = pathlib.Path(save_path)

        # Bug fix 1: mkdir the save_path directory itself, not its parent.
        # When save_path = "results/predictions/", parent = "results/" which
        # already exists — so predictions/ was never created.
        if save_path_obj.suffix == '':
            # save_path is a directory — create it and build filename inside
            save_path_obj.mkdir(parents=True, exist_ok=True)
            epoch_str = f"epoch_{epoch}" if epoch is not None else "final"
            npz_path  = str(save_path_obj / f"predictions_{epoch_str}")
        else:
            # save_path already includes a filename stem
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            suffix   = f"_epoch_{epoch}" if epoch is not None else ""
            npz_path = f"{save_path}{suffix}"

        np.savez_compressed(
            npz_path,
            predictions=epoch_preds,
            counts=epoch_counts,
            patient=patients,
            section=sections,
        )
        logger.info(f"Predictions saved to {npz_path}.npz")

    results['predictions'] = epoch_preds
    results['targets']     = epoch_counts
    return results


# ======================================================================
# Main Trainer Class
# ======================================================================

class SupervisedTrainer:
    """
    Trainer for spatial transcriptomics models.

    Implements the paper's full training strategy:
      1. SGD (lr=1e-3, momentum=0.9, weight_decay=1e-6)
      2. CosineAnnealingLR (T_max=5)
      3. MSELoss
      4. Auxiliary task weighted loss
      5. Per-gene aCC / aMAE / aRMSE metrics
      6. Early stopping
      7. Best-checkpoint restore before evaluation
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model  = model
        self.config = config

        # Device: MPS, CUDA, or CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Hyperparameters — match paper defaults
        def _cfg(key, default):
            keys = key.split('.')
            v = config
            try:
                for k in keys:
                    v = v[k]
                return v
            except (KeyError, TypeError):
                return default

        self.epochs       = _cfg('training.epochs',       50)
        self.lr           = _cfg('training.lr',           1e-3)
        self.learning_rate= _cfg('training.learning_rate',None) or self.lr
        self.momentum     = _cfg('training.momentum',     0.9)
        self.weight_decay = _cfg('training.weight_decay', 1e-6)
        self.aux_ratio    = _cfg('model.aux_ratio',       1.0)
        self.aux_weight   = _cfg('model.aux_weight',      1.0)
        self.save_path    = _cfg('training.pred_root',    None)
        self.cosine_t_max = _cfg('training.cosine_t_max', 5)
        self.batch_size   = _cfg('training.batch_size',   32)
        self.num_workers  = _cfg('training.num_workers',  8)
        self.shuffle      = _cfg('training.shuffle',      True)
        self.debug_mode   = _cfg('training.debug',        False)
        self.optimizer_name = str(_cfg('training.optimizer', 'sgd')).lower()

        # Early stopping
        self.early_stopper = EarlyStopping(
            patience  = _cfg('training.early_stopping_patience', 20),
            min_delta = _cfg('training.early_stopping_delta',    0.0),
        )

        # Move model to device
        self.model.to(self.device)

        # Optimizer — configurable: "sgd" (paper default) or "adam" (better for QML)
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr           = self.lr,
                weight_decay = self.weight_decay,
            )
            opt_str = f"Adam(lr={self.lr}, wd={self.weight_decay})"
        else:
            # Default: SGD matching paper exactly
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr           = self.lr,
                momentum     = self.momentum,
                weight_decay = self.weight_decay,
            )
            opt_str = f"SGD(lr={self.lr}, momentum={self.momentum}, wd={self.weight_decay})"

        # LR Scheduler: CosineAnnealingLR matching paper exactly
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max = self.cosine_t_max,
        )

        # Loss: MSELoss matching paper exactly
        self.criterion = nn.MSELoss()

        # Tracking
        self.train_history: List[Dict] = []
        self.val_history:   List[Dict] = []
        self.train_losses:  List[float] = []
        self.val_losses:    List[float] = []
        self.train_metrics: List[Dict]  = []
        self.val_metrics:   List[Dict]  = []
        self.best_val_loss  = float('inf')
        self.best_loss      = float('inf')
        self.best_metric    = float('-inf')
        self.best_epoch     = 0
        self.best_state     = None
        self.current_epoch  = 0


        # Live results directory — resolved from config or default
        results_base = _cfg('results.base_dir', 'results/')
        self._results_run_dir: Optional[Path] = None
        try:
            from src.utils.result_tracker import get_tracker
            tracker = get_tracker()
            if tracker is not None:
                self._results_run_dir = tracker.run_dir
        except Exception:
            pass
        if self._results_run_dir is None:
            self._results_run_dir = Path(results_base)

        logger.info(
            f"[SupervisedTrainer] device={self.device}  "
            f"{opt_str}  "
            f"CosineAnnealingLR(T_max={self.cosine_t_max})"
        )

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def _run_training(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader:   torch.utils.data.DataLoader,
    ) -> Dict[str, Any]:
        logger.info(f"[SupervisedTrainer] Starting training for {self.epochs} epochs")
        start_time = time.time()

        # ── V2: Phase-aware setup ────────────────────────────────────────
        # If the model is QNNGenePredictorV2, log the active phase and
        # fit the local quantum loss PCA projector on the training set.
        self._setup_v2_model(train_loader)

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            print(f"\nEpoch #{epoch + 1}/{self.epochs}:")

            # Training step
            train_m = fit(
                model        = self.model,
                train_loader = train_loader,
                optimizer    = self.optimizer,
                criterion    = self.criterion,
                aux_ratio    = self.aux_ratio,
                aux_weight   = self.aux_weight,
                device       = self.device,
            )

            # CosineAnnealingLR step
            self.scheduler.step()

            # Validation step
            val_m = validate(
                model      = self.model,
                val_loader = val_loader,
                criterion  = self.criterion,
                aux_ratio  = self.aux_ratio,
                aux_weight = self.aux_weight,
                device     = self.device,
            )

            val_loss = val_m['loss']

            # Log to console
            logger.info(
                f"[SupervisedTrainer] Epoch [{epoch+1}/{self.epochs}] "
                f"Train Loss: {train_m['loss']:.6f}  "
                f"Val Loss: {val_loss:.6f}  "
                f"Val aMAE: {val_m.get('amae', 0):.6f}  "
                f"Val aRMSE: {val_m.get('armse', 0):.6f}  "
                f"Val aCC: {val_m.get('correlation_coefficient', 0):.6f}"
            )

            # Track best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_loss     = val_loss
                self.best_epoch    = epoch
                self.best_state    = {k: v.cpu().clone()
                                      for k, v in self.model.state_dict().items()}
                logger.info(f"  ✓ New best val loss: {self.best_val_loss:.6f} (epoch {epoch+1})")

            # Store history
            self.train_history.append({'epoch': epoch, **train_m})
            self.val_history.append({'epoch': epoch, **val_m})
            self.train_losses.append(train_m['loss'])
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_m)
            self.val_metrics.append(val_m)

            # ── Save live checkpoint after every epoch ─────────────────
            # This ensures results.json exists even if training crashes
            self._save_live_checkpoint()

            # Early stopping
            self.early_stopper(val_loss)
            if self.early_stopper.early_stop:
                logger.info(f"Early stopping at epoch {epoch+1}. Best: epoch {self.best_epoch+1}")
                break

        training_time = time.time() - start_time
        logger.info(
            f"[SupervisedTrainer] Training complete — "
            f"best epoch: {self.best_epoch+1}, "
            f"best val loss: {self.best_val_loss:.6f}, "
            f"time: {training_time:.1f}s"
        )

        return {
            'final_epoch':     self.current_epoch,
            'best_epoch':      self.best_epoch,
            'best_val_loss':   self.best_val_loss,
            'best_val_metric': self.best_val_loss,
            'training_time':   training_time,
            'train_losses':    self.train_losses,
            'val_losses':      self.val_losses,
            'train_metrics':   self.train_metrics,
            'val_metrics':     self.val_metrics,
        }

    def _run_evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
        restore_best: bool = True,
    ) -> Dict[str, Any]:
        if restore_best and self.best_state is not None:
            logger.info(f"Restoring best weights from epoch {self.best_epoch+1}")
            self.model.load_state_dict(self.best_state)

        start_time = time.time()

        # Resolve save_path — always pass as a directory so evaluate()
        # builds the filename itself (avoids empty-stem "_epoch_N.npz" bug).
        # pred_root = "results/predictions/" → evaluate() writes
        #   "results/predictions/predictions_epoch_29.npz"
        pred_root = self.save_path  # may be None — evaluate() skips saving if None

        results    = evaluate(
            model       = self.model,
            test_loader = test_loader,
            criterion   = self.criterion,
            aux_ratio   = self.aux_ratio,
            aux_weight  = self.aux_weight,
            device      = self.device,
            save_path   = pred_root,
            epoch       = self.best_epoch,
        )
        results['evaluation_time'] = time.time() - start_time

        logger.info(
            f"[SupervisedTrainer] Evaluation complete — "
            f"aMAE: {results.get('amae', 'N/A'):.4f}  "
            f"aRMSE: {results.get('armse', 'N/A'):.4f}  "
            f"aCC: {results.get('correlation_coefficient', 'N/A'):.4f}"
        )

        # Return format expected by pipeline_orchestrator
        return {
            'loss':            results.get('loss', 0),
            'metrics': {
                'mae':                   results.get('amae', 0),
                'amae':                  results.get('amae', 0),
                'rmse':                  results.get('armse', 0),
                'armse':                 results.get('armse', 0),
                'correlation_coefficient': results.get('correlation_coefficient', 0),
            },
            'evaluation_time': results['evaluation_time'],
            'predictions':     results.get('predictions'),
            'targets':         results.get('targets'),
        }

    # ------------------------------------------------------------------
    # V2-specific setup helpers
    # ------------------------------------------------------------------

    def _setup_v2_model(self, train_loader) -> None:
        """
        V2-specific setup called once before the epoch loop.

        For QNNGenePredictorV2:
          1. Log the current training phase (0/1/2).
          2. If local_quantum_loss is enabled, collect gene targets from
             the first N batches and fit the PCA projector.

        No-ops for all other model types.
        """
        model = self.model

        if not hasattr(model, "training_phase"):
            return  # Not a V2 model — skip

        phase = getattr(model, "training_phase", 0)
        phase_names = {0: "Classical Warmup", 1: "Quantum Training", 2: "Joint Fine-tuning"}
        logger.info(
            f"[SupervisedTrainer] QNNGenePredictorV2 — "
            f"Phase {phase}: {phase_names.get(phase, 'Unknown')}"
        )

        # Fit local quantum loss PCA projector if enabled
        if getattr(model, "use_local_loss", False) and \
           hasattr(model, "fit_local_loss_projector"):
            logger.info(
                "[SupervisedTrainer] Collecting gene targets for local quantum loss PCA projector..."
            )
            all_targets = []
            max_batches = 50  # Limit to first 50 batches to keep startup fast
            try:
                for i, batch in enumerate(train_loader):
                    if i >= max_batches:
                        break
                    _, y, *_ = _unpack_batch(batch, self.aux_ratio, torch.device("cpu"))
                    all_targets.append(y.float().numpy())

                if all_targets:
                    targets_np = np.concatenate(all_targets, axis=0)
                    model.fit_local_loss_projector(targets_np)
                    logger.info(
                        f"[SupervisedTrainer] Local quantum loss projector fitted on "
                        f"{len(targets_np)} samples"
                    )
            except Exception as e:
                logger.warning(
                    f"[SupervisedTrainer] Could not fit local loss projector: {e} — disabling"
                )
                if hasattr(model, "use_local_loss"):
                    model.use_local_loss = False

    # ------------------------------------------------------------------
    # Data loader preparation (synthetic fallback for testing)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Data loader preparation — matches paper's setup() exactly
    # ------------------------------------------------------------------

    def _prepare_data_loaders(self):
        """
        Build real SpatialDataset loaders from config paths.

        After building the dataset, syncs model.aux_nums with
        dataset.aux_nums if they differ (fixes the hardcoded-6000 bug).
        """
        try:
            from src.training.data_generator import create_dataloaders

            class _Paths:
                def __init__(self, cfg):
                    data = cfg.get('data', {})
                    self.train_counts = data.get(
                        'train_counts_dir', 'data/train/counts/'
                    )
                    self.train_images = data.get(
                        'train_images_dir', 'data/train/images/'
                    )
                    self.test_counts  = data.get(
                        'test_counts_dir', 'data/test/counts/'
                    )
                    self.test_images  = data.get(
                        'test_images_dir', 'data/test/images/'
                    )

            paths = _Paths(self.config)
            # Pass test_patient=None — create_dataloaders resolves from config:
            #   Priority 1: data.test_patient  (authoritative — single string)
            #   Priority 2: split.test_patients[0]  (fallback — list form)
            # Will raise ValueError with a clear message if neither is set.
            train_loader, val_loader, train_ds, val_ds = create_dataloaders(
                paths, self.config, test_patient=None
            )

            # ── Sync aux_nums between dataset and model ───────────────
            # The model may have been built before the dataset was loaded
            # (aux_nums=0 or from a stale config value).
            # Paper: aux_nums = (total_genes - gene_filter) * aux_ratio
            dataset_aux_nums = getattr(train_ds, 'aux_nums', 0)
            model_inner = (
                self.model.module
                if isinstance(self.model, torch.nn.DataParallel)
                else self.model
            )
            model_aux_nums = getattr(model_inner, 'aux_nums', 0)

            if dataset_aux_nums != model_aux_nums and dataset_aux_nums > 0:
                logger.info(
                    f"[SupervisedTrainer] Syncing aux_nums: "
                    f"model had {model_aux_nums}, dataset has {dataset_aux_nums}. "
                    f"Rebuilding model head."
                )
                if hasattr(model_inner, 'set_aux_head'):
                    model_inner.set_aux_head(dataset_aux_nums)
                else:
                    logger.warning(
                        "Model does not support set_aux_head(). "
                        "aux_nums mismatch may cause shape errors."
                    )
                # Update trainer's aux_ratio reference
                self.aux_ratio = self.config.get('model', {}).get('aux_ratio', 1.0)

            logger.info(
                f"[SupervisedTrainer] Using real SpatialDataset loaders "
                f"(train: {len(train_ds)}, test: {len(val_ds)}, "
                f"aux_nums: {dataset_aux_nums})"
            )
            return train_loader, val_loader

        except Exception as e:
            logger.warning(
                f"[SupervisedTrainer] Could not build real loaders: {e}. "
                "Using synthetic data."
            )
            return self._synthetic_loaders()

    def _prepare_test_loader(self):
        """
        Build test loader from config using TRAINING normalization stats.

        CRITICAL: the test loader must use the SAME normalization statistics
        as the training loader (mean/std computed from training data only).
        This method rebuilds the full pipeline to ensure consistency.
        """
        try:
            from src.training.data_generator import create_dataloaders
            from torch.utils.data import DataLoader

            class _Paths:
                def __init__(self, cfg):
                    data = cfg.get('data', {})
                    self.train_counts = data.get(
                        'train_counts_dir', 'data/train/counts/'
                    )
                    self.train_images = data.get(
                        'train_images_dir', 'data/train/images/'
                    )
                    self.test_counts  = data.get(
                        'test_counts_dir', 'data/test/counts/'
                    )
                    self.test_images  = data.get(
                        'test_images_dir', 'data/test/images/'
                    )

            paths = _Paths(self.config)
            # Pass test_patient=None — create_dataloaders resolves from config.
            # Rebuild full pipeline so test dataset inherits training norm stats.
            _, test_loader, _, _ = create_dataloaders(
                paths, self.config, test_patient=None
            )

            logger.info("[SupervisedTrainer] Test loader built with training normalization stats")
            return test_loader

        except Exception as e:
            logger.warning(
                f"[SupervisedTrainer] Could not build test loader: {e}. "
                "Using synthetic data."
            )
            _, val_loader = self._synthetic_loaders(test_only=True)
            return val_loader

    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop.

        Data loaders are built here from config paths.
        Returns a results dict compatible with pipeline_orchestrator.
        """
        train_loader, val_loader = self._prepare_data_loaders()
        return self._run_training(train_loader, val_loader)

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate on the test set using BEST saved weights.

        CRITICAL FIX: The pipeline_orchestrator creates a FRESH model
        instance for evaluation. This means best_state is None and
        we must load from the saved checkpoint file instead.
        """
        test_loader = self._prepare_test_loader()

        # Restore best weights if available from this training run
        if self.best_state is not None:
            logger.info(
                f"[SupervisedTrainer] Restoring best weights "
                f"from epoch {self.best_epoch + 1} "
                f"(val loss: {self.best_val_loss:.6f})"
            )
            self.model.load_state_dict(self.best_state)
        else:
            # Orchestrator created a fresh model — try to load from disk
            model_path = self._find_saved_model()
            if model_path and model_path.exists():
                logger.info(
                    f"[SupervisedTrainer] Loading saved model from {model_path}"
                )
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if isinstance(checkpoint, dict):
                        state_dict = checkpoint.get(
                            'model_state_dict',
                            checkpoint.get('state_dict', checkpoint)
                        )
                        self.model.load_state_dict(state_dict, strict=False)
                    else:
                        # Saved as full model
                        logger.info("Checkpoint is full model — extracting state_dict")
                        self.model.load_state_dict(
                            checkpoint.state_dict(), strict=False
                        )
                except Exception as e:
                    logger.warning(
                        f"Could not load model checkpoint: {e}. "
                        "Evaluating with current (untrained) weights."
                    )
            else:
                logger.warning(
                    "[SupervisedTrainer] No best_state and no saved checkpoint found. "
                    "Evaluating with randomly initialized weights — results will be meaningless."
                )

        return self._run_evaluate(test_loader)

    def _find_saved_model(self) -> Optional[Path]:
        """
        Locate the most recently saved model checkpoint.

        Searches in the result_tracker run directory.
        """
        try:
            run_dir = Path(self._results_run_dir)
            # Look for model files saved by result_tracker.save_model()
            candidates = sorted(
                run_dir.rglob("*.pth"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if candidates:
                logger.info(f"Found saved model: {candidates[0]}")
                return candidates[0]
        except Exception as e:
            logger.debug(f"Could not search for saved models: {e}")
        return None


    def _synthetic_loaders(self, test_only: bool = False):
        """Synthetic data loaders for unit testing / CI."""
        from torch.utils.data import TensorDataset, DataLoader

        genes = getattr(self.model, 'output_genes', 250)
        C, H, W = (getattr(self.model, 'input_channels', 3),
                   getattr(self.model, 'input_height',   224),
                   getattr(self.model, 'input_width',    224))

        def _make(n):
            imgs   = torch.randn(n, C, H, W)
            counts = torch.rand(n, genes)
            return DataLoader(TensorDataset(imgs, counts),
                              batch_size=self.batch_size, shuffle=not test_only,
                              num_workers=0)

        if test_only:
            return None, _make(200)

        logger.info("[SupervisedTrainer] Prepared synthetic data loaders")
        return _make(640), _make(160)

    # ------------------------------------------------------------------
    # Config helper (legacy compat with BaseTrainer callers)
    # ------------------------------------------------------------------

    def get_config_value(self, key: str, default=None):
        keys  = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def log_info(self, msg: str):
        logger.info(f"[SupervisedTrainer] {msg}")

    def save_model(self, path: str) -> None:
        torch.save(self.model, path)
        logger.info(f"[SupervisedTrainer] Model saved to {path}")

    def _save_live_checkpoint(self) -> None:
        """
        Write a live results.json after every epoch.

        This ensures training history is persisted even if the process
        crashes before the pipeline orchestrator calls save_results().
        The file is overwritten each epoch with the latest state.
        """
        import json as _json
        try:
            run_dir = Path(self._results_run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            results_file = run_dir / "results.json"

            def _safe(v):
                """Convert numpy/float types to plain Python for JSON."""
                import numpy as _np
                if isinstance(v, (_np.floating, float)):
                    return float(v)
                if isinstance(v, (_np.integer, int)):
                    return int(v)
                if isinstance(v, _np.ndarray):
                    return v.tolist()
                return v

            def _clean_metrics(history):
                return [{k: _safe(v) for k, v in m.items() if k != 'epoch'}
                        for m in history if isinstance(m, dict)]

            snapshot = {
                'status': 'training',
                'current_epoch': self.current_epoch + 1,
                'best_epoch':    self.best_epoch + 1,
                'best_val_loss': _safe(self.best_val_loss),
                'train_losses':  [_safe(v) for v in self.train_losses],
                'val_losses':    [_safe(v) for v in self.val_losses],
                'train_metrics': _clean_metrics(self.train_metrics),
                'val_metrics':   _clean_metrics(self.val_metrics),
            }
            with open(results_file, 'w') as f:
                _json.dump(snapshot, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"[SupervisedTrainer] Could not write live checkpoint: {e}")


# ======================================================================
# Cross-Validation Support  (matches paper's get_cv_results)
# ======================================================================

def run_cross_validation(
    all_patients: List[str],
    cv_folds:     int,
    config:       Dict[str, Any],
    paths,
    device:       torch.device,
) -> int:
    """
    Run k-fold CV to determine the optimal training epoch.
    Returns best_epoch (int, 1-based index).
    """
    from src.training.data_generator import create_dataloaders

    folds = [all_patients[f::cv_folds] for f in range(cv_folds)]

    fold_losses = []
    fold_maes   = []
    fold_rmses  = []
    fold_accs   = []

    logger.info(f"Starting {cv_folds}-fold cross-validation")

    for f in range(cv_folds):
        print(f"\n### Fold {f+1}/{cv_folds} ###")

        train_pts = [p for i, fold in enumerate(folds) if i != f for p in fold]
        val_pts   = folds[f]

        train_loader, val_loader, train_ds, _ = create_dataloaders(
            paths, config, test_patient=val_pts[0]
        )

        config['model.total_genes'] = len(train_ds.ensg_names)

        from src.models.classical.efficientnet_model import EfficientNetModel
        model   = EfficientNetModel(config)
        trainer = SupervisedTrainer(model=model, config=config)

        cv_epochs  = config.get('training', {}).get('cv_epochs', 50)
        loss_h, mae_h, rmse_h, acc_h = [], [], [], []

        for epoch in range(cv_epochs):
            fit(trainer.model, train_loader, trainer.optimizer,
                trainer.criterion, trainer.aux_ratio, trainer.aux_weight, device)
            trainer.scheduler.step()

            vm = validate(trainer.model, val_loader, trainer.criterion,
                          trainer.aux_ratio, trainer.aux_weight, device)

            loss_h.append(vm['loss'])
            mae_h.append(vm.get('amae',   0.0))
            rmse_h.append(vm.get('armse', 0.0))
            acc_h.append(vm.get('correlation_coefficient', 0.0))

        fold_losses.append(loss_h)
        fold_maes.append(mae_h)
        fold_rmses.append(rmse_h)
        fold_accs.append(acc_h)

    best_loss_e = int(np.argmin(np.vstack(fold_losses).mean(axis=0)))
    best_mae_e  = int(np.argmin(np.vstack(fold_maes).mean(axis=0)))
    best_rmse_e = int(np.argmin(np.vstack(fold_rmses).mean(axis=0)))
    best_acc_e  = int(np.argmax(np.vstack(fold_accs).mean(axis=0)))

    best_epoch = math.ceil(
        np.mean([best_loss_e, best_mae_e, best_rmse_e, best_acc_e])
    )
    best_epoch = max(best_epoch, 3)

    logger.info(
        f"CV → loss_e={best_loss_e} mae_e={best_mae_e} "
        f"rmse_e={best_rmse_e} acc_e={best_acc_e} → best={best_epoch}"
    )
    return best_epoch


# Legacy alias kept so factory.py still resolves 'supervised_trainer'
SpatialTrainer = SupervisedTrainer

__all__ = [
    'SupervisedTrainer',
    'SpatialTrainer',
    'EarlyStopping',
    'fit',
    'validate',
    'evaluate',
    'run_cross_validation',
]

