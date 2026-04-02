"""
Metrics module for spatial transcriptomics gene expression prediction.
Implements per-gene Pearson correlation (aCC) as per the reference paper.
"""

import numpy as np
import torch
from typing import Union


def average_correlation_coefficient(
        y_pred: Union[np.ndarray, torch.Tensor],
        y_true: Union[np.ndarray, torch.Tensor],
        eps: float = 1e-8
) -> float:
    """
    Calculate Average Correlation Coefficient (aCC).

    Computes per-gene Pearson correlation coefficient across all spots,
    then averages across all genes. This matches the reference paper's
    metric computation exactly.

    Reference paper formula:
        aCC = (1/d) * sum_{j=1}^{d} corr(y_pred[:, j], y_true[:, j])

    Args:
        y_pred : Predicted values  — shape (N_spots, d_genes)
        y_true : True values       — shape (N_spots, d_genes)
        eps    : Small epsilon to avoid division by zero

    Returns:
        float: Average per-gene Pearson correlation coefficient

    Raises:
        ValueError: If inputs are not both ndarray or both Tensor
    """
    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        # Per-gene mean (axis=0 → across N spots)
        y_true_mean = np.mean(y_true, axis=0)   # shape: (d,)
        y_pred_mean = np.mean(y_pred, axis=0)   # shape: (d,)

        # Per-gene numerator: covariance × N
        top = np.sum(
            (y_true - y_true_mean) * (y_pred - y_pred_mean),
            axis=0
        )  # shape: (d,)

        # Per-gene denominator: product of std devs × N
        bottom = np.sqrt(
            np.sum((y_true - y_true_mean) ** 2, axis=0) *
            np.sum((y_pred - y_pred_mean) ** 2, axis=0)
        ) + eps  # shape: (d,)

        # Per-gene Pearson r, averaged across d genes
        per_gene_r = top / bottom          # shape: (d,)
        return float(np.mean(per_gene_r))  # scalar

    elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        y_true_mean = torch.mean(y_true, dim=0)   # shape: (d,)
        y_pred_mean = torch.mean(y_pred, dim=0)   # shape: (d,)

        top = torch.sum(
            (y_true - y_true_mean) * (y_pred - y_pred_mean),
            dim=0
        )  # shape: (d,)

        bottom = torch.sqrt(
            torch.sum((y_true - y_true_mean) ** 2, dim=0) *
            torch.sum((y_pred - y_pred_mean) ** 2, dim=0)
        ) + eps  # shape: (d,)

        per_gene_r = top / bottom
        return float(torch.mean(per_gene_r).item())

    else:
        raise ValueError(
            "y_true and y_pred must both be numpy.ndarray or torch.Tensor. "
            f"Got: y_true={type(y_true)}, y_pred={type(y_pred)}"
        )


def average_mae(
        y_pred: np.ndarray,
        y_true: np.ndarray
) -> float:
    """
    Average Mean Absolute Error across all spots and genes.

    Args:
        y_pred: Predicted values (N, d)
        y_true: True values (N, d)

    Returns:
        float: aMAE scalar
    """
    return float(np.mean(np.abs(y_pred - y_true)))


def average_rmse(
        y_pred: np.ndarray,
        y_true: np.ndarray
) -> float:
    """
    Average Root Mean Squared Error across all spots and genes.

    Args:
        y_pred: Predicted values (N, d)
        y_true: True values (N, d)

    Returns:
        float: aRMSE scalar
    """
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def compute_all_metrics(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        prefix: str = ""
) -> dict:
    """
    Compute all metrics at once.

    Args:
        y_pred : Predicted values (N_spots, d_genes)
        y_true : True values      (N_spots, d_genes)
        prefix : Optional prefix for metric keys (e.g. 'main_', 'aux_')

    Returns:
        Dictionary of metric name → value
    """
    return {
        f"{prefix}amae":                    average_mae(y_pred, y_true),
        f"{prefix}armse":                   average_rmse(y_pred, y_true),
        f"{prefix}correlation_coefficient": average_correlation_coefficient(y_pred, y_true),
    }


__all__ = [
    'average_correlation_coefficient',
    'average_mae',
    'average_rmse',
    'compute_all_metrics',
]
