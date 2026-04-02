# src/models/classical/efficientnet_model.py

"""
EfficientNet-B4 model for spatial transcriptomics gene expression prediction.

Architecture (paper):
  Backbone : EfficientNet-B4 (pretrained ImageNet, all layers fine-tuned)
  Main head: Linear(1792 → gene_filter)        — top-250 genes
  Aux head : Linear(1792 → aux_nums)           — remaining genes (log1p only)

FIXES IN THIS VERSION:
  1. Aux head always created on correct device — no CPU/MPS mismatch
  2. set_aux_head() moves new head to device immediately
  3. Device property exposed so callers can create tensors on same device
  4. total_genes resolved from config at both top-level and model-level keys
"""

import logging
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EfficientNetModel(nn.Module):
    """
    EfficientNet-B4 with dual output heads for gene expression prediction.

    Args:
        config: Pipeline config dict. Reads from both top-level and
                config['model'] sub-dict (top-level takes priority).

    Expected config keys:
        model.gene_filter  (int)   : Number of main genes to predict. Default 250.
        model.aux_ratio    (float) : Fraction of remaining genes for aux head. Default 1.0.
        model.pretrained   (bool)  : Load ImageNet weights. Default True.
        model.finetuning   (str)   : 'ftall' | 'ft1' | 'ft2'. Default 'ftall'.
        total_genes        (int)   : Total genes in dataset (sets aux_nums).
                                     Also checked at model.total_genes.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # ── Resolve config — top-level keys override model sub-dict ──
        model_cfg   = config.get("model", {})

        self.gene_filter = int(
            config.get("gene_filter",
                       model_cfg.get("gene_filter", 250))
        )
        self.aux_ratio   = float(
            config.get("aux_ratio",
                       model_cfg.get("aux_ratio", 1.0))
        )
        pretrained       = bool(
            config.get("pretrained",
                       model_cfg.get("pretrained", True))
        )
        finetuning       = str(
            config.get("finetuning",
                       model_cfg.get("finetuning", "ftall"))
        )

        # total_genes — checked at top-level first, then model sub-dict
        total_genes = (
                config.get("total_genes") or
                model_cfg.get("total_genes")
        )

        if total_genes is None:
            logger.warning(
                "total_genes not provided — aux_nums set to 0. "
                "Trainer must call model.set_aux_head(dataset.aux_nums) "
                "after building the dataset."
            )
            self.aux_nums = 0
        else:
            self.aux_nums = int(
                (int(total_genes) - self.gene_filter) * self.aux_ratio
            )

        # ── Device selection ──────────────────────────────────────────
        self._device = self._select_device()

        # ── Backbone ─────────────────────────────────────────────────
        self.backbone, self._feature_dim = self._build_backbone(pretrained)
        self._apply_finetuning(finetuning)

        # ── Main prediction head ──────────────────────────────────────
        self.main_head = nn.Linear(self._feature_dim, self.gene_filter)

        # ── Aux prediction head ───────────────────────────────────────
        # Created as None when aux_nums=0; installed via set_aux_head()
        self.aux_head: Optional[nn.Linear] = (
            nn.Linear(self._feature_dim, self.aux_nums)
            if self.aux_nums > 0 else None
        )

        # ── Move ENTIRE model to device in one call ───────────────────
        # This guarantees backbone + main_head + aux_head all on device.
        self.to(self._device)

        logger.info(
            f"EfficientNetModel initialised — "
            f"main_genes: {self.gene_filter}, "
            f"aux_genes: {self.aux_nums}, "
            f"finetuning: {finetuning}, "
            f"device: {self._device}"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Device the model is on. Use this to create input tensors."""
        return self._device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_aux_head(self, aux_nums: int) -> None:
        """
        Install or replace the aux head after the dataset is known.

        Called by the trainer when total_genes was not available at
        model construction time.

        Args:
            aux_nums: Number of auxiliary genes to predict.
        """
        if self.aux_nums == aux_nums and self.aux_head is not None:
            return  # already correct — no-op

        self.aux_nums = aux_nums
        self.aux_head = nn.Linear(self._feature_dim, aux_nums)

        # FIX: move new head to same device as rest of model immediately
        self.aux_head = self.aux_head.to(self._device)

        logger.info(f"Aux head installed: Linear({self._feature_dim} → {aux_nums}) "
                    f"on {self._device}")

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: RGB image tensor, shape (B, 3, H, W), on model.device.

        Returns:
            (main_pred, aux_pred)
              main_pred : (B, gene_filter)
              aux_pred  : (B, aux_nums) or None if aux head not installed
        """
        features  = self.backbone(x)     # (B, feature_dim)
        main_pred = self.main_head(features)

        aux_pred  = (
            self.aux_head(features)
            if self.aux_head is not None else None
        )

        return main_pred, aux_pred

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("[EfficientNetModel] Using CUDA GPU")
        elif (hasattr(torch.backends, "mps") and
              torch.backends.mps.is_available()):
            device = torch.device("mps")
            logger.info("[EfficientNetModel] Using Apple Metal Performance Shaders (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("[EfficientNetModel] Using CPU")
        return device

    @staticmethod
    def _build_backbone(
            pretrained: bool,
    ) -> Tuple[nn.Module, int]:
        """
        Build EfficientNet-B4 backbone with pooling, without classifier.

        Returns:
            (backbone_module, feature_dim)
        """
        feature_dim = 1792   # EfficientNet-B4 penultimate dimension

        # Try efficientnet_pytorch first (paper's original)
        try:
            from efficientnet_pytorch import EfficientNet
            if pretrained:
                # from_pretrained() loads ImageNet weights — no extra kwargs needed
                net = EfficientNet.from_pretrained("efficientnet-b4")
                logger.info("Loaded efficientnet_pytorch EfficientNet-B4 (pretrained)")
            else:
                net = EfficientNet.from_name("efficientnet-b4")
                logger.info("Loaded efficientnet_pytorch EfficientNet-B4 (random init)")

            # Replace classifier with identity so backbone outputs (B, feature_dim)
            net._fc = nn.Identity()

            backbone = nn.Sequential(net)
            return backbone, feature_dim

        except ImportError:
            logger.info("efficientnet_pytorch not found — trying torchvision fallback")

        # Torchvision fallback
        try:
            import torchvision.models as M

            weights = M.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            net     = M.efficientnet_b4(weights=weights)

            # Remove classifier, keep adaptive avgpool
            backbone = nn.Sequential(
                net.features,                          # conv + BN + activation blocks
                net.avgpool,                           # AdaptiveAvgPool2d(1,1)
                nn.Flatten(start_dim=1),               # (B, 1792)
            )
            logger.info("Loaded torchvision EfficientNet-B4 as fallback")
            return backbone, feature_dim

        except Exception as e:
            raise RuntimeError(
                f"Could not load EfficientNet-B4 from either "
                f"efficientnet_pytorch or torchvision: {e}"
            ) from e

    def _apply_finetuning(self, mode: str) -> None:
        """
        Set which backbone parameters are trainable.

        Modes:
          ftall : all parameters trainable (paper default)
          ft1   : only last block trainable
          ft2   : last two blocks trainable
          frozen: backbone frozen, heads only
        """
        if mode == "ftall":
            for p in self.backbone.parameters():
                p.requires_grad = True
            logger.info("Fine-tuning: ftall — all parameters trainable")

        elif mode == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("Fine-tuning: frozen — backbone frozen")

        elif mode in ("ft1", "ft2"):
            for p in self.backbone.parameters():
                p.requires_grad = False

            # Unfreeze last N children of backbone
            n_unfreeze = 1 if mode == "ft1" else 2
            children   = list(self.backbone.children())
            for child in children[-n_unfreeze:]:
                for p in child.parameters():
                    p.requires_grad = True

            logger.info(f"Fine-tuning: {mode} — "
                        f"last {n_unfreeze} block(s) trainable")

        else:
            logger.warning(f"Unknown finetuning mode '{mode}' — "
                           f"defaulting to ftall")
            for p in self.backbone.parameters():
                p.requires_grad = True
