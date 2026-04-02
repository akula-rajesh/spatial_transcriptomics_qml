"""
Quantum Amplitude Embedding model for spatial transcriptomics.

Architecture:
  Classical backbone : EfficientNet-B4 → feature vector (1792-dim)
  Dimensionality red : Linear(1792 → 2^n_qubits)   e.g. 1792 → 8
  Amplitude embedding: AngleEmbedding or AmplitudeEmbedding
  Variational circuit : StronglyEntanglingLayers(weights shape: (n_layers, n_qubits, 3))
  Classical output   : Linear(n_qubits → gene_filter)
  Aux head           : Linear(n_qubits → aux_nums)

FIXES:
  1. Weight shape: (n_layers, n_qubits, 3) — the 3 is ALWAYS 3
     (Rx, Ry, Rz rotations per qubit), NOT n_qubits.
     StronglyEntanglingLayers always uses 3 rotation params per qubit.
  2. set_aux_head() added so trainer can install aux head after dataset built.
  3. Device consistency: quantum output moved to model device before heads.
"""

import logging
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# ── PennyLane imports (optional) ──────────────────────────────────────
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane not installed — quantum circuit disabled. "
                   "Run: pip install pennylane")


class QuantumAmplitudeEmbeddingModel(nn.Module):
    """
    Hybrid Classical-Quantum model using amplitude embedding.

    Config keys (read from config['model'] or top-level):
        n_qubits      (int)  : Number of qubits. Default 3.
                               Circuit input dim = 2^n_qubits.
        n_layers      (int)  : VQC depth (StronglyEntanglingLayers). Default 2.
        gene_filter   (int)  : Main output genes. Default 250.
        aux_ratio     (float): Aux genes fraction. Default 1.0.
        total_genes   (int)  : Total genes (sets aux_nums). Optional.
        pretrained    (bool) : ImageNet backbone weights. Default True.
        finetuning    (str)  : 'ftall'|'ft1'|'ft2'|'frozen'. Default 'ftall'.

    Weight shape contract (PennyLane StronglyEntanglingLayers):
        weights.shape == (n_layers, n_qubits, 3)
        The trailing 3 is ALWAYS 3 = (Rx, Ry, Rz) — fixed by PennyLane API.
        It does NOT equal n_qubits.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # ── Resolve config ────────────────────────────────────────────
        model_cfg = config.get("model", {})

        def _get(key: str, default):
            return config.get(key, model_cfg.get(key, default))

        self.n_qubits    = int(_get("n_qubits",   3))
        self.n_layers    = int(_get("n_layers",   2))
        self.gene_filter = int(_get("gene_filter", 250))
        self.aux_ratio   = float(_get("aux_ratio", 1.0))
        pretrained       = bool(_get("pretrained", True))
        finetuning       = str(_get("finetuning",  "ftall"))

        total_genes = config.get("total_genes") or model_cfg.get("total_genes")
        if total_genes is None:
            logger.warning("total_genes not provided — aux_nums=0. "
                           "Call set_aux_head(aux_nums) after dataset is built.")
            self.aux_nums = 0
        else:
            self.aux_nums = int(
                (int(total_genes) - self.gene_filter) * self.aux_ratio
            )

        # Quantum input dimension = 2^n_qubits
        self._q_dim = 2 ** self.n_qubits   # 3 qubits → 8, 4 qubits → 16

        # ── Device ───────────────────────────────────────────────────
        self._device = self._select_device()

        # ── Classical backbone (EfficientNet-B4) ─────────────────────
        self.backbone, self._feature_dim = self._build_backbone(pretrained)
        self._apply_finetuning(finetuning)

        # ── Dimensionality reduction: 1792 → 2^n_qubits ──────────────
        self.pre_quantum = nn.Sequential(
            nn.Linear(self._feature_dim, self._q_dim),
            nn.Tanh(),    # Tanh keeps values in [-1, 1] for angle embedding
        )

        # ── Quantum circuit ───────────────────────────────────────────
        self._quantum_layer = self._build_quantum_layer()

        # ── FIX: Weight shape = (n_layers, n_qubits, 3) ──────────────
        # The trailing 3 is ALWAYS 3 regardless of n_qubits.
        # StronglyEntanglingLayers uses 3 rotation gates (Rx, Ry, Rz)
        # per qubit per layer. This is a PennyLane API constant.
        #
        # WRONG (previous bug):  shape = (n_layers, n_qubits)  → dim[-1]=n_qubits
        # CORRECT:               shape = (n_layers, n_qubits, 3) → dim[-1]=3
        self.q_weights = nn.Parameter(
            torch.randn(self.n_layers, self.n_qubits, 3, dtype=torch.float32) * 0.1
        )

        logger.info(
            f"Quantum weights shape: {list(self.q_weights.shape)} "
            f"= (n_layers={self.n_layers}, n_qubits={self.n_qubits}, 3)"
        )

        # ── Output heads ──────────────────────────────────────────────
        # Quantum circuit outputs n_qubits expectation values
        self.main_head = nn.Linear(self.n_qubits, self.gene_filter)
        self.aux_head: Optional[nn.Linear] = (
            nn.Linear(self.n_qubits, self.aux_nums)
            if self.aux_nums > 0 else None
        )

        # ── Move all to device ────────────────────────────────────────
        self.to(self._device)

        logger.info(
            f"QuantumAmplitudeEmbeddingModel initialised — "
            f"n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"q_dim={self._q_dim}, "
            f"main_genes={self.gene_filter}, aux_genes={self.aux_nums}, "
            f"device={self._device}"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_aux_head(self, aux_nums: int) -> None:
        """
        Install or replace aux head after dataset is known.
        Called by SupervisedTrainer when total_genes was not available
        at model construction time.
        """
        if self.aux_nums == aux_nums and self.aux_head is not None:
            return

        self.aux_nums = aux_nums
        self.aux_head = nn.Linear(self.n_qubits, aux_nums).to(self._device)
        logger.info(f"Aux head installed: Linear({self.n_qubits} → {aux_nums}) "
                    f"on {self._device}")

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Image tensor (B, 3, H, W) on model.device.

        Returns:
            (main_pred, aux_pred)
        """
        # Classical feature extraction
        features    = self.backbone(x)                    # (B, 1792) float32
        q_input     = self.pre_quantum(features).float()  # (B, 2^n_qubits) float32

        # Quantum layer — handles CPU transfer and float32 cast internally
        q_output    = self._run_quantum(q_input)          # (B, n_qubits) float32

        # Classical output heads
        main_pred   = self.main_head(q_output)
        aux_pred    = self.aux_head(q_output) if self.aux_head is not None else None

        return main_pred, aux_pred

    # ------------------------------------------------------------------
    # Quantum helpers
    # ------------------------------------------------------------------

    def _build_quantum_layer(self):
        """Build PennyLane quantum circuit if available."""
        if not PENNYLANE_AVAILABLE:
            logger.warning("PennyLane not available — using classical fallback")
            return None

        try:
            dev = qml.device("default.qubit", wires=self.n_qubits)

            # diff_method="parameter-shift":
            #   - Works on ANY device (CPU, MPS, CUDA)
            #   - Does NOT require CUDA-compiled PyTorch
            #   - "backprop" requires torch+CUDA and fails on MPS
            @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
            def circuit(inputs, weights):
                """
                Variational quantum circuit.

                Args:
                    inputs  : (2^n_qubits,) amplitude-embedded features
                              — must be on CPU (PennyLane runs on CPU)
                    weights : (n_layers, n_qubits, 3) rotation angles
                              — the trailing 3 is ALWAYS 3 (Rx, Ry, Rz)
                              — must be on CPU
                """
                qml.AmplitudeEmbedding(
                    inputs,
                    wires=range(self.n_qubits),
                    normalize=True,
                    pad_with=0.0,
                )
                qml.StronglyEntanglingLayers(
                    weights,
                    wires=range(self.n_qubits),
                )
                return [qml.expval(qml.PauliZ(i))
                        for i in range(self.n_qubits)]

            logger.info(
                f"Quantum circuit built: {self.n_qubits} qubits, "
                f"{self.n_layers} layers, diff_method=parameter-shift, "
                f"weights shape=({self.n_layers}, {self.n_qubits}, 3)"
            )
            return circuit

        except Exception as e:
            logger.warning(f"Failed to build quantum circuit: {e} "
                           "— using classical fallback")
            return None

    def _run_quantum(self, q_input: torch.Tensor) -> torch.Tensor:
        """
        Run quantum circuit on batch.

        PennyLane's default.qubit simulator always runs on CPU.
        We explicitly move inputs/weights to CPU before the circuit
        and move results back to model device (MPS/CUDA/CPU) after.

        Args:
            q_input: (B, 2^n_qubits) pre-quantum features on model device

        Returns:
            (B, n_qubits) expectation values on model device
        """
        if self._quantum_layer is None or not PENNYLANE_AVAILABLE:
            return self._classical_fallback(q_input)

        batch_size = q_input.shape[0]

        # Move to CPU for PennyLane (simulator is CPU-only)
        # Also detach from MPS graph — PennyLane needs plain CPU tensors
        q_input_cpu   = q_input.detach().cpu().float()   # float32
        q_weights_cpu = self.q_weights.detach().cpu().float()

        outputs = []
        for i in range(batch_size):
            result = self._quantum_layer(
                q_input_cpu[i],    # (2^n_qubits,) float32 on CPU
                q_weights_cpu,     # (n_layers, n_qubits, 3) float32 on CPU
            )
            # PennyLane returns float64 tensors — cast to float32 immediately
            outputs.append(torch.stack(result).float())   # (n_qubits,) float32

        # Stack (float32 on CPU) → move to model device (MPS / CUDA / CPU)
        stacked = torch.stack(outputs)            # (B, n_qubits) float32
        return stacked.to(device=self._device, dtype=torch.float32)

    def _classical_fallback(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classical approximation when quantum circuit fails.
        Applies tanh to keep outputs in [-1, 1] matching qubit expectations.

        Args:
            x: (B, 2^n_qubits) or (2^n_qubits,)

        Returns:
            (B, n_qubits) classical approximation
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Average pool 2^n_qubits → n_qubits
        # e.g. 8 features → 3 qubit outputs (for n_qubits=3)
        B       = x.shape[0]
        out_dim = self.n_qubits

        # Reshape and mean-pool if q_dim > n_qubits
        if self._q_dim >= out_dim:
            chunk = self._q_dim // out_dim
            # Trim to multiple of out_dim
            x_trim = x[:, : out_dim * chunk]
            out    = x_trim.reshape(B, out_dim, chunk).mean(dim=-1)
        else:
            # Pad if q_dim < n_qubits (unusual)
            pad = out_dim - self._q_dim
            out = torch.cat([x, torch.zeros(B, pad, device=x.device)], dim=1)

        return torch.tanh(out)   # keep in [-1, 1]

    # ------------------------------------------------------------------
    # Backbone helpers (identical to EfficientNetModel)
    # ------------------------------------------------------------------

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("[QuantumModel] Using CUDA")
        elif (hasattr(torch.backends, "mps") and
              torch.backends.mps.is_available()):
            device = torch.device("mps")
            logger.info("[QuantumModel] Using Apple MPS")
        else:
            device = torch.device("cpu")
            logger.info("[QuantumModel] Using CPU")
        return device

    @staticmethod
    def _build_backbone(pretrained: bool) -> Tuple[nn.Module, int]:
        feature_dim = 1792
        try:
            from efficientnet_pytorch import EfficientNet
            if pretrained:
                net = EfficientNet.from_pretrained("efficientnet-b4")
                logger.info("Quantum model: loaded efficientnet_pytorch backbone (pretrained)")
            else:
                net = EfficientNet.from_name("efficientnet-b4")
                logger.info("Quantum model: loaded efficientnet_pytorch backbone (random init)")
            net._fc = nn.Identity()
            return nn.Sequential(net), feature_dim

        except ImportError:
            pass

        import torchvision.models as M
        weights  = M.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        net      = M.efficientnet_b4(weights=weights)
        backbone = nn.Sequential(
            net.features,
            net.avgpool,
            nn.Flatten(start_dim=1),
        )
        logger.info("Quantum model: loaded torchvision EfficientNet-B4 backbone")
        return backbone, feature_dim

    def _apply_finetuning(self, mode: str) -> None:
        if mode == "ftall":
            for p in self.backbone.parameters():
                p.requires_grad = True
        elif mode == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif mode in ("ft1", "ft2"):
            for p in self.backbone.parameters():
                p.requires_grad = False
            n = 1 if mode == "ft1" else 2
            for child in list(self.backbone.children())[-n:]:
                for p in child.parameters():
                    p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True

        logger.info(f"Quantum model finetuning: {mode}")
