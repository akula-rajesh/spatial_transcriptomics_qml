"""
EfficientNet-B4 + Quantum Head — Redesigned for Gene Expression Prediction

Key design decisions:
  n_qubits    = 16   2^16 Hilbert space, 16 PauliZ outputs
  n_layers    = 4    safe with re-uploading, faster simulation
  expansion   = 256  classical expansion before gene heads
                     solves 16→250 underdetermined mapping

Information flow:
  Image (B,3,H,W)
    → EfficientNet-B4               (B, 1792)
    → ClassicalToQuantumBridge      (B, n_layers, n_qubits) = (B, 4, 16)
    → QuantumHead (re-uploading)    (B, n_qubits) = (B, 16)
    → QuantumExpansionHead          (B, expansion_dim) = (B, 256)
    → main_head Linear(256→250)     (B, 250)
    → aux_head  Linear(256→5966)    (B, 5966)

Why expansion head solves the bottleneck:
  Without: Linear(16→250) — rank ≤ 16, 234 output dims are redundant
  With   : Linear(16→256) + GELU + Linear(256→250)
           Rank still ≤ 16 mathematically BUT non-linear expansion
           allows model to learn independent gene predictions via
           the GELU non-linearity between the two linear layers.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# ── PennyLane ─────────────────────────────────────────────────────────
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
    logger.info("PennyLane available")
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning(
        "PennyLane not installed — quantum head in fallback mode. "
        "Run: pip install pennylane"
    )


# ======================================================================
# Quantum Head — Data Re-uploading Circuit
# ======================================================================

class QuantumHead(nn.Module):
    """
    PennyLane variational quantum circuit with data re-uploading.

    Recommended: n_qubits=16, n_layers=4

    Input : (B, n_layers, n_qubits)  — fresh angles per layer
    Output: (B, n_qubits)            — Pauli-Z expectation values [-1,1]

    Weight shape: (n_layers, n_qubits, 3)
      3 = (phi, theta, omega) for Rot gate — ALWAYS 3, PennyLane contract
      n_layers=4, n_qubits=16  → 4×16×3 = 192 quantum params
    """

    def __init__(
            self,
            n_qubits : int = 16,
            n_layers : int = 4,
            q_device : str = "default.qubit",
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        if PENNYLANE_AVAILABLE:
            self._circuit = self._build_circuit(q_device)
        else:
            self._circuit = None

        # Small random init — prevents barren plateau at start
        # Explicitly float32: np.random.uniform returns float64 by default,
        # which causes dtype errors on MPS.
        self.weights = nn.Parameter(
            torch.tensor(
                np.random.uniform(
                    low  = -0.1,
                    high =  0.1,
                    size = (n_layers, n_qubits, 3),   # e.g. 4×16×3 = 192
                ),
                dtype=torch.float32,
            )
        )

        logger.info(
            f"QuantumHead: {n_qubits} qubits × {n_layers} layers "
            f"| weights {list(self.weights.shape)} "
            f"= {self.weights.numel()} params "
            f"| circuit: {'active' if self._circuit else 'fallback'}"
        )

    def _build_circuit(self, q_device_str: str):
        """Build re-uploading QNode."""
        try:
            dev      = qml.device(q_device_str, wires=self.n_qubits)
            n_qubits = self.n_qubits
            n_layers = self.n_layers

            # parameter-shift: device-agnostic — works on CPU, MPS, CUDA
            # backprop requires CUDA-compiled PyTorch and fails on MPS
            @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
            def circuit(inputs, weights):
                """
                Args:
                    inputs  : (n_layers, n_qubits)    — re-uploaded angles (float32, CPU)
                    weights : (n_layers, n_qubits, 3) — Rot params         (float32, CPU)
                              3 = (phi, theta, omega) ALWAYS 3
                Returns:
                    List[tensor] — n_qubits PauliZ expectations
                """
                for layer_idx in range(n_layers):
                    # Re-upload fresh angles at every layer
                    qml.AngleEmbedding(
                        inputs[layer_idx],
                        wires=range(n_qubits),
                        rotation="Y",
                    )
                    # Parameterized rotation per qubit
                    for q in range(n_qubits):
                        qml.Rot(
                            weights[layer_idx, q, 0],
                            weights[layer_idx, q, 1],
                            weights[layer_idx, q, 2],
                            wires=q,
                        )
                    # Ring entanglement
                    for q in range(n_qubits):
                        qml.CNOT(wires=[q, (q + 1) % n_qubits])

                return [
                    qml.expval(qml.PauliZ(q))
                    for q in range(n_qubits)
                ]

            logger.info(
                f"Circuit compiled: {n_qubits}q × {n_layers}L "
                f"| re-uploading | ring CNOT | diff=parameter-shift"
            )
            return circuit

        except Exception as e:
            logger.warning(f"Circuit build failed: {e} — using fallback")
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_layers, n_qubits) angle-encoded features on model device

        Returns:
            (B, n_qubits) PauliZ expectations in [-1, 1] on model device
        """
        if self._circuit is None or not PENNYLANE_AVAILABLE:
            return self._classical_fallback(x)

        B      = x.shape[0]
        device = x.device

        # PennyLane default.qubit runs on CPU — move to CPU + cast to float32
        # MPS does not support float64; PennyLane returns float64 by default
        x_cpu       = x.detach().cpu().float()          # (B, n_layers, n_qubits) float32
        weights_cpu = self.weights.detach().cpu().float() # (n_layers, n_qubits, 3) float32

        outputs = []
        for i in range(B):
            result = self._circuit(
                x_cpu[i],     # (n_layers, n_qubits) float32 on CPU
                weights_cpu,  # (n_layers, n_qubits, 3) float32 on CPU
            )
            # PennyLane returns float64 — cast to float32 immediately
            outputs.append(torch.stack(result).float())  # (n_qubits,) float32

        # Stack and move back to model device (MPS / CUDA / CPU)
        return torch.stack(outputs).to(device=device, dtype=torch.float32)

    def _classical_fallback(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return torch.tanh(x.mean(dim=1))

    def circuit_summary(self) -> str:
        if not PENNYLANE_AVAILABLE or self._circuit is None:
            return "Circuit: classical fallback (PennyLane unavailable)"
        try:
            d_inp = np.zeros((self.n_layers, self.n_qubits))
            d_w   = np.zeros((self.n_layers, self.n_qubits, 3))
            return qml.draw(self._circuit)(d_inp, d_w)
        except Exception:
            return (
                f"{self.n_layers} × "
                f"[AngleEmbedding(Ry) + Rot×{self.n_qubits} + CNOTring] "
                f"+ PauliZ×{self.n_qubits}"
            )


# ======================================================================
# Classical → Quantum Bridge  (1792 → n_layers × n_qubits)
# ======================================================================

class ClassicalToQuantumBridge(nn.Module):
    """
    Compresses EfficientNet features into quantum circuit angles.

    For n_qubits=16, n_layers=4:
      Input  : (B, 1792)
      Output : (B, 4, 16)   — 4 sets of 16 angles

    Compression: 1792 → 64 = 28:1
    Staged to preserve structure:
      1792 → 512 → 256 → 64 (n_layers × n_qubits)
    """

    def __init__(
            self,
            feature_dim : int   = 1792,
            n_qubits    : int   = 16,
            n_layers    : int   = 4,
            scale       : float = float(np.pi),
            dropout     : float = 0.1,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.scale    = scale
        output_dim    = n_qubits * n_layers   # 16 × 4 = 64

        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

        logger.info(
            f"Bridge: {feature_dim}→512→256→{output_dim} "
            f"({n_layers}×{n_qubits}) "
            f"| compression {feature_dim}:{output_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            (B, n_layers, n_qubits) angles in [-π, π]
        """
        B   = x.shape[0]
        out = torch.tanh(self.net(x)) * self.scale
        return out.reshape(B, self.n_layers, self.n_qubits)


# ======================================================================
# Quantum Expansion Head  (KEY FIX — solves 16→250 bottleneck)
# ======================================================================

class QuantumExpansionHead(nn.Module):
    """
    Classical expansion between quantum output and gene prediction heads.

    PROBLEM this solves:
      Linear(n_qubits=16 → gene_filter=250) has rank ≤ 16.
      Only 16 independent values can influence 250 outputs.
      234 of the 250 gene predictions would be linearly dependent.

    SOLUTION:
      Add non-linear expansion: 16 → expansion_dim → gene predictions
      GELU non-linearity breaks the rank constraint:
        Each of the 250 output neurons can be a different
        non-linear combination of the 16 quantum measurements.

    Architecture:
      Q output (B, n_qubits=16)
        → Linear(16 → expansion_dim=256)   expand
        → LayerNorm(256)
        → GELU                              non-linearity breaks rank cap
        → Dropout(0.1)
        → (B, 256)                          ready for gene heads

    Why expansion_dim=256:
      256 > 250 (gene_filter) — overcomplete representation
      256 is power of 2 — efficient matrix multiplication
      16 → 256 → 250: natural hourglass through quantum bottleneck
    """

    def __init__(
            self,
            n_qubits      : int   = 16,
            expansion_dim : int   = 256,
            dropout       : float = 0.1,
    ):
        super().__init__()

        self.n_qubits      = n_qubits
        self.expansion_dim = expansion_dim

        self.net = nn.Sequential(
            nn.Linear(n_qubits, expansion_dim),
            nn.LayerNorm(expansion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        logger.info(
            f"QuantumExpansionHead: {n_qubits} → {expansion_dim} "
            f"| solves rank bottleneck for gene prediction heads"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_qubits) PauliZ expectations

        Returns:
            (B, expansion_dim) expanded representation
        """
        return self.net(x)


# ======================================================================
# Full Hybrid Model — EfficientNet + 16q Quantum Head
# ======================================================================

class EfficientNetQuantumHead(nn.Module):
    """
    EfficientNet-B4 + 16-Qubit Quantum Head with Expansion.

    Recommended defaults:
      n_qubits      = 16    2^16 Hilbert space
      n_layers      = 4     re-uploading, balanced depth/speed
      expansion_dim = 256   classical expansion post-quantum

    Full pipeline:
      (B,3,H,W)
        → EfficientNet-B4              (B, 1792)
        → ClassicalToQuantumBridge     (B, 4, 16)   angles in [-π,π]
        → QuantumHead (re-uploading)   (B, 16)      PauliZ in [-1,1]
        → QuantumExpansionHead         (B, 256)     non-linear expand
        → main_head Linear(256→250)    (B, 250)     gene predictions
        → aux_head  Linear(256→5966)   (B, 5966)    aux predictions

    Parameter budget:
      Bridge          : ~1,058,946
      Quantum weights : 4×16×3 = 192
      Expansion head  : 16×256 = 4,096
      Main head       : 256×250 = 64,000
      Aux head        : 256×5966 = 1,527,296
      Total (no bb)   : ~2,654,530

    Config keys (pipeline_config.yaml):
      model:
        type          : efficientnet_quantum_head
        n_qubits      : 16
        n_layers      : 4
        expansion_dim : 256
        gene_filter   : 250
        aux_ratio     : 1.0
        pretrained    : true
        finetuning    : ftall
        q_device      : default.qubit
        bridge_dropout: 0.1
        head_dropout  : 0.1
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        model_cfg = config.get("model", {})

        def _get(key: str, default):
            return config.get(key, model_cfg.get(key, default))

        self.n_qubits      = int(_get("n_qubits",       16))
        self.n_layers      = int(_get("n_layers",         4))
        self.expansion_dim = int(_get("expansion_dim",  256))
        self.gene_filter   = int(_get("gene_filter",    250))
        self.aux_ratio     = float(_get("aux_ratio",     1.0))
        self.q_device      = str(_get("q_device",  "default.qubit"))
        pretrained         = bool(_get("pretrained",    True))
        finetuning         = str(_get("finetuning",  "ftall"))
        bridge_dropout     = float(_get("bridge_dropout", 0.1))
        head_dropout       = float(_get("head_dropout",   0.1))

        total_genes = (
                config.get("total_genes") or
                model_cfg.get("total_genes")
        )
        if total_genes is None:
            logger.warning(
                "total_genes not in config — aux_nums=0. "
                "Trainer will call set_aux_head() after dataset build."
            )
            self.aux_nums = 0
        else:
            self.aux_nums = int(
                (int(total_genes) - self.gene_filter) * self.aux_ratio
            )

        # ── Device ───────────────────────────────────────────────────
        self._device = self._select_device()

        # ── Stage 1: Backbone ─────────────────────────────────────────
        self.backbone, self._feature_dim = self._build_backbone(pretrained)
        self._apply_finetuning(finetuning)

        # ── Stage 2: Bridge 1792 → (n_layers, n_qubits) ──────────────
        self.bridge = ClassicalToQuantumBridge(
            feature_dim = self._feature_dim,
            n_qubits    = self.n_qubits,
            n_layers    = self.n_layers,
            scale       = float(np.pi),
            dropout     = bridge_dropout,
        )

        # ── Stage 3: Quantum circuit ──────────────────────────────────
        self.quantum_head = QuantumHead(
            n_qubits = self.n_qubits,
            n_layers = self.n_layers,
            q_device = self.q_device,
        )

        # ── Stage 4: Expansion head n_qubits → expansion_dim ─────────
        self.expansion = QuantumExpansionHead(
            n_qubits      = self.n_qubits,
            expansion_dim = self.expansion_dim,
            dropout       = head_dropout,
        )

        # ── Stage 5: Gene prediction heads ───────────────────────────
        # Input is expansion_dim (256), NOT n_qubits (16)
        self.main_head = nn.Linear(self.expansion_dim, self.gene_filter)
        self.aux_head: Optional[nn.Linear] = (
            nn.Linear(self.expansion_dim, self.aux_nums)
            if self.aux_nums > 0 else None
        )

        # ── Move everything to device ─────────────────────────────────
        self.to(self._device)

        self._log_summary(finetuning)

    def _log_summary(self, finetuning: str) -> None:
        q_params  = self.quantum_head.weights.numel()
        br_params = sum(p.numel() for p in self.bridge.parameters())
        ex_params = sum(p.numel() for p in self.expansion.parameters())
        mh_params = sum(p.numel() for p in self.main_head.parameters())
        ah_params = (
            sum(p.numel() for p in self.aux_head.parameters())
            if self.aux_head else 0
        )

        logger.info(
            f"\n{'═'*60}\n"
            f"  EfficientNet + {self.n_qubits}-Qubit Quantum Head\n"
            f"{'─'*60}\n"
            f"  Stage 1 Backbone   : EfficientNet-B4 → {self._feature_dim}\n"
            f"           Finetuning: {finetuning}\n"
            f"  Stage 2 Bridge     : {self._feature_dim}→512→256"
            f"→{self.n_layers}×{self.n_qubits}={self.n_layers*self.n_qubits}\n"
            f"           Params    : {br_params:,}\n"
            f"  Stage 3 Q-Circuit  : {self.n_qubits}q × {self.n_layers}L "
            f"re-uploading | ring CNOT\n"
            f"           Params    : {q_params} "
            f"= {self.n_layers}×{self.n_qubits}×3\n"
            f"           Output    : {self.n_qubits} PauliZ values [-1,1]\n"
            f"  Stage 4 Expansion  : {self.n_qubits}→{self.expansion_dim} "
            f"(solves rank bottleneck)\n"
            f"           Params    : {ex_params:,}\n"
            f"  Stage 5 Main head  : {self.expansion_dim}→{self.gene_filter}\n"
            f"           Aux head  : {self.expansion_dim}→{self.aux_nums}\n"
            f"           Params    : {mh_params + ah_params:,}\n"
            f"{'─'*60}\n"
            f"  Total (excl. bb)   : "
            f"{br_params+q_params+ex_params+mh_params+ah_params:,}\n"
            f"  Device             : {self._device}\n"
            f"{'═'*60}"
        )

    # ------------------------------------------------------------------
    # Properties & Public API
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    def set_aux_head(self, aux_nums: int) -> None:
        """Install aux head after dataset is known."""
        if self.aux_nums == aux_nums and self.aux_head is not None:
            return
        self.aux_nums = aux_nums
        self.aux_head = nn.Linear(
            self.expansion_dim, aux_nums
        ).to(self._device)
        logger.info(
            f"Aux head: Linear({self.expansion_dim}→{aux_nums}) ✓"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
            self,
            x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, 3, H, W) on model.device

        Returns:
            main_pred : (B, gene_filter=250)
            aux_pred  : (B, aux_nums=5966) or None
        """
        # Stage 1
        features  = self.backbone(x)                  # (B, 1792) float32

        # Stage 2
        angles    = self.bridge(features).float()     # (B, n_layers, n_qubits) float32

        # Stage 3 — QuantumHead handles CPU transfer + float32 cast internally
        q_out     = self.quantum_head(angles)         # (B, n_qubits) float32

        # Stage 4 — EXPANSION (key fix for 16→250 rank bottleneck)
        expanded  = self.expansion(q_out)             # (B, expansion_dim=256) float32

        # Stage 5
        main_pred = self.main_head(expanded)          # (B, 250)
        aux_pred  = (
            self.aux_head(expanded)
            if self.aux_head is not None else None
        )

        return main_pred, aux_pred

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            d = torch.device("cuda")
            logger.info("[EfficientNetQuantumHead] CUDA")
        elif (hasattr(torch.backends, "mps") and
              torch.backends.mps.is_available()):
            d = torch.device("mps")
            logger.info("[EfficientNetQuantumHead] Apple MPS")
        else:
            d = torch.device("cpu")
            logger.info("[EfficientNetQuantumHead] CPU")
        return d

    @staticmethod
    def _build_backbone(pretrained: bool) -> Tuple[nn.Module, int]:
        feature_dim = 1792
        try:
            from efficientnet_pytorch import EfficientNet
            if pretrained:
                net = EfficientNet.from_pretrained("efficientnet-b4")
                logger.info("Backbone: efficientnet_pytorch (pretrained)")
            else:
                net = EfficientNet.from_name("efficientnet-b4")
                logger.info("Backbone: efficientnet_pytorch (random init)")
            net._fc = nn.Identity()
            return nn.Sequential(net), feature_dim
        except ImportError:
            pass

        import torchvision.models as M
        tv_weights = M.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        net        = M.efficientnet_b4(weights=tv_weights)
        backbone   = nn.Sequential(
            net.features,
            net.avgpool,
            nn.Flatten(start_dim=1),
        )
        logger.info("Backbone: torchvision EfficientNet-B4")
        return backbone, feature_dim

    def _apply_finetuning(self, mode: str) -> None:
        configs = {
            "ftall" : (True,  0),
            "frozen": (False, 0),
            "ft1"   : (False, 1),
            "ft2"   : (False, 2),
        }
        trainable, n_unfreeze = configs.get(mode, (True, 0))
        for p in self.backbone.parameters():
            p.requires_grad = trainable
        if n_unfreeze:
            for child in list(self.backbone.children())[-n_unfreeze:]:
                for p in child.parameters():
                    p.requires_grad = True
        n = sum(p.numel() for p in self.backbone.parameters()
                if p.requires_grad)
        logger.info(f"Finetuning: {mode} | {n:,} backbone params trainable")


__all__ = ["EfficientNetQuantumHead"]
