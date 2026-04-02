#!/usr/bin/env python3
"""
Pipeline Diagnostic Script for Spatial Transcriptomics ML Pipeline.

Checks all critical components before running full training:
  1. Environment & dependencies
  2. Data directory structure
  3. Metadata files (gene.pkl, subtype.pkl, mean_expression.npy)
  4. NPZ spot files (count, pixel, patient, section, index)
  5. Image loading (RGB, uint8, correct shape)
  6. Image normalization statistics (detects grayscale collapse)
  7. Gene count processing (log1p, shape, range)
  8. DataLoader batch structure (8-tuple)
  9. Model forward pass (main + aux output shapes)
  10. Loss computation (MSE)
  11. Full mini-training dry-run (3 epochs)
  12. Configuration validation

Usage:
    python scripts/diagnose_pipeline.py --config config/pipeline_config.yaml
    python scripts/diagnose_pipeline.py --config config/pipeline_config.yaml --verbose
    python scripts/diagnose_pipeline.py --config config/pipeline_config.yaml --fix
    python scripts/diagnose_pipeline.py --config config/pipeline_config.yaml --skip-training
"""

import sys
import argparse
import logging
import traceback
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# ── Add project root to path ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Result collector
# ======================================================================

class DiagnosticResult:
    """Accumulates pass/warn/fail results across all checks."""

    PASS = "✅ PASS"
    WARN = "⚠️  WARN"
    FAIL = "❌ FAIL"

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def record(self, check: str, status: str, detail: str = "") -> None:
        self.results.append({
            "check":  check,
            "status": status,
            "detail": detail,
        })
        icon = status.split()[0]
        msg  = f"[{icon}] {check}"
        if detail:
            msg += f"\n       {detail}"
        print(msg)

    def passed(self, check: str, detail: str = "") -> None:
        self.record(check, self.PASS, detail)

    def warned(self, check: str, detail: str = "") -> None:
        self.record(check, self.WARN, detail)

    def failed(self, check: str, detail: str = "") -> None:
        self.record(check, self.FAIL, detail)

    def summary(self) -> Tuple[int, int]:
        """Print summary table and return (n_fails, n_warns)."""
        total  = len(self.results)
        passes = sum(1 for r in self.results if r["status"] == self.PASS)
        warns  = sum(1 for r in self.results if r["status"] == self.WARN)
        fails  = sum(1 for r in self.results if r["status"] == self.FAIL)

        print("\n" + "=" * 70)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 70)
        print(f"  Total checks : {total}")
        print(f"  Passed       : {passes}")
        print(f"  Warnings     : {warns}")
        print(f"  Failed       : {fails}")
        print("=" * 70)

        if fails > 0:
            print("\n❌ FAILED CHECKS:")
            for r in self.results:
                if r["status"] == self.FAIL:
                    print(f"   • {r['check']}")
                    if r["detail"]:
                        print(f"     {r['detail']}")

        if warns > 0:
            print("\n⚠️  WARNINGS:")
            for r in self.results:
                if r["status"] == self.WARN:
                    print(f"   • {r['check']}")
                    if r["detail"]:
                        print(f"     {r['detail']}")

        print()
        if fails == 0 and warns == 0:
            print("🎉 ALL CHECKS PASSED — pipeline is ready for training!")
        elif fails == 0:
            print("✅ No failures. Review warnings before training.")
        else:
            print("🚨 Fix failures before running training.")
        print("=" * 70 + "\n")

        return fails, warns


# ======================================================================
# Pipeline Diagnostic — ALL methods inside the class
# ======================================================================

class PipelineDiagnostic:
    """Runs all pipeline checks."""

    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        self.config  = config
        self.verbose = verbose
        self.result  = DiagnosticResult()

        # Resolve paths from config
        data = config.get("data", {})
        prep = config.get("preprocessing", {})

        self.train_counts = Path(data.get("train_counts_dir", "data/train/counts/"))
        self.train_images = Path(data.get("train_images_dir", "data/train/images/"))
        self.test_counts  = Path(data.get("test_counts_dir",  "data/test/counts/"))
        self.test_images  = Path(data.get("test_images_dir",  "data/test/images/"))
        self.filter_root  = Path(data.get("count_filtered_dir",
                                          "data/processed/count_filtered/"))
        self.test_patient = data.get("test_patient", "BC23450")
        self.window       = prep.get("window_size", 299)
        self.gene_filter  = config.get("model", {}).get("gene_filter", 250)
        self.aux_ratio    = config.get("model", {}).get("aux_ratio",   1.0)

    # ------------------------------------------------------------------
    # CHECK 1 — Environment
    # ------------------------------------------------------------------

    def check_environment(self) -> None:
        print("\n── CHECK 1: Environment & Dependencies ──────────────────────────")

        pv = sys.version_info
        if pv >= (3, 8):
            self.result.passed("Python version",
                               f"Python {pv.major}.{pv.minor}.{pv.micro}")
        else:
            self.result.failed("Python version",
                               f"Python {pv.major}.{pv.minor} — need ≥ 3.8")

        packages = {
            "torch":       ("torch",       "1.12.0"),
            "torchvision": ("torchvision", "0.13.0"),
            "numpy":       ("numpy",       "1.21.0"),
            "PIL":         ("PIL",         "8.0.0"),
            "sklearn":     ("sklearn",     "1.0.0"),
            "yaml":        ("yaml",        "5.0"),
        }

        for display_name, (module_name, _) in packages.items():
            try:
                mod = __import__(module_name)
                ver = getattr(mod, "__version__", "unknown")
                self.result.passed(f"Package: {display_name}", f"v{ver}")
            except ImportError:
                self.result.failed(f"Package: {display_name}",
                                   f"Not installed. Run: pip install {module_name}")

        try:
            import efficientnet_pytorch
            self.result.passed("Package: efficientnet_pytorch",
                               f"v{efficientnet_pytorch.__version__}")
        except ImportError:
            self.result.warned("Package: efficientnet_pytorch",
                               "Not installed — will use torchvision fallback. "
                               "Install with: pip install efficientnet_pytorch")

        import torch
        if torch.cuda.is_available():
            self.result.passed(
                "GPU: CUDA",
                f"Device: {torch.cuda.get_device_name(0)}, "
                f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.result.passed("GPU: Apple MPS",
                               "Metal Performance Shaders available")
        else:
            self.result.warned("GPU: None",
                               "Training will be slow on CPU.")

    # ------------------------------------------------------------------
    # CHECK 2 — Directory structure
    # ------------------------------------------------------------------

    def check_directories(self) -> None:
        print("\n── CHECK 2: Directory Structure ─────────────────────────────────")

        required = {
            "Train counts":        self.train_counts,
            "Train images":        self.train_images,
            "Test counts":         self.test_counts,
            "Test images":         self.test_images,
            "Count filtered root": self.filter_root,
        }

        for name, path in required.items():
            if path.exists():
                n_files = len(list(path.rglob("*")))
                self.result.passed(name, f"{path}  ({n_files} files/dirs)")
            else:
                self.result.failed(name,
                                   f"Missing: {path}  "
                                   "Run data processing pipeline first.")

    # ------------------------------------------------------------------
    # CHECK 3 — Metadata files
    # ------------------------------------------------------------------

    def check_metadata(self) -> None:
        print("\n── CHECK 3: Metadata Files ──────────────────────────────────────")
        import pickle
        import numpy as np

        gene_pkl = self.filter_root / "gene.pkl"
        if gene_pkl.exists():
            with open(gene_pkl, "rb") as f:
                genes = pickle.load(f)
            self.result.passed("gene.pkl",
                               f"Contains {len(genes)} genes. "
                               f"First 3: {genes[:3]}")
            self._gene_count = len(genes)
        else:
            self.result.failed("gene.pkl", f"Missing: {gene_pkl}")
            self._gene_count = 0

        sub_pkl = self.filter_root / "subtype.pkl"
        if sub_pkl.exists():
            with open(sub_pkl, "rb") as f:
                subtype = pickle.load(f)
            self.result.passed("subtype.pkl",
                               f"{len(subtype)} patients, "
                               f"subtypes: {sorted(set(subtype.values()))}")
            self._subtype = subtype
        else:
            self.result.failed("subtype.pkl", f"Missing: {sub_pkl}")
            self._subtype = {}

        mean_npy = self.filter_root / "mean_expression.npy"
        if mean_npy.exists():
            mean_expr = np.load(mean_npy)
            self.result.passed("mean_expression.npy",
                               f"Shape: {mean_expr.shape}, "
                               f"Range: [{mean_expr.min():.4f}, "
                               f"{mean_expr.max():.4f}]")
        else:
            self.result.failed("mean_expression.npy", f"Missing: {mean_npy}")

    # ------------------------------------------------------------------
    # CHECK 4 — NPZ spot files
    # ------------------------------------------------------------------

    def check_npz_files(self) -> None:
        print("\n── CHECK 4: NPZ Spot Files ──────────────────────────────────────")
        import numpy as np

        for split, root in [("train", self.train_counts),
                            ("test",  self.test_counts)]:
            npz_files = sorted(root.rglob("*.npz"))
            if not npz_files:
                self.result.failed(f"NPZ files ({split})",
                                   f"No .npz files found under {root}")
                continue

            self.result.passed(f"NPZ files ({split})",
                               f"{len(npz_files)} files found")

            sample = npz_files[0]
            try:
                npz          = np.load(sample)
                keys         = set(npz.keys())
                required_keys = {"count", "pixel", "patient", "section", "index"}
                missing_keys  = required_keys - keys

                if missing_keys:
                    self.result.failed(f"NPZ structure ({split})",
                                       f"Missing keys: {missing_keys} "
                                       f"in {sample.name}")
                else:
                    count   = npz["count"]
                    pixel   = npz["pixel"]
                    patient = str(npz["patient"][0])
                    section = str(npz["section"][0])
                    coord   = npz["index"]
                    self.result.passed(f"NPZ structure ({split})",
                                       f"count={count.shape}, pixel={pixel}, "
                                       f"patient={patient}, section={section}, "
                                       f"coord={coord}")

                    if count.min() < 0:
                        self.result.failed(f"NPZ count values ({split})",
                                           f"Negative counts! min={count.min()}")
                    elif count.max() == 0:
                        self.result.failed(f"NPZ count values ({split})",
                                           "All zero counts — file may be corrupted.")
                    else:
                        self.result.passed(f"NPZ count values ({split})",
                                           f"Range: [{count.min():.0f}, "
                                           f"{count.max():.0f}], "
                                           f"Sum: {count.sum():.0f}")
            except Exception as e:
                self.result.failed(f"NPZ read ({split})",
                                   f"Error reading {sample}: {e}")

    # ------------------------------------------------------------------
    # CHECK 5 — Image loading  (CRITICAL)
    # ------------------------------------------------------------------

    def check_image_loading(self) -> None:
        print("\n── CHECK 5: Image Loading (CRITICAL) ────────────────────────────")
        import numpy as np
        from PIL import Image
        import torchvision.transforms as T
        import pickle

        sub_pkl = self.filter_root / "subtype.pkl"
        if not sub_pkl.exists():
            self.result.failed("Image loading (subtype.pkl missing)",
                               "Cannot proceed without subtype.pkl")
            return

        with open(sub_pkl, "rb") as f:
            subtype = pickle.load(f)

        img_path = self._find_sample_image(subtype)
        if img_path is None:
            self.result.failed(
                "Image path resolution",
                f"Could not find any image under {self.train_images}. "
                "Expected: {img_root}/{subtype}/{patient}/{window}/{sec}_{x}_{y}.jpg"
            )
            return

        self.result.passed("Image path resolution", f"Found sample: {img_path}")

        try:
            img_pil = Image.open(img_path)
        except Exception as e:
            self.result.failed("PIL open", str(e))
            return

        raw_mode = img_pil.mode
        if raw_mode != "RGB":
            self.result.warned(f"PIL image mode: {raw_mode}",
                               "Will be converted via .convert('RGB').")
        else:
            self.result.passed("PIL image mode", "RGB ✓")

        img_rgb = img_pil.convert("RGB")
        w, h    = img_rgb.size
        self.result.passed("PIL RGB size", f"Width={w}, Height={h}")

        if w < 224 or h < 224:
            self.result.failed("PIL image size",
                               f"Too small: {w}×{h}. Expected ≥ 224×224.")

        arr = np.array(img_rgb)
        self.result.passed("NumPy array shape",
                           f"{arr.shape} (expected (H, W, 3))")

        if arr.shape[-1] != 3:
            self.result.failed("NumPy channels",
                               f"Expected 3 channels, got {arr.shape[-1]}")
        if arr.dtype != np.uint8:
            self.result.warned("NumPy dtype",
                               f"Expected uint8, got {arr.dtype}.")
        else:
            self.result.passed("NumPy dtype", "uint8 ✓")

        if arr.max() <= 1.0:
            self.result.failed(
                "Pixel range",
                f"max={arr.max()} ≤ 1.0 — images appear pre-normalized! "
                "ToTensor() will divide again → values ~0.004. "
                "Ensure images are saved as uint8 JPEGs."
            )
        elif arr.max() > 255:
            self.result.failed("Pixel range",
                               f"max={arr.max()} > 255 — out of uint8 range.")
        else:
            self.result.passed("Pixel range",
                               f"[{arr.min()}, {arr.max()}] — uint8 [0,255] ✓")

        r_mean = arr[:, :, 0].mean()
        g_mean = arr[:, :, 1].mean()
        b_mean = arr[:, :, 2].mean()
        r_std  = arr[:, :, 0].std()
        g_std  = arr[:, :, 1].std()
        b_std  = arr[:, :, 2].std()

        print(f"\n       Raw PIL stats (H&E tissue: mean≈140-180, std≈30-60):")
        print(f"         R: mean={r_mean:.1f}, std={r_std:.1f}")
        print(f"         G: mean={g_mean:.1f}, std={g_std:.1f}")
        print(f"         B: mean={b_mean:.1f}, std={b_std:.1f}")

        if abs(r_mean - g_mean) < 0.1 and abs(g_mean - b_mean) < 0.1:
            self.result.failed(
                "Color channel diversity",
                f"All channels identical mean={r_mean:.2f} — GRAYSCALE! "
                "H&E staining info is LOST."
            )
        else:
            self.result.passed("Color channel diversity",
                               f"R={r_mean:.1f}, G={g_mean:.1f}, "
                               f"B={b_mean:.1f} — channels differ ✓")

        if r_mean < 20 or g_mean < 20 or b_mean < 20:
            self.result.warned("Pixel brightness",
                               f"Very dark: R={r_mean:.1f}, G={g_mean:.1f}, "
                               f"B={b_mean:.1f}. Check stain normalization.")

        to_tensor = T.ToTensor()
        tensor    = to_tensor(img_rgb)

        t_min    = tensor.min().item()
        t_max    = tensor.max().item()
        t_mean_r = tensor[0].mean().item()
        t_mean_g = tensor[1].mean().item()
        t_mean_b = tensor[2].mean().item()
        t_std_r  = tensor[0].std().item()
        t_std_g  = tensor[1].std().item()
        t_std_b  = tensor[2].std().item()

        print(f"\n       After ToTensor() (expected range [0,1], mean≈0.5-0.8):")
        print(f"         Range:  [{t_min:.4f}, {t_max:.4f}]")
        print(f"         R: mean={t_mean_r:.4f}, std={t_std_r:.4f}")
        print(f"         G: mean={t_mean_g:.4f}, std={t_std_g:.4f}")
        print(f"         B: mean={t_mean_b:.4f}, std={t_std_b:.4f}")

        if t_max > 1.01:
            self.result.failed("ToTensor range",
                               f"Max={t_max:.4f} > 1.0 — input may not be uint8.")
        elif t_max < 0.1:
            self.result.failed("ToTensor range",
                               f"Max={t_max:.6f} ≈ 0 — pre-normalization suspected.")
        else:
            self.result.passed("ToTensor range",
                               f"[{t_min:.4f}, {t_max:.4f}] — correct [0,1] ✓")

        for ch, m, s in [("R", t_mean_r, t_std_r),
                         ("G", t_mean_g, t_std_g),
                         ("B", t_mean_b, t_std_b)]:
            if m < 0.05:
                self.result.failed(f"ToTensor channel {ch} mean",
                                   f"mean={m:.6f} — critically low. "
                                   "Double-normalization suspected.")
            elif s < 0.01:
                self.result.failed(f"ToTensor channel {ch} std",
                                   f"std={s:.8f} ≈ 0 — all pixels identical.")
            else:
                self.result.passed(f"ToTensor channel {ch}",
                                   f"mean={m:.4f}, std={s:.4f} ✓")

    # ------------------------------------------------------------------
    # CHECK 6 — Image normalization statistics (multi-image)
    # ------------------------------------------------------------------

    def check_normalization_stats(self) -> None:
        print("\n── CHECK 6: Image Normalization Stats (multi-image) ─────────────")
        import numpy as np
        import torchvision.transforms as T
        from PIL import Image

        images_found = self._find_sample_images({}, max_images=50)
        if not images_found:
            self.result.failed("Norm stats (images)",
                               "No images found — cannot compute stats")
            return

        print(f"       Sampling {len(images_found)} images for stat estimation...")

        to_tensor         = T.ToTensor()
        per_channel_means = [[], [], []]
        per_channel_stds  = [[], [], []]

        for p in images_found:
            try:
                img = Image.open(p).convert("RGB")
                if img.size != (224, 224):
                    img = img.resize((224, 224))
                t = to_tensor(img)
                for c in range(3):
                    per_channel_means[c].append(t[c].mean().item())
                    per_channel_stds[c].append(t[c].std().item())
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Could not load {p}: {e}")

        if not per_channel_means[0]:
            self.result.failed("Norm stats", "Could not load any images")
            return

        ch_names      = ["R", "G", "B"]
        overall_means = []
        overall_stds  = []

        print()
        for c, ch in enumerate(ch_names):
            m = float(np.mean(per_channel_means[c]))
            s = float(np.mean(per_channel_stds[c]))
            overall_means.append(m)
            overall_stds.append(s)

            print(f"         Channel {ch}: mean={m:.4f}, std={s:.4f}")

            if m < 0.1:
                self.result.failed(
                    f"Norm mean channel {ch}",
                    f"mean={m:.6f} ← critically low (expected 0.45–0.95). "
                    "Images may be pre-normalized."
                )
            elif m < 0.3:
                self.result.warned(
                    f"Norm mean channel {ch}",
                    f"mean={m:.4f} — lower than expected. "
                    "Verify images are standard H&E patches."
                )
            else:
                self.result.passed(f"Norm mean channel {ch}",
                                   f"mean={m:.4f} in expected range [0.3–0.95] ✓")

            if s < 0.01:
                self.result.failed(f"Norm std channel {ch}",
                                   f"std={s:.8f} ≈ 0. "
                                   "Normalizer will divide by ~0 — training will fail.")
            else:
                self.result.passed(f"Norm std channel {ch}", f"std={s:.4f} ✓")

        if (abs(overall_means[0] - overall_means[1]) < 0.005 and
                abs(overall_means[1] - overall_means[2]) < 0.005):
            self.result.failed(
                "Color channels (multi-image)",
                f"R≈G≈B≈{overall_means[0]:.4f} — GRAYSCALE images detected!"
            )
        else:
            self.result.passed(
                "Color channels (multi-image)",
                f"R={overall_means[0]:.4f}, G={overall_means[1]:.4f}, "
                f"B={overall_means[2]:.4f} — channels differ ✓"
            )

        print(f"\n       Expected healthy H&E normalization values:")
        print(f"         Mean: ~[0.70, 0.55, 0.65]")
        print(f"         Std:  ~[0.15, 0.18, 0.14]")
        print(f"       Computed values:")
        print(f"         Mean: {[f'{m:.4f}' for m in overall_means]}")
        print(f"         Std:  {[f'{s:.4f}' for s in overall_stds]}")

    # ------------------------------------------------------------------
    # CHECK 7 — Gene count processing
    # ------------------------------------------------------------------

    def check_gene_counts(self) -> None:
        print("\n── CHECK 7: Gene Count Processing ──────────────────────────────")
        import numpy as np
        import torch
        import pickle

        npz_files = sorted(self.train_counts.rglob("*.npz"))
        if not npz_files:
            self.result.failed("Gene counts", "No NPZ files found")
            return

        gene_pkl = self.filter_root / "gene.pkl"
        mean_npy = self.filter_root / "mean_expression.npy"

        for f in [gene_pkl, mean_npy]:
            if not f.exists():
                self.result.failed("Gene counts metadata", f"Missing: {f}")
                return

        with open(gene_pkl, "rb") as f:
            gene_names = pickle.load(f)
        mean_expr  = np.load(mean_npy)
        total_genes = len(gene_names)

        self.result.passed("Gene filter",
                           f"gene_filter={self.gene_filter}, "
                           f"total_genes={total_genes}")

        if total_genes < self.gene_filter:
            self.result.failed("Gene count sufficient",
                               f"Only {total_genes} genes available, "
                               f"need ≥ {self.gene_filter}")

        aux_nums = int((total_genes - self.gene_filter) * self.aux_ratio)
        self.result.passed("aux_nums derivation",
                           f"aux_nums = ({total_genes} - {self.gene_filter}) "
                           f"× {self.aux_ratio} = {aux_nums} "
                           f"(NOT hardcoded to 6000)")

        if mean_expr.shape[0] != total_genes:
            self.result.failed("mean_expression shape",
                               f"mean_expression has {mean_expr.shape[0]} entries, "
                               f"gene.pkl has {total_genes}")
        else:
            self.result.passed("mean_expression shape",
                               f"{mean_expr.shape} matches gene list ✓")

        npz   = np.load(npz_files[0])
        count = npz["count"]

        if count.shape[0] != total_genes:
            self.result.failed("NPZ count shape",
                               f"NPZ has {count.shape[0]} values, "
                               f"gene.pkl has {total_genes}")
        else:
            self.result.passed("NPZ count shape",
                               f"count.shape={count.shape} matches gene list ✓")

        sorted_idx = np.argsort(mean_expr)[::-1]
        keep_bool  = np.zeros(total_genes, dtype=bool)
        keep_bool[sorted_idx[:self.gene_filter]] = True
        keep_count = count[keep_bool]
        y          = torch.log1p(torch.as_tensor(keep_count, dtype=torch.float))

        self.result.passed("log1p transform",
                           f"Input: [{keep_count.min():.0f}, {keep_count.max():.0f}] "
                           f"→ log1p: [{y.min():.4f}, {y.max():.4f}]")

        if self.aux_ratio > 0:
            aux_idx  = sorted_idx[self.gene_filter: self.gene_filter + aux_nums]
            aux_bool = np.zeros(total_genes, dtype=bool)
            aux_bool[aux_idx] = True
            aux_count = count[aux_bool]
            aux       = torch.log1p(torch.as_tensor(aux_count, dtype=torch.float))
            self.result.passed("Aux gene extraction",
                               f"aux shape: {aux.shape}, "
                               f"range: [{aux.min():.4f}, {aux.max():.4f}] "
                               f"(no z-score — matches paper)")

    # ------------------------------------------------------------------
    # CHECK 8 — DataLoader batch structure
    # ------------------------------------------------------------------

    def check_dataloader(self) -> None:
        print("\n── CHECK 8: DataLoader Batch Structure ──────────────────────────")
        import torch
        import torchvision.transforms as T
        from torch.utils.data import DataLoader

        try:
            from src.training.data_generator import SpatialDataset

            all_p   = set()
            for f in self.train_counts.rglob("*.npz"):
                all_p.add(f.parts[-2])
            train_p = sorted([p for p in all_p if p != self.test_patient])

            if not train_p:
                self.result.failed("DataLoader patients",
                                   "No training patients found")
                return

            ds = SpatialDataset(
                patient_list  = train_p[:1],
                window        = self.window,
                count_root    = str(self.train_counts),
                img_root      = str(self.train_images),
                gene_filter   = self.gene_filter,
                aux_ratio     = self.aux_ratio,
                transform     = T.ToTensor(),
                normalization = None,
            )

            if len(ds) == 0:
                self.result.failed("DataLoader dataset",
                                   f"Empty for patient {train_p[0]}. "
                                   "Check image path structure.")
                return

            self.result.passed("DataLoader dataset size",
                               f"{len(ds)} samples for patient {train_p[0]}")

            loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
            batch  = next(iter(loader))
            n      = len(batch)

            expected_n = 8 if self.aux_ratio > 0 else 7
            if n == expected_n:
                self.result.passed("DataLoader tuple length",
                                   f"{n} elements (expected {expected_n}) ✓")
            else:
                self.result.failed("DataLoader tuple length",
                                   f"Got {n} elements, expected {expected_n}.")

            X, y   = batch[0], batch[1]
            x_min  = X.min().item()
            x_max  = X.max().item()
            y_min  = y.min().item()
            y_max  = y.max().item()

            self.result.passed("Image tensor shape",
                               f"X.shape={X.shape} (expected [B,3,224,224])")
            if X.shape[1] != 3:
                self.result.failed("Image channels",
                                   f"Expected 3, got {X.shape[1]}")

            self.result.passed("Target tensor shape",
                               f"y.shape={y.shape} "
                               f"(expected [B,{self.gene_filter}])")
            if y.shape[1] != self.gene_filter:
                self.result.failed("Target gene count",
                                   f"Expected {self.gene_filter}, got {y.shape[1]}")

            if x_max > 1.01:
                self.result.failed("X range post ToTensor",
                                   f"[{x_min:.4f},{x_max:.4f}] — should be [0,1]")
            elif x_max < 0.05:
                self.result.failed("X range post ToTensor",
                                   f"[{x_min:.6f},{x_max:.6f}] — near zero! "
                                   "Pre-normalization suspected.")
            else:
                self.result.passed("X range post ToTensor",
                                   f"[{x_min:.4f},{x_max:.4f}] — correct [0,1] ✓")

            if y_min < 0:
                self.result.warned("y range",
                                   f"y.min={y_min:.4f} < 0 — "
                                   "z-score applied? Paper applies after log1p.")
            else:
                self.result.passed("y range (log1p)",
                                   f"[{y_min:.4f},{y_max:.4f}] — non-negative ✓")

            if self.aux_ratio > 0 and n >= 8:
                aux     = batch[2]
                aux_min = aux.min().item()
                aux_max = aux.max().item()
                self.result.passed("Aux tensor shape",
                                   f"aux.shape={aux.shape}")
                if aux_min < 0:
                    self.result.warned("Aux range",
                                       f"aux.min={aux_min:.4f} < 0 — "
                                       "aux should NOT be z-score normalized.")
                else:
                    self.result.passed("Aux range (log1p only)",
                                       f"[{aux_min:.4f},{aux_max:.4f}] — "
                                       "non-negative (no z-score) ✓")

        except Exception as e:
            self.result.failed("DataLoader check",
                               f"Exception: {e}\n"
                               f"       {traceback.format_exc()[-300:]}")

    # ------------------------------------------------------------------
    # CHECK 9 — Model forward pass
    # ------------------------------------------------------------------

    def check_model_forward(self) -> None:
        print("\n── CHECK 9: Model Forward Pass ──────────────────────────────────")
        import torch
        import pickle

        gene_pkl = self.filter_root / "gene.pkl"
        if gene_pkl.exists():
            with open(gene_pkl, "rb") as f:
                genes = pickle.load(f)
            total_genes = len(genes)
            self.result.passed("total_genes from gene.pkl",
                               f"{total_genes} genes found")
        else:
            self.result.failed("gene.pkl missing", f"Not found: {gene_pkl}")
            return

        aux_nums = int((total_genes - self.gene_filter) * self.aux_ratio)

        # Build config with total_genes at both top level and model level
        cfg = {
            **self.config,
            "total_genes": total_genes,
            "gene_filter": self.gene_filter,
            "aux_ratio":   self.aux_ratio,
            "pretrained":  False,
            "model": {
                **self.config.get("model", {}),
                "total_genes": total_genes,
                "gene_filter": self.gene_filter,
                "aux_ratio":   self.aux_ratio,
                "pretrained":  False,
            },
        }

        try:
            from src.models.classical.efficientnet_model import EfficientNetModel

            model  = EfficientNetModel(cfg)
            device = model.device
            model.eval()

            self.result.passed("Model instantiation",
                               f"main_genes={self.gene_filter}, "
                               f"aux_nums={model.aux_nums}, "
                               f"device={device}")

            if model.aux_nums != aux_nums:
                self.result.failed(
                    "Model aux_nums",
                    f"Model has aux_nums={model.aux_nums}, "
                    f"expected {aux_nums} = "
                    f"({total_genes} - {self.gene_filter}) × {self.aux_ratio}. "
                    "Ensure total_genes is passed to model config."
                )
            else:
                self.result.passed("Model aux_nums",
                                   f"{model.aux_nums} = "
                                   f"({total_genes} - {self.gene_filter}) "
                                   f"× {self.aux_ratio} ✓")

            # Create dummy tensor ON the model's device
            dummy = torch.zeros(2, 3, 224, 224, device=device)

            with torch.no_grad():
                out = model(dummy)

            if isinstance(out, (tuple, list)):
                main_pred = out[0]
                aux_pred  = out[1] if len(out) > 1 else None

                if main_pred.shape == torch.Size([2, self.gene_filter]):
                    self.result.passed("Forward pass — main output shape",
                                       f"{list(main_pred.shape)} ✓")
                else:
                    self.result.failed("Forward pass — main output shape",
                                       f"Got {list(main_pred.shape)}, "
                                       f"expected [2, {self.gene_filter}]")

                if aux_pred is not None:
                    if aux_pred.shape == torch.Size([2, aux_nums]):
                        self.result.passed("Forward pass — aux output shape",
                                           f"{list(aux_pred.shape)} ✓")
                    else:
                        self.result.failed("Forward pass — aux output shape",
                                           f"Got {list(aux_pred.shape)}, "
                                           f"expected [2, {aux_nums}]")
                else:
                    self.result.failed("Forward pass — aux output",
                                       "aux_pred is None — AuxNet head not installed")

                if main_pred.device.type == device.type:
                    self.result.passed("Forward pass — device consistency",
                                       f"Output on {main_pred.device} ✓")
                else:
                    self.result.failed("Forward pass — device consistency",
                                       f"Input on {device}, "
                                       f"output on {main_pred.device}")
            else:
                self.result.failed("Forward pass — output type",
                                   f"Expected tuple (main, aux), got {type(out)}")

        except Exception as e:
            self.result.failed("Model forward pass",
                               f"{e}\n       {traceback.format_exc()[-400:]}")

    # ------------------------------------------------------------------
    # CHECK 10 — Loss computation
    # ------------------------------------------------------------------

    def check_loss(self) -> None:
        print("\n── CHECK 10: Loss Computation ───────────────────────────────────")
        import torch
        import pickle

        gene_pkl = self.filter_root / "gene.pkl"
        if not gene_pkl.exists():
            self.result.failed("Loss check", f"gene.pkl missing: {gene_pkl}")
            return

        with open(gene_pkl, "rb") as f:
            genes = pickle.load(f)
        total_genes = len(genes)
        aux_nums    = int((total_genes - self.gene_filter) * self.aux_ratio)

        try:
            criterion = torch.nn.MSELoss()

            # requires_grad=True so backward() works
            pred_main   = torch.randn(32, self.gene_filter, requires_grad=True)
            target_main = torch.randn(32, self.gene_filter)
            main_loss   = criterion(pred_main, target_main)

            self.result.passed("MSELoss (main)",
                               f"shape=[32,{self.gene_filter}], "
                               f"loss={main_loss.item():.4f} ✓")

            pred_aux   = torch.randn(32, aux_nums, requires_grad=True)
            target_aux = torch.randn(32, aux_nums)
            aux_loss   = criterion(pred_aux, target_aux)

            self.result.passed("MSELoss (aux)",
                               f"shape=[32,{aux_nums}], "
                               f"loss={aux_loss.item():.4f} ✓")

            aux_weight = self.config.get("model", {}).get("aux_weight", 1.0)
            combined   = main_loss + aux_weight * aux_loss

            self.result.passed("Combined loss",
                               f"main({main_loss.item():.4f}) + "
                               f"{aux_weight}×aux({aux_loss.item():.4f}) "
                               f"= {combined.item():.4f} ✓")

            combined.backward()

            if pred_main.grad is not None and pred_aux.grad is not None:
                self.result.passed("Loss backward (gradients)",
                                   f"pred_main.grad={list(pred_main.grad.shape)}, "
                                   f"pred_aux.grad={list(pred_aux.grad.shape)} ✓")
            else:
                self.result.failed("Loss backward (gradients)",
                                   "Gradients not computed")

        except Exception as e:
            self.result.failed("Loss computation", f"{e}")

    # ------------------------------------------------------------------
    # CHECK 11 — Mini training dry run
    # ------------------------------------------------------------------

    def check_mini_training(self) -> None:
        print("\n── CHECK 11: Mini Training Dry Run (3 epochs) ───────────────────")
        import torch
        import pickle
        import numpy as np
        import torchvision.transforms as T
        from torch.utils.data import DataLoader

        try:
            from src.training.data_generator import SpatialDataset
            from src.models.classical.efficientnet_model import EfficientNetModel
            from src.training.metrics import compute_all_metrics

            gene_pkl = self.filter_root / "gene.pkl"
            with open(gene_pkl, "rb") as f:
                genes = pickle.load(f)
            total_genes = len(genes)

            all_p   = set()
            for f in self.train_counts.rglob("*.npz"):
                all_p.add(f.parts[-2])
            train_p = sorted([p for p in all_p if p != self.test_patient])[:1]

            if not train_p:
                self.result.failed("Mini training", "No training patients found")
                return

            ds = SpatialDataset(
                patient_list  = train_p,
                window        = self.window,
                count_root    = str(self.train_counts),
                img_root      = str(self.train_images),
                gene_filter   = self.gene_filter,
                aux_ratio     = self.aux_ratio,
                transform     = T.ToTensor(),
                normalization = None,
            )

            if len(ds) == 0:
                self.result.failed("Mini training dataset",
                                   "Empty dataset — cannot run mini training")
                return

            loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

            cfg = {
                **self.config,
                "total_genes": total_genes,
                "gene_filter": self.gene_filter,
                "aux_ratio":   self.aux_ratio,
                "pretrained":  False,
                "model": {
                    **self.config.get("model", {}),
                    "total_genes": total_genes,
                    "gene_filter": self.gene_filter,
                    "aux_ratio":   self.aux_ratio,
                    "pretrained":  False,
                },
            }

            model     = EfficientNetModel(cfg)
            device    = model.device
            optim     = torch.optim.SGD(model.parameters(), lr=1e-3,
                                        momentum=0.9, weight_decay=1e-6)
            criterion = torch.nn.MSELoss()

            print(f"       Running 3 mini-epochs on {len(ds)} samples "
                  f"(device={device})...")
            start = time.time()

            for epoch in range(3):
                model.train()
                epoch_loss   = 0.0
                epoch_preds  = []
                epoch_counts = []
                n_batches    = 0

                for batch in loader:
                    n = len(batch)
                    X = batch[0].to(device)
                    y = batch[1].to(device)
                    aux = batch[2].to(device) if n >= 8 else None

                    optim.zero_grad()
                    out = model(X)

                    main_p = out[0] if isinstance(out, (tuple, list)) else out
                    aux_p  = out[1] if isinstance(out, (tuple, list)) else None

                    loss = criterion(main_p, y)
                    if aux_p is not None and aux is not None:
                        min_a = min(aux.shape[1], aux_p.shape[1])
                        loss  = loss + criterion(aux_p[:, :min_a], aux[:, :min_a])

                    loss.backward()
                    optim.step()

                    epoch_loss += loss.item()
                    epoch_preds.append(main_p.cpu().detach().numpy())
                    epoch_counts.append(y.cpu().detach().numpy())
                    n_batches += 1

                ep = np.concatenate(epoch_preds)
                ec = np.concatenate(epoch_counts)
                m  = compute_all_metrics(ep, ec)
                avg_loss = epoch_loss / max(n_batches, 1)

                print(f"       Epoch {epoch+1}: loss={avg_loss:.4f}  "
                      f"aMAE={m['amae']:.4f}  "
                      f"aCC={m['correlation_coefficient']:.4f}")

            elapsed    = time.time() - start
            final_loss = epoch_loss / max(n_batches, 1)

            self.result.passed("Mini training (3 epochs)",
                               f"Completed in {elapsed:.1f}s, "
                               f"final loss={final_loss:.4f}")

            if final_loss > 10.0:
                self.result.failed("Mini training loss range",
                                   f"Loss={final_loss:.4f} > 10.0 — "
                                   "normalization issue suspected.")
            elif final_loss > 2.0:
                self.result.warned("Mini training loss range",
                                   f"Loss={final_loss:.4f} — higher than "
                                   "expected (~0.09). Check image normalization.")
            else:
                self.result.passed("Mini training loss range",
                                   f"Loss={final_loss:.4f} — reasonable ✓")

        except Exception as e:
            self.result.failed("Mini training",
                               f"{e}\n       {traceback.format_exc()[-500:]}")

    # ------------------------------------------------------------------
    # CHECK 12 — Configuration validation
    # ------------------------------------------------------------------

    def check_config(self) -> None:
        print("\n── CHECK 12: Configuration Validation ───────────────────────────")

        required_keys = {
            "data.test_patient":                "BC23450",
            "data.train_counts_dir":            "data/train/counts/",
            "data.train_images_dir":            "data/train/images/",
            "data.test_counts_dir":             "data/test/counts/",
            "data.test_images_dir":             "data/test/images/",
            "preprocessing.window_size":        299,
            "model.gene_filter":                250,
            "model.aux_ratio":                  1.0,
            "model.pretrained":                 True,
            "training.lr":                      1e-3,
            "training.momentum":                0.9,
            "training.weight_decay":            1e-6,
            "training.cosine_t_max":            5,
            "training.batch_size":              32,
            "training.early_stopping_patience": 20,
        }

        def _get(d, dotkey):
            v = d
            try:
                for k in dotkey.split("."):
                    v = v[k]
                return v
            except (KeyError, TypeError):
                return None

        for key, paper_default in required_keys.items():
            val = _get(self.config, key)
            if val is None:
                self.result.warned(f"Config key: {key}",
                                   f"Missing — paper default: {paper_default}")
            elif val == paper_default:
                self.result.passed(f"Config: {key}",
                                   f"{val} (matches paper default) ✓")
            else:
                self.result.warned(f"Config: {key}",
                                   f"value={val}, paper default={paper_default}. "
                                   "Non-default — ensure intentional.")

    # ------------------------------------------------------------------
    # Helper: find sample images
    # ------------------------------------------------------------------

    def _find_sample_image(self, subtype: dict) -> Optional[Path]:
        """Find one sample image using paper's path structure."""
        for patient, st in subtype.items():
            base = self.train_images / st / patient / str(self.window)
            if base.exists():
                imgs = sorted(base.glob("*.jpg"))
                if imgs:
                    return imgs[0]
        for img in self.train_images.rglob("*.jpg"):
            return img
        return None

    def _find_sample_images(self, subtype: dict,
                            max_images: int = 50) -> List[Path]:
        """Find up to max_images sample images."""
        found = []
        for img in self.train_images.rglob("*.jpg"):
            found.append(img)
            if len(found) >= max_images:
                break
        return found

    # ------------------------------------------------------------------
    # Run all checks
    # ------------------------------------------------------------------

    def run_all(self) -> Tuple[int, int]:
        """Run every check in order. Returns (n_fails, n_warns)."""
        print("\n" + "=" * 70)
        print("  SPATIAL TRANSCRIPTOMICS PIPELINE DIAGNOSTIC")
        print("=" * 70)
        print(f"  Config:       {self.config.get('_config_path', 'loaded')}")
        print(f"  Test patient: {self.test_patient}")
        print(f"  Window size:  {self.window}")
        print(f"  Gene filter:  {self.gene_filter}")
        print(f"  Aux ratio:    {self.aux_ratio}")
        print("=" * 70)

        self.check_environment()
        self.check_directories()
        self.check_metadata()
        self.check_npz_files()
        self.check_image_loading()
        self.check_normalization_stats()
        self.check_gene_counts()
        self.check_dataloader()
        self.check_model_forward()
        self.check_loss()
        self.check_config()
        self.check_mini_training()

        return self.result.summary()


# ======================================================================
# Auto-fix helper (--fix mode)
# ======================================================================

def attempt_fixes(config: Dict[str, Any]) -> None:
    """Suggest auto-fixes for common issues. Never modifies data files."""
    print("\n" + "=" * 70)
    print("  AUTO-FIX MODE")
    print("=" * 70)

    data         = config.get("data", {})
    train_images = Path(data.get("train_images_dir", "data/train/images/"))
    subtype_pkl  = Path(data.get("count_filtered_dir",
                                 "data/processed/count_filtered/")) / "subtype.pkl"

    print("\n[Fix 1] Checking image path structure...")
    if subtype_pkl.exists():
        import pickle
        with open(subtype_pkl, "rb") as f:
            subtype = pickle.load(f)
        window = config.get("preprocessing", {}).get("window_size", 299)

        for patient, st in list(subtype.items())[:2]:
            with_sub    = train_images / st / patient / str(window)
            without_sub = train_images / patient / str(window)

            if with_sub.exists():
                print(f"  ✅ Path includes subtype/: {with_sub}")
            elif without_sub.exists():
                print(f"  ⚠️  Images exist WITHOUT subtype/: {without_sub}")
                print(f"     Paper expects: {with_sub}")
                print(f"     Suggested fix:")
                print(f"       mkdir -p '{train_images / st}'")
                print(f"       mv '{train_images / patient}' "
                      f"'{train_images / st / patient}'")
            else:
                print(f"  ❌ No images found for patient {patient}")
                print(f"     Run patch extraction pipeline first.")

    print("\n[Fix 2] Checking config for missing keys...")
    training = config.get("training", {})

    if "lr" not in training and "learning_rate" not in training:
        print("  ⚠️  training.lr missing — add: lr: 0.001")
    if "cosine_t_max" not in training:
        print("  ⚠️  training.cosine_t_max missing — add: cosine_t_max: 5")
    if config.get("preprocessing", {}).get("window_size", 224) == 224:
        print("  ⚠️  preprocessing.window_size=224 — paper uses 299")


# ======================================================================
# Entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Spatial Transcriptomics Pipeline Diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/diagnose_pipeline.py --config config/pipeline_config.yaml
  python scripts/diagnose_pipeline.py --config config/pipeline_config.yaml --verbose
  python scripts/diagnose_pipeline.py --config config/pipeline_config.yaml --fix
  python scripts/diagnose_pipeline.py --config config/pipeline_config.yaml --skip-training
        """
    )
    parser.add_argument("--config", "-c", required=True,
                        help="Path to pipeline_config.yaml")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show verbose debug output")
    parser.add_argument("--fix", action="store_true",
                        help="Suggest auto-fixes for detected issues")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip the mini training dry-run (faster)")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(config_path)

    diag = PipelineDiagnostic(config, verbose=args.verbose)

    if args.skip_training:
        diag.check_mini_training = lambda: diag.result.warned(
            "Mini training", "Skipped via --skip-training flag"
        )

    fails, warns = diag.run_all()

    if args.fix:
        attempt_fixes(config)

    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
