"""
Data pipeline component for Vahadane H&E stain normalization.
Replaces: normalize.py  +  stain_normalizer.py (the standalone versions)

The Vahadane algorithm classes (LuminosityStandardizer,
VahadaneStainExtractor, VahadaneNormalizerCore) are kept as private
helpers within this module so the public surface is a single
factory-compatible class: SpatialStainNormalizer.
"""

import gc
import glob
import logging
import numpy as np
import cv2
from typing import Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import DictionaryLearning

from src.data_pipeline.base_pipeline import BaseDataPipeline

logger = logging.getLogger(__name__)


# ======================================================================
# Algorithm helpers  (previously stain_normalizer.py)
# ======================================================================

class _LuminosityStandardizer:
    """Standardize image luminosity via LAB L-channel rescaling."""

    @staticmethod
    def standardize(image: np.ndarray, percentile: int = 95) -> np.ndarray:
        """
        Rescale the L channel so that its <percentile>-th value maps to 255.

        Args:
            image     : RGB uint8 image.
            percentile: Percentile used as the white-point reference.

        Returns:
            Luminosity-standardized RGB uint8 image.
        """
        img_f = image.astype(np.float32)
        lab   = cv2.cvtColor(img_f, cv2.COLOR_RGB2LAB)
        L     = lab[:, :, 0]
        p     = np.percentile(L, percentile)
        lab[:, :, 0] = np.clip(255.0 * L / (p + 1e-6), 0, 255)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.uint8)


class _VahadaneStainExtractor:
    """Extract a 2×3 H&E stain matrix via sparse dictionary learning."""

    def __init__(self, luminosity_threshold: float = 0.8):
        self.luminosity_threshold = luminosity_threshold

    def get_stain_matrix(
            self, image: np.ndarray, regularizer: float = 0.1
    ) -> np.ndarray:
        """
        Compute the 2×3 stain matrix for an RGB image.

        Args:
            image      : RGB uint8 image.
            regularizer: L1 regularisation for DictionaryLearning.

        Returns:
            2×3 numpy array [Haematoxylin vector; Eosin vector].

        Raises:
            ValueError: Fewer than 100 tissue pixels available.
        """
        img_f = image.astype(np.float32)
        OD    = -np.log((img_f + 1) / 256.0)
        ODhat = OD[~self._is_background(img_f)]

        if len(ODhat) < 100:
            raise ValueError(
                "Not enough tissue pixels for stain extraction "
                f"(found {len(ODhat)} < 100)"
            )

        try:
            dl = DictionaryLearning(
                n_components=2,
                alpha=regularizer,
                max_iter=1000,
                fit_algorithm='lars',
                transform_algorithm='lasso_lars',
                positive_dict=True,
                positive_code=True,
                random_state=0,
                verbose=0,
            )
            dl.fit(ODhat)
            dictionary = dl.components_
        except Exception as exc:
            logger.warning(f"DictionaryLearning failed ({exc}); using PCA fallback")
            dictionary = self._pca_fallback(ODhat)

        # Ensure Haematoxylin is row 0
        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]

        return dictionary

    # --- private ---

    def _is_background(self, image: np.ndarray) -> np.ndarray:
        return np.all(image > 255 * self.luminosity_threshold, axis=-1)

    @staticmethod
    def _pca_fallback(OD: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit(OD).components_


class _VahadaneNormalizerCore:
    """
    Core Vahadane stain normalizer (fit / transform).

    Previously the StainNormalizer class in standalone stain_normalizer.py.
    Kept private here; SpatialStainNormalizer is the public API.
    """

    def __init__(self) -> None:
        self._extractor         = _VahadaneStainExtractor()
        self._target_stain: Optional[np.ndarray] = None
        self._max_c_target:  Optional[np.ndarray] = None

    def fit(self, target: np.ndarray) -> None:
        """Fit to a reference image (RGB uint8)."""
        target = _LuminosityStandardizer.standardize(target)
        self._target_stain  = self._extractor.get_stain_matrix(target)
        conc                = self._get_concentrations(target, self._target_stain)
        self._max_c_target  = np.percentile(conc, 99, axis=0).reshape((1, 2))

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize staining of an RGB uint8 image to match the fitted target.

        Args:
            image: Source RGB uint8 image.

        Returns:
            Stain-normalised RGB uint8 image.

        Raises:
            ValueError: If fit() has not been called.
        """
        if self._target_stain is None:
            raise ValueError("Call fit() before transform()")

        image          = _LuminosityStandardizer.standardize(image)
        src_stain      = self._extractor.get_stain_matrix(image)
        src_conc       = self._get_concentrations(image, src_stain)
        max_c_src      = np.percentile(src_conc, 99, axis=0).reshape((1, 2))

        src_conc      *= self._max_c_target / (max_c_src + 1e-6)
        h, w           = image.shape[:2]
        I_norm         = 255 * np.exp(
            -(self._target_stain.T @ src_conc.T)
        )
        I_norm         = I_norm.T.reshape(h, w, 3)
        return np.clip(I_norm, 0, 255).astype(np.uint8)

    @staticmethod
    def _get_concentrations(
            image: np.ndarray, stain_matrix: np.ndarray
    ) -> np.ndarray:
        OD   = -np.log((image.astype(np.float32) + 1) / 256.0).reshape(-1, 3)
        try:
            return np.linalg.lstsq(stain_matrix.T, OD.T, rcond=None)[0].T
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(stain_matrix.T) @ OD.T).T


# ======================================================================
# Factory-compatible pipeline component
# ======================================================================

class SpatialStainNormalizer(BaseDataPipeline):
    """
    Vahadane H&E stain normalization for all tissue images.

    Replaces normalize.py's stain_normalization() function and the
    standalone StainNormalizer class from stain_normalizer.py with a
    single factory-compatible pipeline component.

    Pipeline::

        stbc/*.jpg
          → luminosity standardize
          → Vahadane stain normalisation  (fallback: copy original)
          → tissue mask via Otsu threshold
          → stained/*.jpg  +  stained/*_mask.jpg
    """

    # Default reference filename (from original normalize.py)
    _DEFAULT_REFERENCE = "BC24220_E1.jpg"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the stain normalizer.

        Args:
            config: Configuration dictionary containing:
                - data.stbc_dir        : Input directory of raw images
                - data.stained_dir     : Output directory for normalised images
                - stain.reference_file : Reference image filename
                                         (default: BC24220_E1.jpg)
        """
        super().__init__(config)
        self.stbc_dir        = self.get_config_value('data.stbc_dir',    'data/stbc/')
        self.stained_dir     = self.get_config_value('data.stained_dir', 'data/stained/')
        self.reference_file  = self.get_config_value(
            'stain.reference_file', self._DEFAULT_REFERENCE
        )

    # ------------------------------------------------------------------
    # BaseDataPipeline interface
    # ------------------------------------------------------------------

    def execute(self) -> bool:
        """
        Execute the stain normalisation stage.

        Returns:
            True if normalisation succeeded, False otherwise.
        """
        self.log_info("Starting H&E stain normalisation")

        try:
            self._ensure_directory_exists(self.stained_dir)

            normalizer = self._fit_normalizer()
            self._normalize_all_images(normalizer)

            self.log_info("Stain normalisation completed successfully")
            return True

        except Exception as e:
            self.log_error(f"Stain normalisation failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Private helpers  (mirrors original normalize.py logic)
    # ------------------------------------------------------------------

    def _fit_normalizer(self) -> Optional[_VahadaneNormalizerCore]:
        """
        Locate the reference image and fit a Vahadane normaliser to it.

        Replicates the reference-image search and normalizer.fit() call
        from the original normalize.py stain_normalization() function,
        including the fallback to the first available image when the
        default reference is missing.

        Returns:
            Fitted _VahadaneNormalizerCore, or None if fitting fails.
        """
        stbc_path = self._resolve_path(self.stbc_dir)

        # --- search for reference (original normalize.py logic) ---
        ref_matches = list(stbc_path.rglob(self.reference_file))
        if ref_matches:
            ref_path = ref_matches[0]
        else:
            all_imgs = [
                p for p in stbc_path.rglob("*.jpg")
                if "_mask" not in p.name
            ]
            if not all_imgs:
                raise FileNotFoundError(f"No JPG images found in {stbc_path}")
            ref_path = all_imgs[0]
            self.log_warning(
                f"Reference '{self.reference_file}' not found; "
                f"using fallback: {ref_path.name}"
            )

        self.log_info(f"Fitting normaliser to: {ref_path}")

        try:
            target_img = cv2.imread(str(ref_path))
            if target_img is None:
                raise ValueError(f"Could not load reference image: {ref_path}")

            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            target_img = _LuminosityStandardizer.standardize(target_img)

            normalizer = _VahadaneNormalizerCore()
            normalizer.fit(target_img)

            del target_img
            gc.collect()

            self.log_info("Normaliser fitted successfully")
            return normalizer

        except Exception as exc:
            self.log_error(f"Failed to fit normaliser: {exc}")
            self.log_warning("Stain normalisation disabled — originals will be copied")
            return None

    def _normalize_all_images(
            self, normalizer: Optional[_VahadaneNormalizerCore]
    ) -> None:
        """
        Iterate over every non-mask JPG in stbc_dir, normalise, and save.

        Replicates the per-image loop from the original normalize.py
        stain_normalization() function, including:
        - Otsu tissue-mask generation and saving as *_mask.jpg
        - Vahadane normalisation with fallback to the original image
        - BGR/RGB conversion symmetry

        Args:
            normalizer: Fitted normaliser, or None to skip normalisation.
        """
        stbc_path    = self._resolve_path(self.stbc_dir)
        stained_path = self._resolve_path(self.stained_dir)

        images = [
            p for p in stbc_path.rglob("*.jpg")
            if "_mask" not in p.name
        ]

        if not images:
            self.log_warning(f"No images found in {stbc_path}")
            return

        success_count = fail_count = 0

        for img_path in tqdm(images, desc="Normalising"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    self.log_warning(f"Could not read image: {img_path}")
                    continue

                # --- compute output paths ---
                rel_path  = img_path.relative_to(stbc_path)
                out_path  = stained_path / rel_path
                mask_path = out_path.parent / (out_path.stem + "_mask.jpg")
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # --- tissue mask via Otsu (original normalize.py) ---
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, mask = cv2.threshold(
                    blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                cv2.imwrite(str(mask_path), mask)

                # --- stain normalisation ---
                if normalizer is not None:
                    try:
                        img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_norm = normalizer.transform(img_rgb)
                        cv2.imwrite(
                            str(out_path),
                            cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
                        )
                        success_count += 1
                    except Exception as exc:
                        self.log_warning(
                            f"Normalisation failed for {img_path.name}: "
                            f"{str(exc)[:60]} — copying original"
                        )
                        cv2.imwrite(str(out_path), img)
                        fail_count += 1
                else:
                    cv2.imwrite(str(out_path), img)
                    fail_count += 1

                del img, gray, blur, mask

            except Exception as exc:
                self.log_error(f"Error processing {img_path}: {exc}")
                fail_count += 1

        self.log_info(
            f"Normalisation complete — "
            f"normalised: {success_count}, copied as original: {fail_count}"
        )

    # ------------------------------------------------------------------
    # Public alias
    # ------------------------------------------------------------------

    def normalize_stains(self) -> bool:
        """Public alias for execute()."""
        return self.execute()


__all__ = ['SpatialStainNormalizer']
