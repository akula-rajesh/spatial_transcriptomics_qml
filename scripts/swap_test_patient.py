#!/usr/bin/env python3
"""
swap_test_patient.py
====================
Rearranges the train/test data split for a given test patient ID.

What it does
------------
1. Finds the new test patient in data/train/  (counts + images)
2. Finds the current test patient in data/test/ (counts + images)
3. Moves current test patient → data/train/   (preserving subtype/window structure)
4. Moves new test patient    → data/test/     (preserving subtype/window structure)
5. Updates data.test_patient in the specified pipeline config YAML (optional)

Directory structure assumed
---------------------------
data/train/counts/<subtype>/<patient>/<section>_<x>_<y>.npz
data/train/images/<subtype>/<patient>/<window>/<section>_<x>_<y>.jpg

data/test/counts/<subtype>/<patient>/<section>_<x>_<y>.npz
data/test/images/<subtype>/<patient>/<window>/<section>_<x>_<y>.jpg

Usage
-----
# Dry run (no files moved, just preview):
python scripts/swap_test_patient.py --new-test BC23287 --dry-run

# Swap patient + update one config:
python scripts/swap_test_patient.py --new-test BC23287 \
    --config config/pipeline_config_classical_efficientnet.yaml

# Swap patient + update ALL pipeline configs:
python scripts/swap_test_patient.py --new-test BC23287 --update-all-configs

# Swap only (no config update):
python scripts/swap_test_patient.py --new-test BC23287
"""

import argparse
import shutil
import logging
import sys
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data"
TRAIN_ROOT   = DATA_ROOT / "train"
TEST_ROOT    = DATA_ROOT / "test"
CONFIG_DIR   = PROJECT_ROOT / "config"

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("swap_test_patient")


# ══════════════════════════════════════════════════════════════════════════════
# Discovery helpers
# ══════════════════════════════════════════════════════════════════════════════

def find_patient_in_split(split_root: Path, patient_id: str):
    """
    Locate a patient directory inside a split (train or test).

    Returns
    -------
    dict with keys:
        counts_dir   : Path  — e.g. data/train/counts/HER2_luminal/BC23450
        images_dir   : Path  — e.g. data/train/images/HER2_luminal/BC23450
        subtype      : str   — e.g. "HER2_luminal"
        counts_root  : Path  — e.g. data/train/counts
        images_root  : Path  — e.g. data/train/images

    Returns None if patient is not found.
    """
    counts_root = split_root / "counts"
    images_root = split_root / "images"

    # Search counts/<subtype>/<patient>/
    for subtype_dir in sorted(counts_root.iterdir()):
        if not subtype_dir.is_dir():
            continue
        patient_counts = subtype_dir / patient_id
        if patient_counts.is_dir():
            subtype = subtype_dir.name
            patient_images = images_root / subtype / patient_id
            return {
                "counts_dir":  patient_counts,
                "images_dir":  patient_images if patient_images.is_dir() else None,
                "subtype":     subtype,
                "counts_root": counts_root,
                "images_root": images_root,
            }
    return None


def list_all_patients(split_root: Path):
    """Return sorted list of all patient IDs found in a split."""
    counts_root = split_root / "counts"
    patients = set()
    if counts_root.exists():
        for subtype_dir in counts_root.iterdir():
            if subtype_dir.is_dir():
                for p in subtype_dir.iterdir():
                    if p.is_dir():
                        patients.add(p.name)
    return sorted(patients)


# ══════════════════════════════════════════════════════════════════════════════
# Move helpers
# ══════════════════════════════════════════════════════════════════════════════

def move_patient(
    patient_id: str,
    from_info:  dict,
    to_root:    Path,
    dry_run:    bool,
):
    """
    Move counts and images for one patient from one split to another.

    Preserves full subtype / window structure:
        from: data/train/counts/HER2_luminal/BC23287/
          to:  data/test/counts/HER2_luminal/BC23287/

        from: data/train/images/HER2_luminal/BC23287/299/
          to:  data/test/images/HER2_luminal/BC23287/299/
    """
    subtype      = from_info["subtype"]
    counts_src   = from_info["counts_dir"]
    images_src   = from_info["images_dir"]

    counts_dst   = to_root / "counts" / subtype / patient_id
    images_dst   = to_root / "images" / subtype / patient_id

    # ── counts ────────────────────────────────────────────────────────────────
    npz_files = list(counts_src.rglob("*.npz"))
    logger.info(
        f"  counts : {counts_src.relative_to(PROJECT_ROOT)}  →  "
        f"{counts_dst.relative_to(PROJECT_ROOT)}  "
        f"({len(npz_files)} .npz files)"
    )
    if not dry_run:
        counts_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(counts_src), str(counts_dst))

    # ── images ────────────────────────────────────────────────────────────────
    if images_src and images_src.exists():
        jpg_files = list(images_src.rglob("*.jpg"))
        logger.info(
            f"  images : {images_src.relative_to(PROJECT_ROOT)}  →  "
            f"{images_dst.relative_to(PROJECT_ROOT)}  "
            f"({len(jpg_files)} .jpg files)"
        )
        if not dry_run:
            images_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(images_src), str(images_dst))
    else:
        logger.warning(
            f"  images : no image directory found for {patient_id} in "
            f"{from_info['images_root'].relative_to(PROJECT_ROOT)} — skipped"
        )

    # ── cleanup empty subtype dirs ────────────────────────────────────────────
    if not dry_run:
        _remove_if_empty(from_info["counts_root"] / subtype)
        _remove_if_empty(from_info["images_root"] / subtype)


def _remove_if_empty(directory: Path):
    """Remove a directory if it has no remaining children."""
    if directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()
        logger.info(f"  Removed empty directory: {directory.relative_to(PROJECT_ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# Config update helpers
# ══════════════════════════════════════════════════════════════════════════════

def update_config_test_patient(config_path: Path, new_patient: str, dry_run: bool):
    """
    Update data.test_patient in a YAML config file.

    Uses simple line-by-line replacement to avoid reformatting the entire
    YAML (preserves comments and formatting).
    """
    if not config_path.exists():
        logger.warning(f"  Config not found, skipped: {config_path}")
        return False

    content = config_path.read_text()
    lines   = content.splitlines(keepends=True)
    updated = []
    changed = False

    for line in lines:
        stripped = line.lstrip()
        # Match:  test_patient: "BC23450"  or  test_patient: BC23450
        if stripped.startswith("test_patient:") and not stripped.startswith("test_patients:"):
            indent = len(line) - len(stripped)
            new_line = " " * indent + f'test_patient: "{new_patient}"\n'
            if new_line != line:
                changed = True
                logger.info(
                    f"  {config_path.name}: "
                    f"test_patient → \"{new_patient}\""
                )
            updated.append(new_line)
        else:
            updated.append(line)

    if not changed:
        logger.info(f"  {config_path.name}: already set to \"{new_patient}\" — no change")
        return False

    if not dry_run:
        config_path.write_text("".join(updated))
        logger.info(f"  Saved: {config_path.relative_to(PROJECT_ROOT)}")

    return True


def find_all_pipeline_configs():
    """Return all pipeline_config*.yaml files in config/."""
    return sorted(CONFIG_DIR.glob("pipeline_config*.yaml"))


# ══════════════════════════════════════════════════════════════════════════════
# Main logic
# ══════════════════════════════════════════════════════════════════════════════

def swap_test_patient(
    new_test_patient: str,
    config_path:      Path | None,
    update_all:       bool,
    dry_run:          bool,
):
    prefix = "[DRY RUN] " if dry_run else ""
    logger.info(f"{prefix}{'=' * 60}")
    logger.info(f"{prefix}Swap test patient → \"{new_test_patient}\"")
    logger.info(f"{prefix}{'=' * 60}")

    # ── 1. Discover current state ─────────────────────────────────────────────
    current_test_patients = list_all_patients(TEST_ROOT)
    current_train_patients = list_all_patients(TRAIN_ROOT)

    logger.info(f"Current test  patients : {current_test_patients}")
    logger.info(f"Current train patients : {current_train_patients}")

    # ── 2. Validate new test patient ──────────────────────────────────────────
    if new_test_patient in current_test_patients:
        logger.warning(
            f"\"{new_test_patient}\" is already the test patient. Nothing to do."
        )
        return

    new_info = find_patient_in_split(TRAIN_ROOT, new_test_patient)
    if new_info is None:
        logger.error(
            f"Patient \"{new_test_patient}\" not found in data/train/counts/. "
            f"Available train patients: {current_train_patients}"
        )
        sys.exit(1)

    logger.info(
        f"\nFound \"{new_test_patient}\" in train "
        f"(subtype: {new_info['subtype']})"
    )

    # ── 3. Move existing test patients → train ────────────────────────────────
    for old_test_id in current_test_patients:
        logger.info(f"\n{prefix}Moving existing test patient \"{old_test_id}\" → train/")
        old_info = find_patient_in_split(TEST_ROOT, old_test_id)
        if old_info is None:
            logger.warning(f"  Could not find \"{old_test_id}\" in test — skipped")
            continue
        move_patient(old_test_id, old_info, TRAIN_ROOT, dry_run)

    # ── 4. Move new test patient → test ───────────────────────────────────────
    logger.info(f"\n{prefix}Moving \"{new_test_patient}\" → test/")
    move_patient(new_test_patient, new_info, TEST_ROOT, dry_run)

    # ── 5. Update config(s) ───────────────────────────────────────────────────
    configs_to_update = []

    if update_all:
        configs_to_update = find_all_pipeline_configs()
        logger.info(f"\n{prefix}Updating all pipeline configs ({len(configs_to_update)} files):")
    elif config_path is not None:
        configs_to_update = [config_path]
        logger.info(f"\n{prefix}Updating config: {config_path.name}")

    if configs_to_update:
        for cfg in configs_to_update:
            update_config_test_patient(cfg, new_test_patient, dry_run)
    else:
        logger.info(
            "\nNo config updated. Pass --config <file> or --update-all-configs "
            "to auto-update test_patient in YAML."
        )

    # ── 6. Summary ────────────────────────────────────────────────────────────
    logger.info(f"\n{prefix}{'=' * 60}")
    if dry_run:
        logger.info("DRY RUN complete — no files were moved or modified.")
        logger.info("Remove --dry-run to apply the changes.")
    else:
        logger.info(f"Done. New test patient: \"{new_test_patient}\"")
        logger.info(
            f"  data/test/  ← \"{new_test_patient}\""
        )
        logger.info(
            f"  data/train/ ← {current_test_patients} (former test)"
        )
    logger.info(f"{prefix}{'=' * 60}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Swap the test patient in the train/test data split.\n\n"
            "Examples:\n"
            "  # Preview only (no files moved):\n"
            "  python scripts/swap_test_patient.py --new-test BC23287 --dry-run\n\n"
            "  # Swap + update one config:\n"
            "  python scripts/swap_test_patient.py --new-test BC23287 \\\n"
            "      --config config/pipeline_config_classical_efficientnet.yaml\n\n"
            "  # Swap + update all pipeline configs:\n"
            "  python scripts/swap_test_patient.py --new-test BC23287 --update-all-configs\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--new-test",
        required=True,
        metavar="PATIENT_ID",
        help="Patient ID to use as the new test patient (must exist in data/train/).",
    )
    parser.add_argument(
        "--config",
        metavar="CONFIG_YAML",
        default=None,
        help=(
            "Path to a single pipeline config YAML to update. "
            "Relative paths are resolved from the project root."
        ),
    )
    parser.add_argument(
        "--update-all-configs",
        action="store_true",
        default=False,
        help="Update data.test_patient in ALL pipeline_config*.yaml files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview what would happen without moving any files or editing configs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve config path
    config_path = None
    if args.config:
        p = Path(args.config)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        config_path = p

    swap_test_patient(
        new_test_patient = args.new_test,
        config_path      = config_path,
        update_all       = args.update_all_configs,
        dry_run          = args.dry_run,
    )


if __name__ == "__main__":
    main()
