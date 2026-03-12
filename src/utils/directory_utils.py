"""
Utility functions for ensuring necessary directories exist.
"""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def ensure_directories_exist(directories: List[str], base_path: Optional[Path] = None) -> None:
    """
    Ensure that all specified directories exist, creating them if necessary.

    Args:
        directories: List of directory paths to create
        base_path: Optional base path to prepend to all directories
    """
    for directory in directories:
        if base_path:
            dir_path = base_path / directory
        else:
            dir_path = Path(directory)

        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            raise


def ensure_project_structure(project_root: Optional[Path] = None) -> None:
    """
    Ensure the basic project directory structure exists.

    Args:
        project_root: Root directory of the project (defaults to current directory)
    """
    if project_root is None:
        project_root = Path.cwd()

    # Standard project directories
    standard_dirs = [
        'logs',
        'data',
        'data/input',
        'data/processed',
        'data/train',
        'data/test',
        'results',
        'results/checkpoints',
        'config'
    ]

    ensure_directories_exist(standard_dirs, project_root)
    logger.info(f"Project directory structure verified at {project_root}")


def ensure_file_parent_exists(file_path: Path) -> None:
    """
    Ensure the parent directory of a file exists.

    Args:
        file_path: Path to the file
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)


def safe_open(file_path: Path, mode: str = 'r', **kwargs):
    """
    Safely open a file, creating parent directories if necessary for write modes.

    Args:
        file_path: Path to the file
        mode: File open mode
        **kwargs: Additional arguments to pass to open()

    Returns:
        File object
    """
    if 'w' in mode or 'a' in mode or 'x' in mode:
        ensure_file_parent_exists(file_path)

    return open(file_path, mode, **kwargs)


__all__ = [
    'ensure_directories_exist',
    'ensure_project_structure',
    'ensure_file_parent_exists',
    'safe_open'
]
