#!/usr/bin/env python3
"""
Quick test to verify directory structure is created correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.directory_utils import ensure_project_structure

def test_directory_creation():
    """Test that all necessary directories are created."""
    print("Testing directory creation...")

    # Ensure project structure
    ensure_project_structure(project_root)

    # Check that critical directories exist
    critical_dirs = [
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

    print("\nVerifying directories:")
    all_exist = True
    for dir_name in critical_dirs:
        dir_path = project_root / dir_name
        exists = dir_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_name}: {dir_path}")
        if not exists:
            all_exist = False

    if all_exist:
        print("\n✓ All directories exist!")
        return 0
    else:
        print("\n✗ Some directories are missing!")
        return 1

if __name__ == "__main__":
    exit_code = test_directory_creation()
    sys.exit(exit_code)
