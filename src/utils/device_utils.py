"""
Device detection and management utilities for cross-platform GPU support.
"""

import logging
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_available_devices() -> Dict[str, Any]:
    """
    Detect and return information about available compute devices.

    Returns:
        Dictionary containing device availability and specifications
    """
    devices = {
        'cpu': {
            'available': True,
            'name': 'CPU',
            'device': 'cpu'
        },
        'cuda': {
            'available': False,
            'name': None,
            'device': None,
            'count': 0,
            'devices': []
        },
        'mps': {
            'available': False,
            'name': None,
            'device': None
        }
    }

    # Check CUDA availability
    if torch.cuda.is_available():
        devices['cuda']['available'] = True
        devices['cuda']['count'] = torch.cuda.device_count()
        devices['cuda']['device'] = 'cuda'

        # Get information for each CUDA device
        for i in range(torch.cuda.device_count()):
            device_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'capability': torch.cuda.get_device_capability(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
            }
            devices['cuda']['devices'].append(device_info)

        # Set primary device name
        if devices['cuda']['devices']:
            devices['cuda']['name'] = devices['cuda']['devices'][0]['name']

    # Check MPS (Metal Performance Shaders for Mac) availability
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices['mps']['available'] = True
        devices['mps']['name'] = 'Apple Metal Performance Shaders (MPS)'
        devices['mps']['device'] = 'mps'

    return devices


def get_optimal_device(prefer_cuda: bool = True, prefer_mps: bool = True) -> torch.device:
    """
    Get the optimal device for computation.

    Args:
        prefer_cuda: Whether to prefer CUDA over MPS if both are available
        prefer_mps: Whether to use MPS if available and CUDA is not

    Returns:
        torch.device: Optimal device for computation
    """
    # Check CUDA
    if torch.cuda.is_available() and prefer_cuda:
        device = torch.device('cuda')
        logger.info(f"Selected CUDA device: {torch.cuda.get_device_name(0)}")
        return device

    # Check MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and prefer_mps:
        device = torch.device('mps')
        logger.info("Selected Apple MPS device")
        return device

    # Fallback to CPU
    logger.info("Selected CPU device")
    return torch.device('cpu')


def print_device_info(verbose: bool = True) -> None:
    """
    Print information about available devices.

    Args:
        verbose: Whether to print detailed information
    """
    devices = get_available_devices()

    print("\n" + "="*70)
    print("AVAILABLE COMPUTE DEVICES")
    print("="*70)

    # CPU info
    print("\n[CPU]")
    print("  ✓ Available (always)")

    # CUDA info
    print("\n[NVIDIA CUDA]")
    if devices['cuda']['available']:
        print(f"  ✓ Available ({devices['cuda']['count']} device(s))")
        if verbose:
            for i, dev in enumerate(devices['cuda']['devices']):
                print(f"\n  Device {i}: {dev['name']}")
                print(f"    - Compute Capability: {dev['capability'][0]}.{dev['capability'][1]}")
                print(f"    - Total Memory: {dev['memory_total']:.2f} GB")
    else:
        print("  ✗ Not available")

    # MPS info
    print("\n[Apple Metal Performance Shaders (MPS)]")
    if devices['mps']['available']:
        print("  ✓ Available")
        print("    - Apple Silicon GPU acceleration enabled")
        print("    - Supports M1, M2, M3+ chips")
    else:
        print("  ✗ Not available")
        if not hasattr(torch.backends, 'mps'):
            print("    - PyTorch version may not support MPS")
            print("    - Requires PyTorch 1.12+ on macOS 12.3+")

    print("\n" + "="*70)

    # Print recommended device
    optimal = get_optimal_device()
    print(f"\nRecommended device: {optimal}")
    print("="*70 + "\n")


def configure_device_for_training(config: Dict[str, Any]) -> torch.device:
    """
    Configure and return the appropriate device based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        torch.device: Configured device
    """
    gpu_enabled = config.get('execution', {}).get('gpu_enabled', True)
    cuda_enabled = config.get('execution', {}).get('cuda_enabled', True)
    mps_enabled = config.get('execution', {}).get('mps_enabled', True)

    if not gpu_enabled:
        logger.info("GPU acceleration disabled in config")
        return torch.device('cpu')

    # Try CUDA first
    if torch.cuda.is_available() and cuda_enabled:
        device = torch.device('cuda')
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")

        # Set memory allocation strategy
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            # Reserve 90% of GPU memory
            torch.cuda.set_per_process_memory_fraction(0.9)

        return device

    # Try MPS if CUDA not available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and mps_enabled:
        device = torch.device('mps')
        logger.info("Using Apple MPS for GPU acceleration")

        # Set MPS optimization flags if available
        if hasattr(torch.backends.mps, 'is_built'):
            logger.info(f"MPS built: {torch.backends.mps.is_built()}")

        return device

    # Fallback to CPU
    logger.info("No GPU available, using CPU")
    return torch.device('cpu')


def optimize_for_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Apply device-specific optimizations to the model.

    Args:
        model: PyTorch model
        device: Target device

    Returns:
        Optimized model
    """
    model = model.to(device)

    if device.type == 'cuda':
        # Enable cuDNN benchmarking for optimal performance
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmarking for CUDA")

        # Enable TF32 on Ampere+ GPUs for better performance
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for Ampere+ GPU")

    elif device.type == 'mps':
        # MPS-specific optimizations
        logger.info("Applied MPS optimizations")

        # Note: MPS doesn't support all operations yet
        # Some operations may fall back to CPU
        logger.warning("Note: Some operations may fall back to CPU on MPS")

    return model


def check_device_compatibility(operation: str = None) -> Dict[str, bool]:
    """
    Check compatibility of various operations across devices.

    Args:
        operation: Specific operation to check (optional)

    Returns:
        Dictionary of operation compatibility across devices
    """
    compatibility = {
        'cuda': {
            'available': torch.cuda.is_available(),
            'mixed_precision': torch.cuda.is_available(),
            'distributed': torch.cuda.is_available(),
        },
        'mps': {
            'available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'mixed_precision': False,  # MPS doesn't support AMP yet
            'distributed': False,  # MPS doesn't support distributed training
        }
    }

    return compatibility


if __name__ == "__main__":
    # Print device information when run as a script
    print_device_info(verbose=True)

    # Test device selection
    print("\nTesting device selection:")
    optimal = get_optimal_device()
    print(f"Optimal device: {optimal}")

    # Check compatibility
    print("\nDevice compatibility check:")
    compat = check_device_compatibility()
    print(f"CUDA compatibility: {compat['cuda']}")
    print(f"MPS compatibility: {compat['mps']}")
