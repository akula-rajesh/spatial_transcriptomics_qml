#!/usr/bin/env python3
"""
Test script to verify GPU support (CUDA, MPS, CPU) across platforms.
Run this to ensure device detection and model initialization works correctly.
"""

import sys
import torch
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.device_utils import (
    get_available_devices,
    get_optimal_device,
    print_device_info,
    configure_device_for_training,
    optimize_for_device,
    check_device_compatibility
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_device_detection():
    """Test device detection functionality."""
    print("\n" + "="*70)
    print("TEST 1: Device Detection")
    print("="*70)

    devices = get_available_devices()

    print(f"\nCPU Available: {devices['cpu']['available']}")
    print(f"CUDA Available: {devices['cuda']['available']}")
    if devices['cuda']['available']:
        print(f"  - Device Count: {devices['cuda']['count']}")
        for i, dev in enumerate(devices['cuda']['devices']):
            print(f"  - GPU {i}: {dev['name']} ({dev['memory_total']:.1f} GB)")

    print(f"MPS Available: {devices['mps']['available']}")
    if devices['mps']['available']:
        print(f"  - {devices['mps']['name']}")

    return devices


def test_optimal_device():
    """Test optimal device selection."""
    print("\n" + "="*70)
    print("TEST 2: Optimal Device Selection")
    print("="*70)

    # Test default selection
    device = get_optimal_device()
    print(f"\nDefault selection: {device}")

    # Test CUDA preference
    device_cuda = get_optimal_device(prefer_cuda=True, prefer_mps=False)
    print(f"CUDA preferred: {device_cuda}")

    # Test MPS preference
    device_mps = get_optimal_device(prefer_cuda=False, prefer_mps=True)
    print(f"MPS preferred: {device_mps}")

    return device


def test_config_based_selection():
    """Test configuration-based device selection."""
    print("\n" + "="*70)
    print("TEST 3: Configuration-Based Selection")
    print("="*70)

    # Test with all enabled
    config_all = {
        'execution': {
            'gpu_enabled': True,
            'cuda_enabled': True,
            'mps_enabled': True
        }
    }
    device = configure_device_for_training(config_all)
    print(f"\nAll enabled: {device}")

    # Test with GPU disabled
    config_cpu = {
        'execution': {
            'gpu_enabled': False,
            'cuda_enabled': True,
            'mps_enabled': True
        }
    }
    device = configure_device_for_training(config_cpu)
    print(f"GPU disabled: {device}")

    # Test with only CUDA
    config_cuda = {
        'execution': {
            'gpu_enabled': True,
            'cuda_enabled': True,
            'mps_enabled': False
        }
    }
    device = configure_device_for_training(config_cuda)
    print(f"Only CUDA: {device}")

    # Test with only MPS
    config_mps = {
        'execution': {
            'gpu_enabled': True,
            'cuda_enabled': False,
            'mps_enabled': True
        }
    }
    device = configure_device_for_training(config_mps)
    print(f"Only MPS: {device}")


def test_model_on_device():
    """Test creating a model on the detected device."""
    print("\n" + "="*70)
    print("TEST 4: Model Creation and Device Placement")
    print("="*70)

    device = get_optimal_device()
    print(f"\nUsing device: {device}")

    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(224*224*3, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 250)
    )

    # Move to device
    model = model.to(device)
    print(f"Model created and moved to {device}")

    # Create sample input
    batch_size = 4
    sample_input = torch.randn(batch_size, 224*224*3).to(device)
    print(f"Sample input shape: {sample_input.shape} on {sample_input.device}")

    # Test forward pass
    try:
        with torch.no_grad():
            output = model(sample_input)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape} on {output.device}")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_device_optimization():
    """Test device-specific optimizations."""
    print("\n" + "="*70)
    print("TEST 5: Device-Specific Optimizations")
    print("="*70)

    device = get_optimal_device()

    # Create model
    model = torch.nn.Linear(100, 50)

    # Apply optimizations
    model = optimize_for_device(model, device)
    print(f"\n✓ Applied optimizations for {device}")

    if device.type == 'cuda':
        print(f"  - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        if torch.cuda.get_device_capability()[0] >= 8:
            print(f"  - TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    elif device.type == 'mps':
        print(f"  - MPS optimizations applied")

    return model


def test_compatibility():
    """Test device compatibility checks."""
    print("\n" + "="*70)
    print("TEST 6: Device Compatibility Check")
    print("="*70)

    compat = check_device_compatibility()

    print("\nCUDA Compatibility:")
    for feature, available in compat['cuda'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {feature}: {available}")

    print("\nMPS Compatibility:")
    for feature, available in compat['mps'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {feature}: {available}")


def test_memory_allocation():
    """Test memory allocation on device."""
    print("\n" + "="*70)
    print("TEST 7: Memory Allocation Test")
    print("="*70)

    device = get_optimal_device()

    try:
        # Allocate tensors of increasing size
        sizes = [1, 10, 100, 1000]
        for size in sizes:
            tensor = torch.randn(size, size).to(device)
            print(f"✓ Allocated {size}x{size} tensor on {device}")
            del tensor

        # Clear cache if CUDA
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print("✓ Cleared CUDA cache")

        return True
    except Exception as e:
        print(f"✗ Memory allocation failed: {e}")
        return False


def run_all_tests():
    """Run all device tests."""
    print("\n")
    print("#" * 70)
    print("# GPU SUPPORT TEST SUITE")
    print("# Testing CUDA, MPS, and CPU device support")
    print("#" * 70)

    # Print PyTorch version info
    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version}")

    # Run tests
    try:
        test_device_detection()
        test_optimal_device()
        test_config_based_selection()
        model_test_passed = test_model_on_device()
        test_device_optimization()
        test_compatibility()
        memory_test_passed = test_memory_allocation()

        # Print detailed device info
        print("\n" + "="*70)
        print("DETAILED DEVICE INFORMATION")
        print("="*70)
        print_device_info(verbose=True)

        # Summary
        print("\n" + "#" * 70)
        print("# TEST SUMMARY")
        print("#" * 70)
        print(f"\nModel Test: {'✓ PASSED' if model_test_passed else '✗ FAILED'}")
        print(f"Memory Test: {'✓ PASSED' if memory_test_passed else '✗ FAILED'}")

        if model_test_passed and memory_test_passed:
            print("\n✓ All tests PASSED - Device support is working correctly!")
            return 0
        else:
            print("\n✗ Some tests FAILED - Please check the output above")
            return 1

    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
