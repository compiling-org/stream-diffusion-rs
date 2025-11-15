#!/usr/bin/env python3
"""
Test script for Python integration with Stream Diffusion RS
"""

import torch
import numpy as np
import json
import sys
import os

def test_torch():
    """Test PyTorch installation"""
    print("Testing PyTorch...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    return True

def test_diffusers():
    """Test diffusers library"""
    try:
        from diffusers import StableDiffusionPipeline
        print("Diffusers library loaded successfully")
        return True
    except ImportError as e:
        print(f"Failed to import diffusers: {e}")
        return False

def test_transformers():
    """Test transformers library"""
    try:
        from transformers import CLIPTextModel, CLIPTokenizer
        print("Transformers library loaded successfully")
        return True
    except ImportError as e:
        print(f"Failed to import transformers: {e}")
        return False

def test_numpy():
    """Test NumPy installation"""
    print("Testing NumPy...")
    print(f"NumPy version: {np.__version__}")
    
    # Create a simple array and perform operations
    arr = np.random.rand(10, 10)
    result = np.fft.fft2(arr)
    print(f"FFT operation successful, result shape: {result.shape}")
    return True

def test_eeg_processing():
    """Test EEG processing capabilities"""
    print("Testing EEG processing...")
    
    # Create dummy EEG data
    eeg_data = np.random.rand(32, 1000)  # 32 channels, 1000 time points
    
    # Extract frequency bands
    fft_data = np.fft.fft(eeg_data, axis=1)
    freqs = np.fft.fftfreq(eeg_data.shape[1], d=1/250)  # 250 Hz sampling rate
    
    # Define frequency bands
    alpha_band = (freqs >= 8) & (freqs <= 13)
    beta_band = (freqs >= 13) & (freqs <= 30)
    
    # Calculate band powers
    alpha_power = np.mean(np.abs(fft_data[:, alpha_band]), axis=1)
    beta_power = np.mean(np.abs(fft_data[:, beta_band]), axis=1)
    
    print(f"Alpha power shape: {alpha_power.shape}")
    print(f"Beta power shape: {beta_power.shape}")
    print("EEG processing successful")
    return True

def main():
    """Main test function"""
    print("Stream Diffusion RS Python Integration Test")
    print("=" * 50)
    
    tests = [
        test_numpy,
        test_torch,
        test_diffusers,
        test_transformers,
        test_eeg_processing
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"Test {test.__name__} failed: {e}")
            results.append(False)
            print()
    
    success_count = sum(results)
    total_count = len(results)
    
    print("=" * 50)
    print(f"Test Results: {success_count}/{total_count} tests passed")
    
    if success_count == total_count:
        print("All tests passed! Python integration is working correctly.")
        return 0
    else:
        print("Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())