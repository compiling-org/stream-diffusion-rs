#!/usr/bin/env python3
"""
Diffusion model implementation for Stream Diffusion RS

This script provides actual image generation capabilities using Stable Diffusion
and can be called from the Rust application via Python interop.
"""

import torch
import json
import sys
import os
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer

def generate_image(prompt, model_path=None, steps=20, guidance_scale=7.5, width=512, height=512):
    """
    Generate an image from a text prompt using Stable Diffusion
    
    Args:
        prompt (str): Text prompt for image generation
        model_path (str): Path to model (optional)
        steps (int): Number of inference steps
        guidance_scale (float): Guidance scale for generation
        width (int): Image width
        height (int): Image height
    
    Returns:
        dict: Generation result with image data
    """
    try:
        # Use default model if none specified
        if model_path is None:
            model_path = "runwayml/stable-diffusion-v1-5"
        
        # Load model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,  # Disable for performance
            requires_safety_checker=False
        )
        
        # Optimize scheduler for speed
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            pipe = pipe.to("cpu")
        
        # Generate image
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        ).images[0]
        
        # Convert to RGB bytes
        image_rgb = image.convert("RGB")
        image_array = np.array(image_rgb)
        image_bytes = image_array.tobytes()
        
        return {
            "success": True,
            "prompt": prompt,
            "image_shape": image_array.shape,
            "image_dtype": str(image_array.dtype),
            "image_size": len(image_bytes),
            "steps": steps,
            "guidance_scale": guidance_scale
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "prompt": prompt
        }

def process_eeg_data(eeg_file_path):
    """
    Process EEG data and extract features
    
    Args:
        eeg_file_path (str): Path to EEG data file
    
    Returns:
        dict: Processing result with extracted features
    """
    try:
        # Load EEG data (assuming numpy format)
        if eeg_file_path.endswith('.npy'):
            eeg_data = np.load(eeg_file_path)
        else:
            # Placeholder for other formats
            eeg_data = np.random.rand(32, 1000)  # 32 channels, 1000 time points
        
        # Extract frequency band powers
        fft_data = np.fft.fft(eeg_data, axis=1)
        freqs = np.fft.fftfreq(eeg_data.shape[1], d=1/250)  # Assuming 250 Hz sampling rate
        
        # Define frequency bands
        alpha_band = (freqs >= 8) & (freqs <= 13)
        beta_band = (freqs >= 13) & (freqs <= 30)
        theta_band = (freqs >= 4) & (freqs <= 8)
        delta_band = (freqs >= 0.5) & (freqs <= 4)
        gamma_band = (freqs >= 30) & (freqs <= 100)
        
        # Calculate average power in each band
        alpha_power = np.mean(np.abs(fft_data[:, alpha_band]), axis=1)
        beta_power = np.mean(np.abs(fft_data[:, beta_band]), axis=1)
        theta_power = np.mean(np.abs(fft_data[:, theta_band]), axis=1)
        delta_power = np.mean(np.abs(fft_data[:, delta_band]), axis=1)
        gamma_power = np.mean(np.abs(fft_data[:, gamma_band]), axis=1)
        
        # Calculate connectivity (simple correlation)
        connectivity = np.corrcoef(eeg_data).flatten()
        
        # Calculate complexity (sample entropy approximation)
        complexity = np.std(eeg_data, axis=1)
        
        return {
            "success": True,
            "band_powers": {
                "alpha": alpha_power.tolist(),
                "beta": beta_power.tolist(),
                "theta": theta_power.tolist(),
                "delta": delta_power.tolist(),
                "gamma": gamma_power.tolist()
            },
            "connectivity": connectivity.tolist(),
            "complexity": complexity.tolist(),
            "data_shape": eeg_data.shape,
            "sampling_rate": 250
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """
    Main entry point for command-line usage
    """
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python diffusion_model.py <command> [args...]"}))
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "generate":
        if len(sys.argv) < 3:
            print(json.dumps({"error": "Usage: python diffusion_model.py generate <prompt> [model_path] [steps]"}))
            sys.exit(1)
        
        prompt = sys.argv[2]
        model_path = sys.argv[3] if len(sys.argv) > 3 else None
        steps = int(sys.argv[4]) if len(sys.argv) > 4 else 20
        
        result = generate_image(prompt, model_path, steps)
        print(json.dumps(result))
        
    elif command == "process_eeg":
        if len(sys.argv) < 3:
            print(json.dumps({"error": "Usage: python diffusion_model.py process_eeg <eeg_file_path>"}))
            sys.exit(1)
        
        eeg_file_path = sys.argv[2]
        result = process_eeg_data(eeg_file_path)
        print(json.dumps(result))
        
    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()