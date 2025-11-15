# Python Integration for Stream Diffusion RS

This document explains how to set up and use the Python integration features in Stream Diffusion RS.

## Overview

Stream Diffusion RS includes Python interop capabilities that allow the Rust application to leverage Python-based AI/ML libraries such as:

- PyTorch for deep learning models
- Diffusers for diffusion models
- Transformers for NLP models
- MNE/SciPy for EEG processing
- NumPy for numerical computations

## Setup Instructions

### Prerequisites

1. Python 3.8 or later
2. pip package manager
3. Virtual environment support (venv module)

### Automatic Setup

Run the setup script for your platform:

**Linux/macOS:**
```bash
./scripts/setup_python_env.sh
```

**Windows:**
```cmd
scripts\setup_python_env.bat
```

### Manual Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   # Linux/macOS
   source venv/bin/activate
   
   # Windows
   venv\Scripts\activate.bat
   ```

3. Upgrade pip:
   ```bash
   pip install --upgrade pip
   ```

4. Install dependencies:
   ```bash
   pip install -r scripts/requirements.txt
   ```

## Testing the Integration

After setting up the Python environment, you can test the integration:

1. Activate the virtual environment
2. Run the Python test script:
   ```bash
   python scripts/test_python_integration.py
   ```

3. Run the Rust test binary:
   ```bash
   cargo run --bin test_python_integration
   ```

## Usage Examples

### Image Generation

The Python integration allows for real image generation using Stable Diffusion models:

```python
from diffusers import StableDiffusionPipeline
import torch

# Load model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Generate image
image = pipe("a beautiful landscape").images[0]
```

### EEG Processing

The integration also supports EEG data processing:

```python
import numpy as np
from scipy import signal

# Process EEG data
eeg_data = np.load("eeg_data.npy")
filtered_data = signal.butterworth(eeg_data, 4, [8, 13], btype='band')
```

## API Reference

### PythonEnvironment

Manages the Python execution environment.

Methods:
- `new(python_path: &str)` - Create new environment
- `install_dependencies()` - Install required packages
- `execute_script(script: &str, args: Option<&[&str]>)` - Execute Python script
- `execute_script_file(script_path: &Path, args: Option<&[&str]>)` - Execute Python script file

### PythonModel

Interface for Python-based AI/ML models.

Methods:
- `new(model_path: &str, model_type: &str, python_env: PythonEnvironment)` - Create new model
- `load_model()` - Load model
- `generate_image(prompt: &str, steps: usize)` - Generate image from prompt
- `process_eeg(eeg_data_path: &str)` - Process EEG data

## Troubleshooting

### Common Issues

1. **Python not found**: Ensure Python 3.8+ is installed and in PATH
2. **Import errors**: Make sure all dependencies are installed
3. **CUDA issues**: Verify CUDA drivers and PyTorch CUDA version compatibility

### Debugging

Enable verbose logging by setting the environment variable:
```bash
export RUST_LOG=debug
```

## Contributing

To extend the Python integration:

1. Add new methods to the `PythonModel` struct in `src/python.rs`
2. Create corresponding Python functions in `scripts/diffusion_model.py`
3. Update the requirements.txt file if new dependencies are needed
4. Add tests to verify new functionality

## License

This project is licensed under the MIT License. See the LICENSE file for details.