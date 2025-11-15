#!/bin/bash

# Setup Python environment for Stream Diffusion RS
# This script creates a virtual environment and installs required dependencies

set -e

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

# Check if virtualenv is installed
if ! python3 -c "import venv" &> /dev/null; then
    echo "Python venv module is not available. Please install python3-venv package."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r scripts/requirements.txt

# Download pre-trained models (optional)
echo "Setup complete! To activate the virtual environment, run:"
echo "source venv/bin/activate"