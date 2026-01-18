#!/bin/bash
# Environment setup script

set -e

echo "=== NERVE-CL Environment Setup ==="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

if [[ "$python_version" < "3.9" ]]; then
    echo "Error: Python 3.9+ required"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate
echo "Virtual environment activated"

# Upgrade pip
pip install --upgrade pip

# Install package
echo "Installing NERVE-CL..."
pip install -e .

# Install dev dependencies
echo "Installing dev dependencies..."
pip install -e ".[dev]"

# Verify installation
echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

from nerve_cl.models import EnhancementEngine
print('NERVE-CL imported successfully!')
"

# Create directories
mkdir -p checkpoints
mkdir -p logs
mkdir -p data

echo "=== Setup Complete ==="
echo "To activate: source venv/bin/activate"
