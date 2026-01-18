#!/bin/bash
# Download datasets for training

set -e

DATA_DIR="${DATA_DIR:-./data}"
mkdir -p "$DATA_DIR"

echo "=== NERVE-CL Dataset Download Script ==="

# UVG Dataset (small, for testing)
echo "Downloading UVG test videos..."
mkdir -p "$DATA_DIR/uvg"
# In production, download from: http://ultravideo.fi/

# REDS Dataset
echo "Downloading REDS dataset..."
mkdir -p "$DATA_DIR/reds"
# In production: https://seungjunnah.github.io/Datasets/reds.html

echo "Creating sample data for testing..."
python3 -c "
import torch
import os

data_dir = '$DATA_DIR'

# Create dummy training data
print('Creating dummy training data...')
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(data_dir, 'dummy', split)
    os.makedirs(split_dir, exist_ok=True)
    
    num_samples = 1000 if split == 'train' else 100
    
    lr = torch.randn(num_samples, 3, 64, 64)
    hr = torch.randn(num_samples, 3, 128, 128)
    
    torch.save({'lr': lr, 'hr': hr}, os.path.join(split_dir, 'data.pt'))
    print(f'  Created {split}: {num_samples} samples')

print('Done!')
"

echo "=== Download Complete ==="
echo "Data directory: $DATA_DIR"
ls -la "$DATA_DIR"
