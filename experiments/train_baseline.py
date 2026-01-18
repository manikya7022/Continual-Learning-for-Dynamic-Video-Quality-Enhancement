"""
Train NERVE Baseline Model with real dataset.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import time

from nerve_cl.models import EnhancementEngine, EnhancementConfig
from nerve_cl.models import SuperResolutionNet


def load_dataset(data_dir: str = "data"):
    """Load training and validation datasets."""
    train_data = torch.load(f"{data_dir}/train/data.pt")
    val_data = torch.load(f"{data_dir}/val/data.pt")
    
    train_dataset = TensorDataset(train_data['lr'], train_data['hr'])
    val_dataset = TensorDataset(val_data['lr'], val_data['hr'])
    
    return train_dataset, val_dataset


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between prediction and target."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def train(args):
    """Train enhancement model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading dataset...")
    train_dataset, val_dataset = load_dataset(args.data_dir)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    print("Creating model...")
    model = SuperResolutionNet(
        scale_factor=2,
        num_features=32,
        num_residual_blocks=4,
        temporal_window=1,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    
    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)
    
    best_psnr = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch_idx, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(device), hr.to(device)
            
            # Add temporal dimension (B, T, C, H, W)
            lr_temporal = lr.unsqueeze(1).expand(-1, 3, -1, -1, -1)
            
            optimizer.zero_grad()
            output = model(lr_temporal)
            loss = criterion(output, hr)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                lr_temporal = lr.unsqueeze(1).expand(-1, 3, -1, -1, -1)
                output = model(lr_temporal)
                
                val_loss += criterion(output, hr).item()
                val_psnr += compute_psnr(output, hr)
        
        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        
        scheduler.step()
        
        # Print progress
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val PSNR: {val_psnr:.2f} dB | "
              f"Time: {elapsed:.1f}s")
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
            }, 'checkpoints/best_model.pt')
    
    print("-" * 60)
    print(f"Training complete!")
    print(f"  Best PSNR: {best_psnr:.2f} dB")
    print(f"  Total time: {time.time() - start_time:.1f}s")
    print(f"  Model saved: checkpoints/best_model.pt")


def main():
    parser = argparse.ArgumentParser(description='Train NERVE baseline')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    Path('checkpoints').mkdir(exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
