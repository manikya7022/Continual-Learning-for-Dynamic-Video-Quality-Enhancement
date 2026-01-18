"""
Continual Learning Training Script.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from nerve_cl.models import EnhancementEngine, EnhancementConfig
from nerve_cl.continual import EpisodicMemory, EWC, FOMAML, ContinualDistillation


def create_task_data(content_type: str, num_samples: int = 100):
    """Create dummy task data."""
    # Simulate different content types with different statistics
    offsets = {'sports': 0.2, 'animation': -0.2, 'movie': 0.0, 'news': 0.1}
    offset = offsets.get(content_type, 0)
    
    lr = torch.randn(num_samples, 3, 64, 64) + offset
    hr = torch.randn(num_samples, 3, 128, 128) + offset
    return lr, hr


def train_with_ewc(model, tasks, config):
    """Train with Elastic Weight Consolidation."""
    device = next(model.parameters()).device
    ewc = EWC(model, ewc_lambda=config.get('ewc_lambda', 5000))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for task_id, (task_name, task_data) in enumerate(tasks):
        print(f"\n=== Training on Task {task_id}: {task_name} ===")
        
        lr, hr = task_data
        dataset = TensorDataset(lr, hr)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        for epoch in range(5):
            model.train()
            total_loss = 0.0
            
            for lr_batch, hr_batch in loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                
                optimizer.zero_grad()
                
                # Simplified forward
                lr_temporal = lr_batch.unsqueeze(1).expand(-1, 3, -1, -1, -1)
                output = model(lr_temporal)['enhanced']
                
                # Task loss + EWC penalty
                task_loss = criterion(output, hr_batch)
                ewc_loss = ewc.penalty()
                loss = task_loss + ewc_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"  Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}")
        
        # Register task for EWC
        ewc.register_task(task_id, loader)
        print(f"  Registered task {task_id} for EWC protection")
    
    return model


def train_with_replay(model, tasks, memory, config):
    """Train with experience replay."""
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for task_id, (task_name, task_data) in enumerate(tasks):
        print(f"\n=== Training on Task {task_id}: {task_name} ===")
        
        lr, hr = task_data
        
        for epoch in range(5):
            model.train()
            
            # Current task data
            indices = torch.randperm(len(lr))[:16]
            lr_batch = lr[indices].to(device)
            hr_batch = hr[indices].to(device)
            
            # Replay data
            if len(memory) > 0:
                replay_lr, replay_hr, _ = memory.sample(batch_size=8, device=device)
                lr_batch = torch.cat([lr_batch, replay_lr])
                hr_batch = torch.cat([hr_batch, replay_hr])
            
            optimizer.zero_grad()
            lr_temporal = lr_batch.unsqueeze(1).expand(-1, 3, -1, -1, -1)
            output = model(lr_temporal)['enhanced']
            loss = criterion(output, hr_batch)
            loss.backward()
            optimizer.step()
            
            print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}")
        
        # Store samples in memory
        for i in range(min(50, len(lr))):
            memory.store(lr[i], hr[i], metadata={'content_type': task_name})
        
        print(f"  Memory size: {len(memory)}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', choices=['ewc', 'replay', 'maml'], default='ewc')
    parser.add_argument('--memory-size', type=int, default=200)
    parser.add_argument('--ewc-lambda', type=float, default=5000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = EnhancementEngine(EnhancementConfig(
        frame_recovery_enabled=False,
        super_resolution_enabled=True,
    )).to(device)
    
    # Create task sequence
    content_types = ['sports', 'animation', 'movie', 'news']
    tasks = [(ct, create_task_data(ct, 200)) for ct in content_types]
    
    config = {'ewc_lambda': args.ewc_lambda}
    
    if args.strategy == 'ewc':
        model = train_with_ewc(model, tasks, config)
    elif args.strategy == 'replay':
        memory = EpisodicMemory(capacity=args.memory_size, strategy='stratified')
        model = train_with_replay(model, tasks, memory, config)
    
    # Save model
    Path('checkpoints').mkdir(exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/continual_model.pt')
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
