"""Tests for continual learning components."""

import pytest
import torch
import torch.nn as nn

from nerve_cl.continual import (
    EpisodicMemory,
    EWC,
    FOMAML,
    ContinualDistillation,
)


class TestEpisodicMemory:
    """Test episodic memory buffer."""
    
    def test_store_and_sample(self):
        memory = EpisodicMemory(capacity=100)
        
        for i in range(50):
            lr = torch.randn(3, 32, 32)
            hr = torch.randn(3, 64, 64)
            memory.store(lr, hr, {'content_type': 'test'})
        
        assert len(memory) == 50
        
        lr, hr, meta = memory.sample(batch_size=16)
        assert lr.shape[0] == 16
        assert hr.shape[0] == 16
    
    def test_capacity_limit(self):
        memory = EpisodicMemory(capacity=20)
        
        for i in range(50):
            lr = torch.randn(3, 32, 32)
            hr = torch.randn(3, 64, 64)
            memory.store(lr, hr)
        
        assert len(memory) == 20
    
    def test_stratified_sampling(self):
        memory = EpisodicMemory(capacity=100, strategy='stratified')
        
        for ct in ['sports', 'animation', 'movie']:
            for _ in range(20):
                memory.store(
                    torch.randn(3, 32, 32),
                    torch.randn(3, 64, 64),
                    {'content_type': ct}
                )
        
        stats = memory.get_stats()
        assert len(stats['content_distribution']) == 3


class TestEWC:
    """Test Elastic Weight Consolidation."""
    
    def test_register_task(self):
        model = nn.Linear(10, 10)
        ewc = EWC(model, ewc_lambda=1000)
        
        from torch.utils.data import TensorDataset, DataLoader
        data = TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
        loader = DataLoader(data, batch_size=32)
        
        ewc.register_task(0, loader)
        assert ewc.num_tasks == 1
    
    def test_penalty_increases(self):
        model = nn.Linear(10, 10)
        ewc = EWC(model, ewc_lambda=1000)
        
        from torch.utils.data import TensorDataset, DataLoader
        data = TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
        loader = DataLoader(data, batch_size=32)
        
        ewc.register_task(0, loader)
        
        penalty_before = ewc.penalty().item()
        
        # Modify weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        
        penalty_after = ewc.penalty().item()
        assert penalty_after > penalty_before


class TestMAML:
    """Test MAML meta-learning."""
    
    def test_adapt(self):
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 10))
        maml = FOMAML(model, inner_lr=0.01, inner_steps=5)
        
        data = (torch.randn(16, 10), torch.randn(16, 10))
        adapted = maml.adapt(data, nn.MSELoss())
        
        # Should be a different model
        assert adapted is not model


class TestDistillation:
    """Test knowledge distillation."""
    
    def test_compute_loss(self):
        model = nn.Linear(10, 10)
        distill = ContinualDistillation(model)
        
        inputs = torch.randn(16, 10)
        targets = torch.randn(16, 10)
        
        losses = distill.compute_loss(inputs, targets, nn.MSELoss())
        assert 'total' in losses
        assert 'task' in losses
    
    def test_register_task(self):
        model = nn.Linear(10, 10)
        distill = ContinualDistillation(model)
        
        assert distill.teacher is None
        distill.register_task()
        assert distill.teacher is not None
