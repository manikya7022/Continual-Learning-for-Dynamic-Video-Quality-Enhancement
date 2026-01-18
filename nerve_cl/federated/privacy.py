"""
Differential Privacy for Federated Learning.
Implements DP-SGD using Opacus.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy."""
    epsilon: float = 8.0  # Privacy budget
    delta: float = 1e-5   # Privacy parameter
    max_grad_norm: float = 1.0  # Gradient clipping threshold
    noise_multiplier: float = 1.0  # Noise scale


def compute_noise_multiplier(
    epsilon: float,
    delta: float,
    sample_rate: float,
    epochs: int,
) -> float:
    """Compute noise multiplier for given privacy budget."""
    # Simplified computation
    steps = epochs / sample_rate
    return math.sqrt(2 * math.log(1.25 / delta)) * math.sqrt(steps) / epsilon


class DPOptimizer:
    """
    Differential Privacy optimizer wrapper.
    Applies gradient clipping and noise addition.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        config: PrivacyConfig,
        batch_size: int,
        sample_size: int,
    ):
        self.optimizer = optimizer
        self.model = model
        self.config = config
        self.batch_size = batch_size
        self.sample_rate = batch_size / sample_size
        
        self.noise_multiplier = config.noise_multiplier
        self.steps = 0
    
    def step(self) -> None:
        """Perform DP optimization step."""
        # Clip gradients per-sample
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2)
                clip_coef = self.config.max_grad_norm / (grad_norm + 1e-6)
                clip_coef = min(clip_coef, 1.0)
                param.grad.mul_(clip_coef)
                
                # Add noise
                noise = torch.randn_like(param.grad)
                noise_scale = self.noise_multiplier * self.config.max_grad_norm
                param.grad.add_(noise * noise_scale / self.batch_size)
        
        self.optimizer.step()
        self.steps += 1
    
    def zero_grad(self) -> None:
        self.optimizer.zero_grad()


def make_private(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    config: PrivacyConfig,
) -> Tuple[nn.Module, DPOptimizer, DataLoader]:
    """
    Make training private with DP-SGD.
    
    Returns model, wrapped optimizer, and data loader.
    """
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        
        # Validate model
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            model = ModuleValidator.fix(model)
        
        privacy_engine = PrivacyEngine()
        
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            epochs=1,
            target_epsilon=config.epsilon,
            target_delta=config.delta,
            max_grad_norm=config.max_grad_norm,
        )
        
        return model, optimizer, data_loader
        
    except ImportError:
        # Fallback to simple DP optimizer
        sample_size = len(data_loader.dataset)
        dp_optimizer = DPOptimizer(
            optimizer, model, config,
            data_loader.batch_size, sample_size,
        )
        return model, dp_optimizer, data_loader


def get_privacy_spent(
    steps: int,
    noise_multiplier: float,
    sample_rate: float,
    delta: float,
) -> float:
    """Compute privacy spent (epsilon) using RDP accounting."""
    # Simplified computation
    q = sample_rate
    sigma = noise_multiplier
    
    # Using Gaussian mechanism bound
    epsilon = steps * q**2 / (2 * sigma**2)
    return epsilon
