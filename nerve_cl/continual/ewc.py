"""
Elastic Weight Consolidation (EWC) for Continual Learning.

Implements EWC to prevent catastrophic forgetting by constraining
important weights from changing too much on new tasks.

Reference:
    Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks"
    PNAS 2017
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Iterator
from copy import deepcopy


class EWC:
    """
    Elastic Weight Consolidation for continual learning.
    
    EWC prevents catastrophic forgetting by:
    1. Computing Fisher Information Matrix (importance of each weight)
    2. Adding regularization to penalize changes to important weights
    
    Modes:
        - 'separate': Maintain separate Fisher for each task
        - 'online': Single Fisher updated incrementally (memory efficient)
    
    Args:
        model: Neural network model
        ewc_lambda: Regularization strength (higher = more preservation)
        mode: 'separate' or 'online'
        decay: Decay factor for online mode (0-1)
    
    Example:
        >>> ewc = EWC(model, ewc_lambda=5000)
        >>> ewc.register_task(task_id=0, dataloader=train_loader)
        >>> 
        >>> # During training on new task
        >>> loss = task_loss + ewc.penalty(model)
    """
    
    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 5000.0,
        mode: str = 'online',
        decay: float = 0.999,
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.mode = mode
        self.decay = decay
        
        # Storage for Fisher Information and optimal parameters
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optpar_dict: Dict[str, torch.Tensor] = {}
        
        # Task-specific storage for 'separate' mode
        self.task_fisher: Dict[int, Dict[str, torch.Tensor]] = {}
        self.task_optpar: Dict[int, Dict[str, torch.Tensor]] = {}
        
        self.num_tasks = 0
    
    def _get_params(self) -> Iterator[tuple]:
        """Get named parameters requiring grad."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                yield name, param
    
    def compute_fisher(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None,
        empirical: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix (diagonal approximation).
        
        Fisher Information measures the importance of each parameter
        for the current task. High Fisher = important weight.
        
        Args:
            dataloader: DataLoader for current task data
            num_samples: Number of samples to use (None = all)
            empirical: If True, use empirical Fisher (gradients of loss)
        
        Returns:
            Dictionary of Fisher values per parameter
        """
        fisher = {}
        
        # Initialize Fisher
        for name, param in self._get_params():
            fisher[name] = torch.zeros_like(param)
        
        self.model.eval()
        num_samples_used = 0
        
        for batch in dataloader:
            if num_samples is not None and num_samples_used >= num_samples:
                break
            
            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
                if len(batch) > 1:
                    targets = batch[1]
                else:
                    targets = None
            else:
                inputs = batch
                targets = None
            
            inputs = inputs.to(next(self.model.parameters()).device)
            
            self.model.zero_grad()
            
            if empirical and targets is not None:
                # Empirical Fisher: use actual loss
                targets = targets.to(inputs.device)
                outputs = self.model(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
                loss.backward()
            else:
                # True Fisher: use log-likelihood
                outputs = self.model(inputs)
                # Sample from output distribution
                if outputs.dim() > 1:
                    # For image outputs, use per-pixel log-likelihood
                    log_prob = -0.5 * (outputs ** 2).sum()
                else:
                    log_prob = outputs.sum()
                log_prob.backward()
            
            # Accumulate squared gradients
            for name, param in self._get_params():
                if param.grad is not None:
                    fisher[name] += param.grad.data.clone() ** 2
            
            num_samples_used += inputs.size(0)
        
        # Normalize
        for name in fisher:
            fisher[name] /= max(num_samples_used, 1)
        
        return fisher
    
    def register_task(
        self,
        task_id: int,
        dataloader: DataLoader,
        num_samples: Optional[int] = None,
    ) -> None:
        """
        Register a completed task for EWC protection.
        
        Call this after training on a task to protect learned weights.
        
        Args:
            task_id: Unique task identifier
            dataloader: DataLoader for the task
            num_samples: Number of samples for Fisher computation
        """
        # Compute Fisher for this task
        fisher = self.compute_fisher(dataloader, num_samples)
        
        # Store optimal parameters
        optpar = {}
        for name, param in self._get_params():
            optpar[name] = param.data.clone()
        
        if self.mode == 'separate':
            # Store per-task
            self.task_fisher[task_id] = fisher
            self.task_optpar[task_id] = optpar
        
        elif self.mode == 'online':
            # Update running Fisher with decay
            if len(self.fisher_dict) == 0:
                self.fisher_dict = fisher
                self.optpar_dict = optpar
            else:
                for name in fisher:
                    self.fisher_dict[name] = (
                        self.decay * self.fisher_dict[name] +
                        (1 - self.decay) * fisher[name]
                    )
                self.optpar_dict = optpar
        
        self.num_tasks += 1
    
    def penalty(self, model: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Compute EWC penalty for current model parameters.
        
        Penalty = λ/2 * Σ F_i * (θ_i - θ*_i)²
        
        Args:
            model: Model to compute penalty for (default: self.model)
        
        Returns:
            EWC penalty term (add to loss)
        """
        if model is None:
            model = self.model
        
        penalty = 0.0
        device = next(model.parameters()).device
        
        if self.mode == 'separate':
            # Sum penalties across all tasks
            for task_id in self.task_fisher:
                fisher = self.task_fisher[task_id]
                optpar = self.task_optpar[task_id]
                
                for name, param in model.named_parameters():
                    if name in fisher:
                        f = fisher[name].to(device)
                        o = optpar[name].to(device)
                        penalty += (f * (param - o) ** 2).sum()
        
        elif self.mode == 'online':
            for name, param in model.named_parameters():
                if name in self.fisher_dict:
                    f = self.fisher_dict[name].to(device)
                    o = self.optpar_dict[name].to(device)
                    penalty += (f * (param - o) ** 2).sum()
        
        return self.ewc_lambda / 2 * penalty
    
    def get_importance_stats(self) -> Dict[str, float]:
        """Get statistics about parameter importance."""
        if self.mode == 'online':
            fisher = self.fisher_dict
        else:
            # Combine all task Fishers
            fisher = {}
            for task_fisher in self.task_fisher.values():
                for name, f in task_fisher.items():
                    if name not in fisher:
                        fisher[name] = f.clone()
                    else:
                        fisher[name] += f
        
        stats = {}
        for name, f in fisher.items():
            stats[name] = {
                'mean': f.mean().item(),
                'max': f.max().item(),
                'std': f.std().item(),
                'nonzero': (f > 0).float().mean().item(),
            }
        
        return stats
    
    def state_dict(self) -> Dict:
        """Get state for saving."""
        return {
            'ewc_lambda': self.ewc_lambda,
            'mode': self.mode,
            'decay': self.decay,
            'num_tasks': self.num_tasks,
            'fisher_dict': {k: v.cpu() for k, v in self.fisher_dict.items()},
            'optpar_dict': {k: v.cpu() for k, v in self.optpar_dict.items()},
            'task_fisher': {
                t: {k: v.cpu() for k, v in f.items()}
                for t, f in self.task_fisher.items()
            },
            'task_optpar': {
                t: {k: v.cpu() for k, v in o.items()}
                for t, o in self.task_optpar.items()
            },
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """Load state from checkpoint."""
        self.ewc_lambda = state['ewc_lambda']
        self.mode = state['mode']
        self.decay = state['decay']
        self.num_tasks = state['num_tasks']
        self.fisher_dict = state['fisher_dict']
        self.optpar_dict = state['optpar_dict']
        self.task_fisher = state['task_fisher']
        self.task_optpar = state['task_optpar']


class OnlineEWC(EWC):
    """
    Convenience class for Online EWC.
    
    More memory efficient - doesn't store per-task Fisher.
    """
    
    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 5000.0,
        decay: float = 0.999,
    ):
        super().__init__(model, ewc_lambda, mode='online', decay=decay)


class SynapticIntelligence:
    """
    Synaptic Intelligence (SI) - online importance estimation.
    
    Tracks parameter importance during training, not just at task end.
    More efficient than EWC for some scenarios.
    
    Reference:
        Zenke et al., "Continual Learning Through Synaptic Intelligence"
    """
    
    def __init__(
        self,
        model: nn.Module,
        si_lambda: float = 1.0,
        damping: float = 0.1,
    ):
        self.model = model
        self.si_lambda = si_lambda
        self.damping = damping
        
        # Track parameter changes and gradients
        self.W: Dict[str, torch.Tensor] = {}  # Running importance
        self.p_old: Dict[str, torch.Tensor] = {}  # Parameters at task start
        self.omega: Dict[str, torch.Tensor] = {}  # Accumulated importance
        
        self._init_tracking()
    
    def _init_tracking(self) -> None:
        """Initialize tracking variables."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.W[name] = torch.zeros_like(param)
                self.p_old[name] = param.data.clone()
                self.omega[name] = torch.zeros_like(param)
    
    def update_importance(self) -> None:
        """
        Update running importance during training.
        Call after each optimizer step.
        """
        for name, param in self.model.named_parameters():
            if name in self.W and param.grad is not None:
                # Importance = gradient * parameter change direction
                delta = param.data - self.p_old[name]
                self.W[name] += -param.grad.data * delta
                self.p_old[name] = param.data.clone()
    
    def register_task(self) -> None:
        """Register completed task."""
        for name, param in self.model.named_parameters():
            if name in self.W:
                # Normalize importance
                delta = param.data - self.p_old[name]
                denom = delta ** 2 + self.damping
                
                self.omega[name] += self.W[name] / denom
                
                # Reset for next task
                self.W[name] = torch.zeros_like(param)
                self.p_old[name] = param.data.clone()
    
    def penalty(self) -> torch.Tensor:
        """Compute SI penalty."""
        penalty = 0.0
        device = next(self.model.parameters()).device
        
        for name, param in self.model.named_parameters():
            if name in self.omega:
                o = self.omega[name].to(device)
                p_old = self.p_old[name].to(device)
                penalty += (o * (param - p_old) ** 2).sum()
        
        return self.si_lambda * penalty


if __name__ == "__main__":
    # Test EWC
    print("Testing EWC...")
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    
    ewc = EWC(model, ewc_lambda=5000, mode='online')
    
    # Create dummy dataloader
    from torch.utils.data import TensorDataset, DataLoader
    
    X = torch.randn(100, 10)
    Y = torch.randn(100, 10)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=32)
    
    # Register task
    ewc.register_task(task_id=0, dataloader=loader)
    print(f"Registered task 0, num_tasks: {ewc.num_tasks}")
    
    # Compute penalty
    penalty = ewc.penalty()
    print(f"Initial penalty: {penalty.item():.6f}")
    
    # Modify weights and check penalty increases
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    
    penalty_after = ewc.penalty()
    print(f"Penalty after weight change: {penalty_after.item():.6f}")
    
    # Importance stats
    stats = ewc.get_importance_stats()
    for name, s in list(stats.items())[:2]:
        print(f"  {name}: mean={s['mean']:.6f}, max={s['max']:.6f}")
