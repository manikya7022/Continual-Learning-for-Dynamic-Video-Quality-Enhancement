"""
Model-Agnostic Meta-Learning (MAML) for Video Enhancement.

Enables fast adaptation to new content types with few gradient steps.

Reference:
    Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation"
    ICML 2017
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Optional, Tuple
from copy import deepcopy
try:
    import higher  # For differentiable optimization (optional)
    HIGHER_AVAILABLE = True
except ImportError:
    HIGHER_AVAILABLE = False


class MAML:
    """
    Model-Agnostic Meta-Learning for fast adaptation.
    
    MAML trains a model initialization that can be quickly adapted
    to new tasks (content types) with just a few gradient steps.
    
    For video enhancement:
        - Each "task" is a content type (sports, animation, etc.)
        - Inner loop: Adapt to specific content type
        - Outer loop: Learn good initialization for all types
    
    Args:
        model: Neural network model
        inner_lr: Learning rate for inner loop adaptation
        outer_lr: Learning rate for meta-update (outer loop)
        inner_steps: Number of gradient steps in inner loop
        first_order: If True, use first-order approximation (faster)
    
    Example:
        >>> maml = MAML(model, inner_lr=0.01, inner_steps=5)
        >>> 
        >>> # Meta-training
        >>> for task_batch in meta_dataloader:
        ...     meta_loss = maml.meta_step(task_batch)
        ...     meta_loss.backward()
        >>>
        >>> # Fast adaptation to new task
        >>> adapted_model = maml.adapt(new_task_data, steps=5)
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        first_order: bool = True,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
        # Outer optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=outer_lr,
        )
    
    def _inner_loop(
        self,
        model: nn.Module,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable,
        steps: Optional[int] = None,
    ) -> nn.Module:
        """
        Perform inner loop adaptation.
        
        Args:
            model: Model to adapt
            support_data: (inputs, targets) for adaptation
            loss_fn: Loss function
            steps: Number of gradient steps (default: self.inner_steps)
        
        Returns:
            Adapted model
        """
        steps = steps or self.inner_steps
        inputs, targets = support_data
        
        # Create a copy for adaptation
        adapted_model = deepcopy(model)
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.inner_lr,
        )
        
        for _ in range(steps):
            inner_optimizer.zero_grad()
            outputs = adapted_model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            inner_optimizer.step()
        
        return adapted_model
    
    def meta_step(
        self,
        task_batch: List[Dict],
        loss_fn: Callable,
    ) -> torch.Tensor:
        """
        Perform one meta-learning step on a batch of tasks.
        
        Args:
            task_batch: List of task dicts with 'support' and 'query' data
            loss_fn: Loss function (e.g., MSE for video enhancement)
        
        Returns:
            Meta-loss (average query loss across tasks)
        """
        meta_loss = 0.0
        
        for task in task_batch:
            support_data = task['support']  # (inputs, targets)
            query_data = task['query']  # (inputs, targets)
            
            # Inner loop: adapt to task
            if self.first_order:
                # First-order MAML (no second derivatives)
                with torch.no_grad():
                    adapted_model = self._inner_loop(
                        self.model, support_data, loss_fn
                    )
                
                # Query loss
                query_inputs, query_targets = query_data
                query_outputs = adapted_model(query_inputs)
                task_loss = loss_fn(query_outputs, query_targets)
            else:
                # Full MAML with second-order gradients using higher
                with higher.innerloop_ctx(
                    self.model,
                    torch.optim.SGD(self.model.parameters(), lr=self.inner_lr),
                    copy_initial_weights=True,
                ) as (fmodel, diffopt):
                    # Inner loop
                    support_inputs, support_targets = support_data
                    for _ in range(self.inner_steps):
                        support_outputs = fmodel(support_inputs)
                        support_loss = loss_fn(support_outputs, support_targets)
                        diffopt.step(support_loss)
                    
                    # Query loss
                    query_inputs, query_targets = query_data
                    query_outputs = fmodel(query_inputs)
                    task_loss = loss_fn(query_outputs, query_targets)
            
            meta_loss += task_loss
        
        meta_loss /= len(task_batch)
        return meta_loss
    
    def adapt(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable,
        steps: Optional[int] = None,
    ) -> nn.Module:
        """
        Adapt model to new task (content type).
        
        Args:
            data: (inputs, targets) for the new task
            loss_fn: Loss function
            steps: Number of adaptation steps
        
        Returns:
            Adapted model
        """
        return self._inner_loop(self.model, data, loss_fn, steps)
    
    def train_step(
        self,
        task_batch: List[Dict],
        loss_fn: Callable,
    ) -> float:
        """
        Complete meta-training step with optimizer update.
        
        Args:
            task_batch: Batch of tasks
            loss_fn: Loss function
        
        Returns:
            Meta-loss value
        """
        self.meta_optimizer.zero_grad()
        meta_loss = self.meta_step(task_batch, loss_fn)
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def state_dict(self) -> Dict:
        """Get state for saving."""
        return {
            'model': self.model.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'inner_steps': self.inner_steps,
            'first_order': self.first_order,
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """Load state from checkpoint."""
        self.model.load_state_dict(state['model'])
        self.meta_optimizer.load_state_dict(state['meta_optimizer'])
        self.inner_lr = state['inner_lr']
        self.outer_lr = state['outer_lr']
        self.inner_steps = state['inner_steps']
        self.first_order = state['first_order']


class FOMAML(MAML):
    """First-Order MAML (faster, simpler)."""
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
    ):
        super().__init__(
            model, inner_lr, outer_lr, inner_steps,
            first_order=True,
        )


class Reptile:
    """
    Reptile meta-learning algorithm.
    
    Simpler than MAML - just moves initialization toward
    adapted parameters. No second-order gradients needed.
    
    Reference:
        Nichol et al., "On First-Order Meta-Learning Algorithms"
    
    Args:
        model: Neural network model
        inner_lr: Learning rate for inner loop
        outer_lr: Learning rate for meta-update (interpolation factor)
        inner_steps: Number of inner loop steps
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.1,
        inner_steps: int = 10,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
    
    def train_step(
        self,
        task_batch: List[Dict],
        loss_fn: Callable,
    ) -> float:
        """
        Perform Reptile meta-update.
        
        Args:
            task_batch: Batch of tasks
            loss_fn: Loss function
        
        Returns:
            Average task loss
        """
        device = next(self.model.parameters()).device
        
        # Store initial parameters
        init_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        total_loss = 0.0
        adapted_params_list = []
        
        for task in task_batch:
            # Reset to initial parameters
            for name, param in self.model.named_parameters():
                param.data.copy_(init_params[name])
            
            # Inner loop on task
            inputs, targets = task['support']
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.inner_lr,
            )
            
            for _ in range(self.inner_steps):
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            # Store adapted parameters
            adapted_params_list.append({
                name: param.data.clone()
                for name, param in self.model.named_parameters()
            })
        
        # Reptile update: move toward average of adapted parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Average of adapted parameters
                avg_adapted = torch.stack([
                    adapted[name] for adapted in adapted_params_list
                ]).mean(dim=0)
                
                # Interpolate
                param.data.copy_(
                    init_params[name] + 
                    self.outer_lr * (avg_adapted - init_params[name])
                )
        
        return total_loss / len(task_batch)
    
    def adapt(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable,
        steps: Optional[int] = None,
    ) -> nn.Module:
        """Adapt to new task."""
        steps = steps or self.inner_steps
        inputs, targets = data
        device = next(self.model.parameters()).device
        inputs, targets = inputs.to(device), targets.to(device)
        
        adapted_model = deepcopy(self.model)
        optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.inner_lr,
        )
        
        for _ in range(steps):
            optimizer.zero_grad()
            outputs = adapted_model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        
        return adapted_model


class ContentAdaptiveMAML(MAML):
    """
    MAML specialized for video content adaptation.
    
    Additional features:
        - Content-type aware task sampling
        - Adaptive inner learning rate
        - Quality-based task difficulty
    """
    
    def __init__(
        self,
        model: nn.Module,
        content_types: List[str],
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
    ):
        super().__init__(model, inner_lr, outer_lr, inner_steps, first_order=True)
        
        self.content_types = content_types
        
        # Adaptive learning rates per content type
        self.content_lr = nn.ParameterDict({
            ct: nn.Parameter(torch.tensor(inner_lr))
            for ct in content_types
        })
    
    def adapt_to_content(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        content_type: str,
        loss_fn: Callable,
        steps: Optional[int] = None,
    ) -> nn.Module:
        """
        Adapt to specific content type with adaptive LR.
        
        Args:
            data: (inputs, targets)
            content_type: Content type string
            loss_fn: Loss function
            steps: Adaptation steps
        
        Returns:
            Adapted model
        """
        steps = steps or self.inner_steps
        inputs, targets = data
        
        # Get content-specific learning rate
        if content_type in self.content_lr:
            lr = self.content_lr[content_type].item()
        else:
            lr = self.inner_lr
        
        # Adapt
        adapted_model = deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            outputs = adapted_model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        
        return adapted_model


if __name__ == "__main__":
    # Test MAML
    print("Testing MAML...")
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    
    maml = FOMAML(model, inner_lr=0.01, inner_steps=5)
    
    # Create dummy task batch
    task_batch = []
    for _ in range(4):  # 4 tasks
        support_x = torch.randn(16, 10)
        support_y = torch.randn(16, 10)
        query_x = torch.randn(16, 10)
        query_y = torch.randn(16, 10)
        
        task_batch.append({
            'support': (support_x, support_y),
            'query': (query_x, query_y),
        })
    
    # Meta-training step
    loss_fn = nn.MSELoss()
    meta_loss = maml.train_step(task_batch, loss_fn)
    print(f"Meta-loss: {meta_loss:.4f}")
    
    # Test adaptation
    new_data = (torch.randn(16, 10), torch.randn(16, 10))
    adapted_model = maml.adapt(new_data, loss_fn, steps=10)
    print("Adaptation complete!")
    
    # Test Reptile
    print("\nTesting Reptile...")
    reptile = Reptile(model, inner_steps=10)
    loss = reptile.train_step(task_batch, loss_fn)
    print(f"Reptile loss: {loss:.4f}")
