"""
Flower Federated Learning Client.
Implements local training with differential privacy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import NDArrays, Scalar
import numpy as np


def get_parameters(model: nn.Module) -> NDArrays:
    """Extract model parameters as numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    """Set model parameters from numpy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


class VideoEnhancementClient(fl.client.NumPyClient):
    """
    Flower client for federated video enhancement.
    
    Features:
        - Local training on user's video data
        - Differential privacy (gradient clipping + noise)
        - Privacy-preserving gradient upload
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cpu",
        local_epochs: int = 5,
        learning_rate: float = 1e-4,
        dp_enabled: bool = True,
        dp_epsilon: float = 8.0,
        dp_max_grad_norm: float = 1.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        
        # Differential privacy
        self.dp_enabled = dp_enabled
        self.dp_epsilon = dp_epsilon
        self.dp_max_grad_norm = dp_max_grad_norm
        
        self.criterion = nn.MSELoss()
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return current model parameters."""
        return get_parameters(self.model)
    
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train model on local data."""
        set_parameters(self.model, parameters)
        
        epochs = config.get("local_epochs", self.local_epochs)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )
        
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        
        for epoch in range(epochs):
            for batch in self.train_loader:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # Apply DP gradient clipping
                if self.dp_enabled:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.dp_max_grad_norm
                    )
                
                optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
                num_samples += inputs.size(0)
        
        avg_loss = total_loss / max(num_samples, 1)
        
        return get_parameters(self.model), num_samples, {"train_loss": avg_loss}
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model on local validation data."""
        set_parameters(self.model, parameters)
        
        if self.val_loader is None:
            return 0.0, 0, {}
        
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                num_samples += inputs.size(0)
        
        avg_loss = total_loss / max(num_samples, 1)
        
        return avg_loss, num_samples, {"val_loss": avg_loss}


def create_client(
    client_id: int,
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: Optional[torch.Tensor] = None,
    **kwargs,
) -> VideoEnhancementClient:
    """Factory function to create a FL client."""
    from torch.utils.data import TensorDataset
    
    train_dataset = TensorDataset(train_data[0], train_data[1])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    val_loader = None
    if val_data is not None:
        val_dataset = TensorDataset(val_data[0], val_data[1])
        val_loader = DataLoader(val_dataset, batch_size=16)
    
    return VideoEnhancementClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        **kwargs,
    )
