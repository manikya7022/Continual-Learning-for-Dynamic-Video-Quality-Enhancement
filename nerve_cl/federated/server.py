"""
Flower Federated Learning Server.
Implements FedAvg aggregation with client sampling.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
import flwr as fl
from flwr.common import (
    Metrics, NDArrays, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg
import numpy as np


class VideoEnhancementStrategy(FedAvg):
    """
    Custom federated averaging strategy for video enhancement.
    
    Features:
        - Weighted aggregation by sample count
        - Client clustering support
        - Model versioning
    """
    
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_evaluate: float = 0.05,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        initial_parameters: Optional[Parameters] = None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            initial_parameters=initial_parameters,
        )
        self.round_number = 0
        self.best_loss = float('inf')
        self.model_versions: List[Parameters] = []
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from clients."""
        self.round_number = server_round
        
        # Call parent aggregation
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Save version
        if parameters is not None:
            self.model_versions.append(parameters)
            if len(self.model_versions) > 5:  # Keep last 5
                self.model_versions.pop(0)
        
        return parameters, metrics
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure clients for training."""
        config = {
            "server_round": server_round,
            "local_epochs": 5,
        }
        
        # Adaptive local epochs based on round
        if server_round > 50:
            config["local_epochs"] = 3  # Reduce for later rounds
        
        sample_size = max(
            int(client_manager.num_available() * self.fraction_fit),
            self.min_fit_clients,
        )
        
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients,
        )
        
        return [(client, fl.common.FitIns(parameters, config)) for client in clients]


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute weighted average of metrics."""
    total_samples = sum([num_samples for num_samples, _ in metrics])
    
    weighted_metrics = {}
    for num_samples, m in metrics:
        for key, value in m.items():
            if key not in weighted_metrics:
                weighted_metrics[key] = 0.0
            weighted_metrics[key] += num_samples * value
    
    return {k: v / total_samples for k, v in weighted_metrics.items()}


def start_server(
    model: nn.Module,
    num_rounds: int = 100,
    server_address: str = "[::]:8080",
    min_clients: int = 2,
) -> None:
    """Start the federated learning server."""
    from nerve_cl.federated.client import get_parameters
    
    # Initial parameters
    initial_params = ndarrays_to_parameters(get_parameters(model))
    
    strategy = VideoEnhancementStrategy(
        fraction_fit=0.1,
        fraction_evaluate=0.05,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=initial_params,
    )
    
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


class FederatedTrainer:
    """
    High-level federated training manager.
    Simulates federated learning for development/testing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_clients: int = 10,
        clients_per_round: int = 5,
        local_epochs: int = 5,
    ):
        self.model = model
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.local_epochs = local_epochs
        
        self.client_data = {}
        self.global_round = 0
    
    def set_client_data(self, client_id: int, data: Tuple):
        """Set data for a client."""
        self.client_data[client_id] = data
    
    def train_round(self) -> Dict[str, float]:
        """Execute one federated round."""
        import random
        
        # Sample clients
        available = list(self.client_data.keys())
        selected = random.sample(
            available, min(self.clients_per_round, len(available))
        )
        
        # Collect updates
        updates = []
        total_samples = 0
        
        for client_id in selected:
            data = self.client_data[client_id]
            # Simulate local training
            client_samples = len(data[0])
            total_samples += client_samples
            updates.append((client_samples, self.model.state_dict()))
        
        self.global_round += 1
        
        return {
            "round": self.global_round,
            "clients": len(selected),
            "samples": total_samples,
        }
