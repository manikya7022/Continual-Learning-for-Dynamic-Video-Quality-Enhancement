"""
Federated Learning Training Script.
"""

import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from nerve_cl.models import SuperResolutionNet
from nerve_cl.federated import (
    VideoEnhancementClient,
    FederatedTrainer,
    PrivacyConfig,
)


def create_client_data(client_id: int, num_samples: int = 100):
    """Create heterogeneous client data."""
    # Each client has different data distribution
    offset = (client_id % 5) * 0.1
    lr = torch.randn(num_samples, 3, 64, 64) + offset
    hr = torch.randn(num_samples, 3, 128, 128) + offset
    return lr, hr


def run_simulation(args):
    """Run federated learning simulation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Global model
    model = SuperResolutionNet(scale_factor=2).to(device)
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create federated trainer
    trainer = FederatedTrainer(
        model=model,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
    )
    
    # Create client data
    for client_id in range(args.num_clients):
        lr, hr = create_client_data(client_id, num_samples=200)
        trainer.set_client_data(client_id, (lr, hr))
    
    print(f"\nStarting federated training with {args.num_clients} clients")
    print(f"Clients per round: {args.clients_per_round}")
    
    # Training rounds
    for round_num in range(args.num_rounds):
        metrics = trainer.train_round()
        
        if (round_num + 1) % 10 == 0:
            print(f"Round {metrics['round']}: {metrics['clients']} clients, {metrics['samples']} samples")
    
    # Save global model
    Path('checkpoints').mkdir(exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/federated_model.pt')
    print("\nFederated training complete!")


def run_flower_server(args):
    """Run Flower FL server."""
    from nerve_cl.federated import start_server
    from nerve_cl.models import SuperResolutionNet
    
    model = SuperResolutionNet(scale_factor=2)
    
    print(f"Starting Flower server on {args.server_address}")
    print(f"Waiting for {args.min_clients} clients...")
    
    start_server(
        model=model,
        num_rounds=args.num_rounds,
        server_address=args.server_address,
        min_clients=args.min_clients,
    )


def run_flower_client(args):
    """Run Flower FL client."""
    import flwr as fl
    from nerve_cl.models import SuperResolutionNet
    
    model = SuperResolutionNet(scale_factor=2)
    
    # Create local data
    lr, hr = create_client_data(args.client_id, num_samples=500)
    dataset = TensorDataset(lr, hr)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create client
    client = VideoEnhancementClient(
        model=model,
        train_loader=train_loader,
        local_epochs=args.local_epochs,
        dp_enabled=args.dp_enabled,
    )
    
    print(f"Starting client {args.client_id}")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['simulation', 'server', 'client'], default='simulation')
    parser.add_argument('--num-clients', type=int, default=10)
    parser.add_argument('--clients-per-round', type=int, default=5)
    parser.add_argument('--num-rounds', type=int, default=50)
    parser.add_argument('--local-epochs', type=int, default=5)
    parser.add_argument('--server-address', type=str, default='[::]:8080')
    parser.add_argument('--client-id', type=int, default=0)
    parser.add_argument('--min-clients', type=int, default=2)
    parser.add_argument('--dp-enabled', action='store_true')
    args = parser.parse_args()
    
    if args.mode == 'simulation':
        run_simulation(args)
    elif args.mode == 'server':
        run_flower_server(args)
    elif args.mode == 'client':
        run_flower_client(args)


if __name__ == '__main__':
    main()
