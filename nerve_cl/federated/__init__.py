"""Federated learning components for personalized video enhancement."""

from nerve_cl.federated.client import VideoEnhancementClient, create_client
from nerve_cl.federated.server import (
    VideoEnhancementStrategy,
    FederatedTrainer,
    start_server,
)
from nerve_cl.federated.privacy import (
    PrivacyConfig,
    DPOptimizer,
    make_private,
)
from nerve_cl.federated.clustering import UserProfile, UserClustering

__all__ = [
    # Client
    "VideoEnhancementClient",
    "create_client",
    # Server
    "VideoEnhancementStrategy",
    "FederatedTrainer",
    "start_server",
    # Privacy
    "PrivacyConfig",
    "DPOptimizer",
    "make_private",
    # Clustering
    "UserProfile",
    "UserClustering",
]
