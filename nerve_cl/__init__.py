"""
NERVE-CL: Continual Learning for Dynamic Video Quality Enhancement
===================================================================

A production-grade framework for real-time neural video enhancement
on mobile devices with continual learning, federated personalization,
and adaptive resource management.

Components:
    - models: Neural network architectures (Frame Recovery, Super-Resolution)
    - continual: Continual learning (Episodic Memory, EWC, MAML)
    - federated: Federated learning (Flower client/server, DP)
    - abr: Adaptive bitrate RL agent
    - data: Video datasets and transforms
    - metrics: Quality metrics (PSNR, SSIM, VMAF)
    - training: Training utilities
    - utils: Helper functions
"""

__version__ = "0.1.0"
__author__ = "NERVE-CL Team"

from nerve_cl.models import (
    FrameRecoveryNet,
    SuperResolutionNet,
    EnhancementEngine,
)
from nerve_cl.continual import (
    EpisodicMemory,
    EWC,
    MAML,
)

__all__ = [
    # Models
    "FrameRecoveryNet",
    "SuperResolutionNet", 
    "EnhancementEngine",
    # Continual Learning
    "EpisodicMemory",
    "EWC",
    "MAML",
]
