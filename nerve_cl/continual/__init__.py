"""Continual learning components for video enhancement."""

from nerve_cl.continual.memory import EpisodicMemory, StreamingEpisodicMemory
from nerve_cl.continual.ewc import EWC, OnlineEWC, SynapticIntelligence
from nerve_cl.continual.maml import MAML, FOMAML, Reptile
from nerve_cl.continual.distillation import DistillationLoss, ContinualDistillation

__all__ = [
    # Memory
    "EpisodicMemory",
    "StreamingEpisodicMemory",
    # EWC
    "EWC",
    "OnlineEWC",
    "SynapticIntelligence",
    # Meta-Learning
    "MAML",
    "FOMAML",
    "Reptile",
    # Distillation
    "DistillationLoss",
    "ContinualDistillation",
]
