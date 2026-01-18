"""
Episodic Memory Buffer for Continual Learning.

Implements experience replay with various sampling strategies
to prevent catastrophic forgetting in video enhancement models.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
import random


@dataclass
class MemorySample:
    """A single sample stored in episodic memory."""
    
    frame_lr: torch.Tensor  # Low-resolution frame
    frame_hr: torch.Tensor  # High-resolution frame (ground truth)
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0  # For importance-based sampling
    access_count: int = 0  # For usage tracking
    
    def to(self, device: torch.device) -> 'MemorySample':
        """Move tensors to device."""
        return MemorySample(
            frame_lr=self.frame_lr.to(device),
            frame_hr=self.frame_hr.to(device),
            metadata=self.metadata,
            importance=self.importance,
            access_count=self.access_count,
        )


class EpisodicMemory:
    """
    Episodic Memory Buffer for Experience Replay.
    
    Stores representative samples from past tasks/content types
    to prevent catastrophic forgetting during continual learning.
    
    Sampling Strategies:
        - 'uniform': Random uniform sampling
        - 'reservoir': Reservoir sampling (stream-based)
        - 'stratified': Stratified by content type
        - 'importance': Importance-weighted sampling
        - 'diversity': Maximize sample diversity
    
    Args:
        capacity: Maximum number of samples to store
        strategy: Sampling strategy for eviction/retrieval
        diversity_weight: Weight for diversity in sample selection
    
    Example:
        >>> memory = EpisodicMemory(capacity=1000, strategy='stratified')
        >>> memory.store(frame_lr, frame_hr, {'content_type': 'sports'})
        >>> batch = memory.sample(batch_size=32)
    """
    
    def __init__(
        self,
        capacity: int = 1000,
        strategy: str = 'reservoir',
        diversity_weight: float = 0.3,
    ):
        self.capacity = capacity
        self.strategy = strategy
        self.diversity_weight = diversity_weight
        
        self.buffer: List[MemorySample] = []
        self.total_seen = 0  # For reservoir sampling
        
        # Content type tracking for stratified sampling
        self.content_type_indices: Dict[str, List[int]] = defaultdict(list)
        
        # Feature cache for diversity computation
        self.feature_cache: Optional[torch.Tensor] = None
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def store(
        self,
        frame_lr: torch.Tensor,
        frame_hr: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
    ) -> bool:
        """
        Store a sample in memory.
        
        Args:
            frame_lr: Low-resolution frame (C, H, W)
            frame_hr: High-resolution frame (C, H, W)
            metadata: Sample metadata (content_type, bitrate, etc.)
            importance: Sample importance score
        
        Returns:
            True if sample was stored, False if rejected
        """
        metadata = metadata or {}
        self.total_seen += 1
        
        sample = MemorySample(
            frame_lr=frame_lr.detach().cpu(),
            frame_hr=frame_hr.detach().cpu(),
            metadata=metadata,
            importance=importance,
        )
        
        if len(self.buffer) < self.capacity:
            # Buffer not full - always add
            idx = len(self.buffer)
            self.buffer.append(sample)
            self._update_content_indices(idx, metadata.get('content_type', 'unknown'))
            return True
        
        # Buffer full - use eviction strategy
        if self.strategy == 'reservoir':
            return self._reservoir_update(sample)
        elif self.strategy == 'stratified':
            return self._stratified_update(sample)
        elif self.strategy == 'importance':
            return self._importance_update(sample)
        elif self.strategy == 'diversity':
            return self._diversity_update(sample)
        else:  # FIFO
            return self._fifo_update(sample)
    
    def _reservoir_update(self, sample: MemorySample) -> bool:
        """Reservoir sampling: equal probability for all seen samples."""
        # Probability of keeping new sample
        prob = self.capacity / self.total_seen
        
        if random.random() < prob:
            # Replace random existing sample
            idx = random.randint(0, self.capacity - 1)
            old_content = self.buffer[idx].metadata.get('content_type', 'unknown')
            self._remove_from_content_indices(idx, old_content)
            
            self.buffer[idx] = sample
            self._update_content_indices(idx, sample.metadata.get('content_type', 'unknown'))
            return True
        
        return False
    
    def _stratified_update(self, sample: MemorySample) -> bool:
        """Maintain balanced content type distribution."""
        content_type = sample.metadata.get('content_type', 'unknown')
        
        # Find content type with most samples
        if self.content_type_indices:
            max_type = max(self.content_type_indices.keys(),
                          key=lambda x: len(self.content_type_indices[x]))
            
            # If new content type is underrepresented, prioritize it
            if content_type not in self.content_type_indices or \
               len(self.content_type_indices[content_type]) < len(self.content_type_indices[max_type]):
                # Evict from overrepresented type
                evict_idx = random.choice(self.content_type_indices[max_type])
                self._remove_from_content_indices(evict_idx, max_type)
                self.buffer[evict_idx] = sample
                self._update_content_indices(evict_idx, content_type)
                return True
        
        # Default to reservoir
        return self._reservoir_update(sample)
    
    def _importance_update(self, sample: MemorySample) -> bool:
        """Replace least important sample if new is more important."""
        # Find least important
        min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i].importance)
        
        if sample.importance > self.buffer[min_idx].importance:
            old_content = self.buffer[min_idx].metadata.get('content_type', 'unknown')
            self._remove_from_content_indices(min_idx, old_content)
            
            self.buffer[min_idx] = sample
            self._update_content_indices(min_idx, sample.metadata.get('content_type', 'unknown'))
            return True
        
        return False
    
    def _diversity_update(self, sample: MemorySample) -> bool:
        """Maintain diverse samples by replacing most similar."""
        # Compute simple feature (mean color)
        sample_feat = sample.frame_lr.mean(dim=(1, 2))
        
        if self.feature_cache is None:
            self.feature_cache = torch.stack([
                s.frame_lr.mean(dim=(1, 2)) for s in self.buffer
            ])
        
        # Find most similar sample
        distances = torch.norm(self.feature_cache - sample_feat, dim=1)
        min_idx = distances.argmin().item()
        
        # Replace if new sample adds diversity
        if distances[min_idx] > 0.1:  # Threshold for uniqueness
            old_content = self.buffer[min_idx].metadata.get('content_type', 'unknown')
            self._remove_from_content_indices(min_idx, old_content)
            
            self.buffer[min_idx] = sample
            self.feature_cache[min_idx] = sample_feat
            self._update_content_indices(min_idx, sample.metadata.get('content_type', 'unknown'))
            return True
        
        return False
    
    def _fifo_update(self, sample: MemorySample) -> bool:
        """First-in-first-out eviction."""
        old_content = self.buffer[0].metadata.get('content_type', 'unknown')
        self._remove_from_content_indices(0, old_content)
        
        self.buffer.pop(0)
        self.buffer.append(sample)
        
        # Rebuild indices
        self.content_type_indices.clear()
        for i, s in enumerate(self.buffer):
            ct = s.metadata.get('content_type', 'unknown')
            self.content_type_indices[ct].append(i)
        
        return True
    
    def _update_content_indices(self, idx: int, content_type: str) -> None:
        """Update content type index mapping."""
        self.content_type_indices[content_type].append(idx)
    
    def _remove_from_content_indices(self, idx: int, content_type: str) -> None:
        """Remove from content type index mapping."""
        if content_type in self.content_type_indices:
            if idx in self.content_type_indices[content_type]:
                self.content_type_indices[content_type].remove(idx)
    
    def sample(
        self,
        batch_size: int = 32,
        content_type: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Sample a batch from memory.
        
        Args:
            batch_size: Number of samples to retrieve
            content_type: Filter by content type (optional)
            device: Device to move tensors to
        
        Returns:
            Tuple of (frames_lr, frames_hr, metadata_list)
        """
        if len(self.buffer) == 0:
            raise ValueError("Memory buffer is empty")
        
        batch_size = min(batch_size, len(self.buffer))
        
        # Select indices
        if content_type is not None and content_type in self.content_type_indices:
            # Sample from specific content type
            available_indices = self.content_type_indices[content_type]
            indices = random.sample(available_indices, min(batch_size, len(available_indices)))
        else:
            # Stratified sampling across content types
            indices = self._stratified_sample(batch_size)
        
        # Gather samples
        samples = [self.buffer[i] for i in indices]
        for s in samples:
            s.access_count += 1
        
        # Stack tensors
        frames_lr = torch.stack([s.frame_lr for s in samples])
        frames_hr = torch.stack([s.frame_hr for s in samples])
        metadata = [s.metadata for s in samples]
        
        if device is not None:
            frames_lr = frames_lr.to(device)
            frames_hr = frames_hr.to(device)
        
        return frames_lr, frames_hr, metadata
    
    def _stratified_sample(self, batch_size: int) -> List[int]:
        """Sample with content type stratification."""
        if not self.content_type_indices:
            return random.sample(range(len(self.buffer)), batch_size)
        
        indices = []
        content_types = list(self.content_type_indices.keys())
        samples_per_type = batch_size // len(content_types)
        remainder = batch_size % len(content_types)
        
        for ct in content_types:
            available = self.content_type_indices[ct]
            n = samples_per_type + (1 if remainder > 0 else 0)
            remainder -= 1
            
            n = min(n, len(available))
            indices.extend(random.sample(available, n))
        
        return indices[:batch_size]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        content_counts = {k: len(v) for k, v in self.content_type_indices.items()}
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'total_seen': self.total_seen,
            'content_distribution': content_counts,
            'strategy': self.strategy,
        }
    
    def clear(self) -> None:
        """Clear all memory."""
        self.buffer.clear()
        self.content_type_indices.clear()
        self.total_seen = 0
        self.feature_cache = None
    
    def save(self, path: str) -> None:
        """Save memory to disk."""
        data = {
            'buffer': [(s.frame_lr, s.frame_hr, s.metadata, s.importance) for s in self.buffer],
            'total_seen': self.total_seen,
            'strategy': self.strategy,
            'capacity': self.capacity,
        }
        torch.save(data, path)
    
    def load(self, path: str) -> None:
        """Load memory from disk."""
        data = torch.load(path)
        
        self.buffer = [
            MemorySample(frame_lr=lr, frame_hr=hr, metadata=meta, importance=imp)
            for lr, hr, meta, imp in data['buffer']
        ]
        self.total_seen = data['total_seen']
        
        # Rebuild content indices
        self.content_type_indices.clear()
        for i, s in enumerate(self.buffer):
            ct = s.metadata.get('content_type', 'unknown')
            self.content_type_indices[ct].append(i)


class StreamingEpisodicMemory(EpisodicMemory):
    """
    Streaming variant optimized for online learning.
    
    Additional features:
        - Recency bias
        - Adaptive importance
        - Memory compression
    """
    
    def __init__(
        self,
        capacity: int = 1000,
        recency_weight: float = 0.2,
        compress_old: bool = True,
    ):
        super().__init__(capacity, strategy='reservoir')
        
        self.recency_weight = recency_weight
        self.compress_old = compress_old
        self.timestamps: List[int] = []
        self.current_time = 0
    
    def store(
        self,
        frame_lr: torch.Tensor,
        frame_hr: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
    ) -> bool:
        """Store with timestamp tracking."""
        self.current_time += 1
        
        stored = super().store(frame_lr, frame_hr, metadata, importance)
        
        if stored:
            if len(self.timestamps) < len(self.buffer):
                self.timestamps.append(self.current_time)
            else:
                # Find where sample was stored
                idx = len(self.buffer) - 1
                if idx < len(self.timestamps):
                    self.timestamps[idx] = self.current_time
        
        return stored
    
    def sample(
        self,
        batch_size: int = 32,
        content_type: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_recency: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """Sample with optional recency bias."""
        if not use_recency:
            return super().sample(batch_size, content_type, device)
        
        batch_size = min(batch_size, len(self.buffer))
        
        # Compute weights based on recency
        weights = []
        for i, sample in enumerate(self.buffer):
            time_weight = 1.0 / (1 + self.current_time - self.timestamps[i])
            importance_weight = sample.importance
            weight = (1 - self.recency_weight) * importance_weight + self.recency_weight * time_weight
            weights.append(weight)
        
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
            p=weights,
        ).tolist()
        
        # Gather samples
        samples = [self.buffer[i] for i in indices]
        frames_lr = torch.stack([s.frame_lr for s in samples])
        frames_hr = torch.stack([s.frame_hr for s in samples])
        metadata = [s.metadata for s in samples]
        
        if device is not None:
            frames_lr = frames_lr.to(device)
            frames_hr = frames_hr.to(device)
        
        return frames_lr, frames_hr, metadata


if __name__ == "__main__":
    # Test episodic memory
    print("Testing EpisodicMemory...")
    
    memory = EpisodicMemory(capacity=100, strategy='stratified')
    
    # Store samples
    for i in range(150):
        frame_lr = torch.randn(3, 64, 64)
        frame_hr = torch.randn(3, 128, 128)
        content_type = random.choice(['sports', 'animation', 'movie', 'news'])
        
        memory.store(
            frame_lr, frame_hr,
            metadata={'content_type': content_type, 'quality': random.random()},
            importance=random.random(),
        )
    
    print(f"Memory stats: {memory.get_stats()}")
    
    # Sample
    lr, hr, meta = memory.sample(batch_size=16)
    print(f"Sampled batch - LR shape: {lr.shape}, HR shape: {hr.shape}")
    print(f"Content types: {[m['content_type'] for m in meta]}")
    
    # Test streaming memory
    print("\nTesting StreamingEpisodicMemory...")
    stream_memory = StreamingEpisodicMemory(capacity=50)
    
    for i in range(100):
        frame_lr = torch.randn(3, 64, 64)
        frame_hr = torch.randn(3, 128, 128)
        stream_memory.store(frame_lr, frame_hr, importance=random.random())
    
    lr, hr, _ = stream_memory.sample(batch_size=8, use_recency=True)
    print(f"Streaming sample shape: {lr.shape}")
