"""
User Clustering for Personalized Federated Learning.
Groups users by behavior for specialized model training.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from dataclasses import dataclass


@dataclass
class UserProfile:
    """User behavior profile for clustering."""
    user_id: str
    content_preferences: Dict[str, float]  # content_type -> watch_time
    quality_preference: float  # 0=battery, 1=quality
    network_pattern: str  # wifi, cellular, mixed
    device_tier: str  # low, mid, high
    update_vector: Optional[np.ndarray] = None


class UserClustering:
    """
    Cluster users by viewing behavior for personalization.
    
    Features:
        - Profile-based clustering
        - Model update-based clustering
        - Adaptive cluster assignment
    """
    
    def __init__(
        self,
        num_clusters: int = 8,
        method: str = "kmeans",
        update_frequency: int = 10,
    ):
        self.num_clusters = num_clusters
        self.method = method
        self.update_frequency = update_frequency
        
        self.users: Dict[str, UserProfile] = {}
        self.clusters: Dict[int, List[str]] = {i: [] for i in range(num_clusters)}
        self.cluster_models: Dict[int, torch.Tensor] = {}
        self.clusterer = None
    
    def register_user(self, profile: UserProfile) -> int:
        """Register user and assign to cluster."""
        self.users[profile.user_id] = profile
        
        if self.clusterer is not None:
            # Assign to existing cluster
            features = self._extract_features(profile)
            cluster_id = self.clusterer.predict([features])[0]
        else:
            # Random assignment until enough users
            cluster_id = len(self.users) % self.num_clusters
        
        self.clusters[cluster_id].append(profile.user_id)
        return cluster_id
    
    def _extract_features(self, profile: UserProfile) -> np.ndarray:
        """Extract feature vector from user profile."""
        features = []
        
        # Content preferences
        content_types = ['sports', 'animation', 'movie', 'news', 'music']
        for ct in content_types:
            features.append(profile.content_preferences.get(ct, 0.0))
        
        # Quality preference
        features.append(profile.quality_preference)
        
        # Network pattern encoding
        network_map = {'wifi': 0, 'cellular': 1, 'mixed': 0.5}
        features.append(network_map.get(profile.network_pattern, 0.5))
        
        # Device tier encoding
        tier_map = {'low': 0, 'mid': 0.5, 'high': 1}
        features.append(tier_map.get(profile.device_tier, 0.5))
        
        return np.array(features)
    
    def update_clusters(self) -> None:
        """Recompute clusters based on current user profiles."""
        if len(self.users) < self.num_clusters:
            return
        
        # Extract all features
        user_ids = list(self.users.keys())
        features = np.array([
            self._extract_features(self.users[uid]) for uid in user_ids
        ])
        
        # Cluster
        self.clusterer = KMeans(n_clusters=self.num_clusters, random_state=42)
        labels = self.clusterer.fit_predict(features)
        
        # Update cluster assignments
        self.clusters = {i: [] for i in range(self.num_clusters)}
        for uid, label in zip(user_ids, labels):
            self.clusters[label].append(uid)
    
    def get_cluster(self, user_id: str) -> int:
        """Get cluster ID for user."""
        for cluster_id, users in self.clusters.items():
            if user_id in users:
                return cluster_id
        return 0
    
    def get_cluster_stats(self) -> Dict[int, Dict]:
        """Get statistics for each cluster."""
        stats = {}
        for cluster_id, user_ids in self.clusters.items():
            if not user_ids:
                continue
            
            profiles = [self.users[uid] for uid in user_ids]
            stats[cluster_id] = {
                'size': len(user_ids),
                'avg_quality_pref': np.mean([p.quality_preference for p in profiles]),
                'content_mix': self._get_dominant_content(profiles),
            }
        return stats
    
    def _get_dominant_content(self, profiles: List[UserProfile]) -> str:
        """Get most popular content type in cluster."""
        content_totals: Dict[str, float] = {}
        for p in profiles:
            for ct, val in p.content_preferences.items():
                content_totals[ct] = content_totals.get(ct, 0) + val
        
        if not content_totals:
            return "unknown"
        return max(content_totals, key=content_totals.get)
