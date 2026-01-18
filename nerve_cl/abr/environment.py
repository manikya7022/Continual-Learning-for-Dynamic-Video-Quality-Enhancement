"""
Streaming Environment for ABR RL Agent.
Simulates video streaming with network conditions.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class QualityLevel:
    """Video quality level definition."""
    resolution: int  # e.g., 360, 480, 720, 1080
    bitrate: int     # kbps


class StreamingEnv(gym.Env):
    """
    Gymnasium environment for ABR decision making.
    
    State: [buffer_level, bandwidth, battery, last_quality, content_complexity]
    Action: (bitrate_idx, enhancement_strength)
    Reward: QoE (quality - rebuffer_penalty - smoothness_penalty + battery_bonus)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        quality_ladder: Optional[List[QualityLevel]] = None,
        segment_duration: float = 4.0,
        buffer_size: float = 30.0,
        max_steps: int = 100,
    ):
        super().__init__()
        
        self.quality_ladder = quality_ladder or [
            QualityLevel(360, 365),
            QualityLevel(480, 750),
            QualityLevel(720, 1500),
            QualityLevel(1080, 3000),
            QualityLevel(1440, 6000),
        ]
        
        self.segment_duration = segment_duration
        self.buffer_size = buffer_size
        self.max_steps = max_steps
        
        self.num_qualities = len(self.quality_ladder)
        self.enhancement_levels = 5  # 0.0, 0.25, 0.5, 0.75, 1.0
        
        # Action space: (quality_index, enhancement_level)
        self.action_space = spaces.MultiDiscrete([
            self.num_qualities,
            self.enhancement_levels,
        ])
        
        # Observation space
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        self.buffer_level = 10.0  # seconds
        self.bandwidth = np.random.uniform(2, 15)  # Mbps
        self.battery = 1.0
        self.last_quality = 2  # Start at 720p
        self.last_vmaf = 70.0
        self.step_count = 0
        self.total_rebuffer = 0.0
        
        return self._get_obs(), {}
    
    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.buffer_level / self.buffer_size,
            min(self.bandwidth / 20, 1.0),
            self.battery,
            self.last_quality / self.num_qualities,
            self.last_vmaf / 100,
            np.random.uniform(0.3, 0.8),  # content complexity
            self.step_count / self.max_steps,
        ], dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        quality_idx = int(action[0])
        enhancement = action[1] / (self.enhancement_levels - 1)
        
        quality = self.quality_ladder[quality_idx]
        
        # Calculate download time
        chunk_size = quality.bitrate * self.segment_duration  # kbits
        download_time = chunk_size / (self.bandwidth * 1000)  # seconds
        
        # Update buffer
        old_buffer = self.buffer_level
        self.buffer_level -= download_time
        
        # Calculate rebuffering
        rebuffer = max(0, -self.buffer_level)
        self.total_rebuffer += rebuffer
        self.buffer_level = max(0, self.buffer_level) + self.segment_duration
        self.buffer_level = min(self.buffer_level, self.buffer_size)
        
        # Simulate VMAF (quality metric)
        base_vmaf = 50 + (quality_idx / self.num_qualities) * 40
        enhanced_vmaf = base_vmaf + enhancement * 10
        self.last_vmaf = min(enhanced_vmaf, 100)
        
        # Battery consumption
        battery_cost = 0.01 + enhancement * 0.02
        self.battery = max(0, self.battery - battery_cost)
        
        # Compute reward
        quality_reward = self.last_vmaf / 100
        rebuffer_penalty = rebuffer * 10
        smoothness_penalty = abs(quality_idx - self.last_quality) * 0.1
        battery_bonus = self.battery * 0.1
        
        reward = quality_reward - rebuffer_penalty - smoothness_penalty + battery_bonus
        
        self.last_quality = quality_idx
        self.step_count += 1
        
        # Update bandwidth (simulate network variation)
        self.bandwidth *= np.random.uniform(0.8, 1.2)
        self.bandwidth = np.clip(self.bandwidth, 0.5, 50)
        
        terminated = self.step_count >= self.max_steps
        truncated = self.battery <= 0
        
        info = {
            'vmaf': self.last_vmaf,
            'rebuffer': rebuffer,
            'bandwidth': self.bandwidth,
            'buffer': self.buffer_level,
        }
        
        return self._get_obs(), reward, terminated, truncated, info


def make_env(env_id: str = "Streaming-v0", **kwargs) -> StreamingEnv:
    """Create streaming environment."""
    return StreamingEnv(**kwargs)
