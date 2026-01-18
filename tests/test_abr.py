"""Tests for ABR components."""

import pytest
import numpy as np
import torch

from nerve_cl.abr import StreamingEnv, PPOAgent, ABRConfig


class TestStreamingEnv:
    """Test streaming environment."""
    
    def test_reset(self):
        env = StreamingEnv()
        obs, info = env.reset()
        
        assert obs.shape == (7,)
        assert obs.dtype == np.float32
    
    def test_step(self):
        env = StreamingEnv()
        obs, _ = env.reset()
        
        action = np.array([2, 2])  # Mid quality, mid enhancement
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert next_obs.shape == (7,)
        assert isinstance(reward, float)
        assert 'vmaf' in info
    
    def test_episode(self):
        env = StreamingEnv(max_steps=20)
        obs, _ = env.reset()
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        assert steps == 20


class TestPPOAgent:
    """Test PPO agent."""
    
    def test_select_action(self):
        agent = PPOAgent(obs_dim=7, num_actions=(5, 5))
        obs = np.random.randn(7).astype(np.float32)
        
        action = agent.select_action(obs)
        
        assert action.shape == (2,)
        assert 0 <= action[0] < 5
        assert 0 <= action[1] < 5
    
    def test_training_step(self):
        env = StreamingEnv(max_steps=10)
        agent = PPOAgent(obs_dim=7, num_actions=(5, 5))
        
        obs, _ = env.reset()
        for _ in range(64):
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            agent.store_transition(action, reward, term or trunc)
            
            if term or trunc:
                obs, _ = env.reset()
            else:
                obs = next_obs
        
        metrics = agent.update()
        assert 'loss' in metrics
    
    def test_save_load(self, tmp_path):
        agent = PPOAgent(obs_dim=7, num_actions=(5, 5))
        
        save_path = str(tmp_path / "agent.pt")
        agent.save(save_path)
        
        agent2 = PPOAgent(obs_dim=7, num_actions=(5, 5))
        agent2.load(save_path)
