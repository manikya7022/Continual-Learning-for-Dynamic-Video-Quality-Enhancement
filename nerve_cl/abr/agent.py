"""
PPO Agent for Adaptive Bitrate Control.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ABRConfig:
    """ABR agent configuration."""
    hidden_dims: Tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(
        self,
        obs_dim: int,
        num_actions: Tuple[int, int],
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        
        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim
        self.features = nn.Sequential(*layers)
        
        # Policy heads (one per action dimension)
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], n) for n in num_actions
        ])
        
        # Value head
        self.value_head = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        features = self.features(obs)
        
        # Policy logits for each action dimension
        logits = [head(features) for head in self.policy_heads]
        
        # Value
        value = self.value_head(features)
        
        return tuple(logits) + (value,)
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Sample action and return log probability."""
        with torch.no_grad():
            *logits_list, value = self.forward(obs)
        
        actions = []
        log_probs = []
        
        for logits in logits_list:
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
            actions.append(action)
        
        action = torch.stack(actions, dim=-1).cpu().numpy()
        log_prob = sum(log_probs) if log_probs else torch.zeros(1)
        
        return action, log_prob, value


class PPOAgent:
    """PPO agent for ABR decisions."""
    
    def __init__(
        self,
        obs_dim: int,
        num_actions: Tuple[int, int],
        config: Optional[ABRConfig] = None,
        device: str = "cpu",
    ):
        self.config = config or ABRConfig()
        self.device = device
        
        self.network = ActorCritic(
            obs_dim, num_actions, self.config.hidden_dims
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.config.learning_rate
        )
        
        # Rollout buffer
        self.buffer = {
            'obs': [], 'actions': [], 'rewards': [],
            'values': [], 'log_probs': [], 'dones': [],
        }
    
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob, value = self.network.get_action(obs_t, deterministic)
        
        if not deterministic:
            self.buffer['obs'].append(obs)
            self.buffer['log_probs'].append(log_prob.item())
            self.buffer['values'].append(value.item())
        
        return action[0]
    
    def store_transition(self, action: np.ndarray, reward: float, done: bool):
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
    
    def update(self) -> Dict[str, float]:
        """Perform PPO update."""
        obs = torch.FloatTensor(np.array(self.buffer['obs'])).to(self.device)
        actions = torch.LongTensor(np.array(self.buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae()
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0.0
        for _ in range(10):  # PPO epochs
            *logits_list, values = self.network(obs)
            
            # Compute new log probs
            new_log_probs = 0
            entropy = 0
            for i, logits in enumerate(logits_list):
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = new_log_probs + dist.log_prob(actions[:, i])
                entropy = entropy + dist.entropy().mean()
            
            # Policy loss (clipped)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.config.clip_ratio, 1+self.config.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = ((values.squeeze() - returns) ** 2).mean()
            
            # Total loss
            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Clear buffer
        self.buffer = {k: [] for k in self.buffer}
        
        return {'loss': total_loss / 10}
    
    def _compute_gae(self):
        rewards = self.buffer['rewards']
        values = self.buffer['values'] + [0]
        dones = self.buffer['dones']
        
        gae = 0
        returns = []
        advantages = []
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return returns, advantages
    
    def save(self, path: str):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
