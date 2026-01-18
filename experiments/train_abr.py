"""
Train ABR RL Agent.
"""

import argparse
import numpy as np
from pathlib import Path

from nerve_cl.abr import StreamingEnv, PPOAgent, ABRConfig


def train(args):
    """Train PPO agent for ABR decisions."""
    env = StreamingEnv(max_steps=100)
    
    agent = PPOAgent(
        obs_dim=7,
        num_actions=(5, 5),  # 5 quality levels, 5 enhancement levels
        config=ABRConfig(
            hidden_dims=(256, 256),
            learning_rate=args.learning_rate,
        ),
        device=args.device,
    )
    
    print(f"Training ABR agent for {args.num_steps} steps")
    
    episode_rewards = []
    best_reward = -float('inf')
    
    obs, _ = env.reset()
    episode_reward = 0
    
    for step in range(args.num_steps):
        # Select action
        action = agent.select_action(obs)
        
        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition
        agent.store_transition(action, reward, done)
        episode_reward += reward
        
        obs = next_obs
        
        if done:
            episode_rewards.append(episode_reward)
            
            # Update agent
            if len(agent.buffer['obs']) >= 64:
                metrics = agent.update()
            
            if len(episode_rewards) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {len(episode_rewards)}: Avg Reward = {avg_reward:.2f}")
                
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    agent.save('checkpoints/best_abr_agent.pt')
            
            obs, _ = env.reset()
            episode_reward = 0
    
    agent.save('checkpoints/final_abr_agent.pt')
    print(f"\nTraining complete. Best reward: {best_reward:.2f}")


def evaluate(args):
    """Evaluate trained agent."""
    env = StreamingEnv(max_steps=100)
    
    agent = PPOAgent(
        obs_dim=7,
        num_actions=(5, 5),
        device=args.device,
    )
    agent.load(args.model_path)
    
    rewards = []
    for episode in range(10):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, VMAF = {info['vmaf']:.1f}")
    
    print(f"\nAverage reward: {np.mean(rewards):.2f} (+/- {np.std(rewards):.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--num-steps', type=int, default=100000)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model-path', type=str, default='checkpoints/best_abr_agent.pt')
    args = parser.parse_args()
    
    Path('checkpoints').mkdir(exist_ok=True)
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
