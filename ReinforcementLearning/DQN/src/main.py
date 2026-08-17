# -*- coding:utf-8 -*-
"""
@file name  : main.py
@author     : Qi Lv (https://lvqi1013.github.io/)
@Email		: lvqi@hunnu.edu.cn
@date       : 2026/08/17
@brief      : DQN算法主训练程序。使用CartPole-v1环境进行训练，
              实现了经验回放、目标网络更新、ε-贪婪策略等DQN核心机制。
              每轮训练输出奖励、损失、探索率等关键指标，支持批量训练和性能监控。
"""

import gymnasium as gym
from collections import deque
import numpy as np

from dqnagent import DQNAgent
from buffer import ReplayBuffer

def train():
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(obs_dim, action_dim, env.action_space, 128)
    replay_buffer = ReplayBuffer(capacity=50000)

    num_episodes = 500
    target_update_interval = 10
    recent_rewards = deque(maxlen=20)

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()

        episode_reward = 0
        episode_losses = []
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.push(obs, action, reward, next_obs, done)
            loss = agent.update(replay_buffer)
            if loss is not None:
                episode_losses.append(loss)
            
            episode_reward += reward
            obs = next_obs
        
        agent.decay_epsilon()
        recent_rewards.append(episode_reward)

        if episode % target_update_interval == 0:
            agent.update_target_network()
        
        mean_loss = np.mean(episode_losses) if episode_losses else 0.0
        mean_reward = np.mean(recent_rewards)

        print(
            f"episode={episode:03d} "
            f"reward={episode_reward:6.1f} "
            f"mean20={mean_reward:6.1f} "
            f"epsilon={agent.epsilon:.3f} "
            f"loss={mean_loss:.4f} "
            f"buffer={len(replay_buffer)}"
        )
    env.close()

if __name__ == '__main__':
    train()