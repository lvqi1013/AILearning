# -*- coding:utf-8 -*-
"""
@file name  : dqnagent.py
@author     : Qi Lv (https://lvqi1013.github.io/)
@Email		: lvqi@hunnu.edu.cn
@date       : 2026/08/17
@brief      : DQN智能体核心类，实现深度Q网络算法的智能体逻辑。
              包含在线网络和目标网络双网络结构、ε-贪婪探索策略、经验回放更新机制。
              提供动作选择、网络参数更新、目标网络同步、探索率衰减等核心方法，
              支持CartPole等离散动作空间环境的强化学习训练。
"""

import torch
from torch import nn, optim
import random
from torch.nn import functional as F

from Qnetwork import QNetWork

class DQNAgent:
    def __init__(self, obs_dim, action_dim, action_space, batch_size = 64,
                 lr = 1e-3, gamma = 0.99, epsilon = 1.0,epsilon_min = 0.05, epsilon_decay = 0.995, device = "cuda:0"):
        self.device = device
        self.online_q_net = QNetWork(obs_dim, action_dim).to(self.device)
        self.target_q_net = QNetWork(obs_dim, action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.online_q_net.state_dict())

        self.optimizer = optim.Adam(self.online_q_net.parameters(), lr=lr)


        self.action_space = action_space

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
    
    def select_action(self, obs):

        if random.random() < self.epsilon:
            # Exploration
            return self.action_space.sample()
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        # Exploitation
        with torch.no_grad():
            q_values = self.online_q_net(obs_tensor)
        
        return q_values.argmax(dim = 1).item() # 获取最大值的索引
    
    def update(self, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return None
        
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(self.batch_size)

        # actions 是选择动作的索引形状为(batch_size, 1)
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        
        current_q = self.online_q_net(obs).gather(1, actions)
        # 从神经网络输出的“所有动作的Q值”中，精准地“抠出”智能体实际选择的那个动作对应的Q值。gather(dim, index) 1表示横向挑列

        with torch.no_grad():
            # 计算Target
            max_next_q = self.target_q_net(next_obs).max(dim = 1, keepdim = True).values
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.item()
    
    def update_target_network(self):
        self.target_q_net.load_state_dict(self.online_q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)