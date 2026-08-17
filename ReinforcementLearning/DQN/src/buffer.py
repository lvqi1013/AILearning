# -*- coding:utf-8 -*-
"""
@file name  : buffer.py
@author     : Qi Lv (https://lvqi1013.github.io/)
@Email		: lvqi@hunnu.edu.cn
@date       : 2026/08/17
@brief      : 经典DQN的最简单的ReplayBuffer
"""



from collections import deque
import random
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen= capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch =  random.sample(self.buffer, batch_size)

        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
        )
    
    def __len__(self):
        return len(self.buffer)

"""
使用：

"""