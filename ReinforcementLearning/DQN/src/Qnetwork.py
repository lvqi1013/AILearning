# -*- coding:utf-8 -*-
"""
@file name  : Qnetwork.py
@author     : Qi Lv (https://lvqi1013.github.io/)
@Email		: lvqi@hunnu.edu.cn
@date       : 2026/08/17
@brief      : 定义DQN算法中的Q网络类，用于估计状态-动作价值函数。输出层直接映射到动作空间维度，用于计算每个动作在当前状态下的Q值。
"""

import torch
from torch import nn

class QNetWork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs):
        return self.net(obs)

# q_values = q_net(obs)