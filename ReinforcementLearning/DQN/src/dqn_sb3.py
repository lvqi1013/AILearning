# -*- coding:utf-8 -*-
"""
@file name  : dqn_sb3.py
@author     : Qi Lv (https://lvqi1013.github.io/)
@Email		: lvqi@hunnu.edu.cn
@date       : 2026/08/17
@brief      : 使用Stable-Baselines3库实现DQN算法的示例程序。
              展示如何快速构建、训练、保存和评估DQN模型，支持CartPole等标准环境。
              包含自定义超参数配置、模型持久化、确定性预测和可视化渲染等完整流程，
              适合作为SB3库DQN使用的入门参考和快速原型验证。
"""

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# ========== 创建环境 ==========
# 简单环境示例
env = gym.make("CartPole-v1")

# Atari 环境示例（推荐用 SB3 内置的 wrapper）
# env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=0)
# env = DummyVecEnv(env)

# ========== 创建并训练模型 ==========
model = DQN(
    "MlpPolicy",
    env,
    device="cuda",
    learning_rate=3e-4,        # 适当提高学习率
    buffer_size=50000,         # CartPole 不需要太大 buffer
    learning_starts=1000,      # 减少纯随机探索
    batch_size=64,             # 增大 batch 稳定梯度
    train_freq=1,              # 每步都训练（简单环境可以激进些）
    gradient_steps=1,
    target_update_interval=500,# 更频繁更新目标网络
    exploration_fraction=0.05, # 更快结束探索
    exploration_final_eps=0.01,# 更低最终探索率
    policy_kwargs=dict(net_arch=[128, 128]),  # CartPole 不需要大网络
    verbose=1,
)
model.learn(total_timesteps=200000)

# ========== 保存与加载 ==========
model.save("dqn_cartpole")
del model

model = DQN.load("dqn_cartpole")

# ========== 评估 ==========
obs, info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

env.close()