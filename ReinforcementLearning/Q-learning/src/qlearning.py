import numpy as np
import gymnasium as gym

# ========== 1. 初始化 ==========
# FrozenLake-v1 环境（4×4冰湖，有陷阱和终点）
env = gym.make("FrozenLake-v1", is_slippery=False)  # 关闭滑动，降低难度便于理解, True时为随机环境，可以考验算法的鲁棒性
n_states = env.observation_space.n  # 16个格子
n_actions = env.action_space.n      # 4个方向

print(n_states)
print(n_actions)

# Q表：16行4列，初始全0
Q_tables = np.zeros((n_states, n_actions))

# 超参数
alpha = 0.1 # 学习率
gamma = 0.95 # 折扣因子
epsilon = 1.0 # 初始探索率
epsilon_decay = 0.9997
epsilon_min = 0.01
episodes = 30000

rewards_history = []

# ========== 2. 训练循环 ==========
for episode in range(episodes):
    state, info = env.reset()

    total_reward = 0
    done = False

    while not done:
        # ε-greedy 策略：探索 vs 利用

        random_num = np.random.rand() # 1. 随机抽取一个[0,1) 的数字
        if random_num < epsilon:
            action = env.action_space.sample() # 2. 随机数小于epsilon则在动作空间中随机选择一个
        else:
            action = np.argmax(Q_tables[state]) # 2. 随机数大于等于epsilon，则根据当前状态的动作空间选择Q值最大（最优）的动作
        
        # 执行动作，获取反馈
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ★ 核心更新公式（背下来）★
        best_next_q = np.max(Q_tables[next_state]) if not terminated else 0
        td_target = reward + gamma * best_next_q
        td_error = td_target - Q_tables[state, action]
        Q_tables[state, action] += alpha * td_error

        # 迭代
        state = next_state
        total_reward += reward
    
    # 衰减探索率
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_history.append(total_reward)

    if (episode + 1) % 200 == 0:
        avg = np.mean(rewards_history[-100:])
        print(f"Episode {episode+1} | Avg Reward: {avg:.3f} | Epsilon: {epsilon:.3f}")

# ========== 3. 查看学到的Q表 ==========
print("\n=== 最终Q表 ===")
for s in range(n_states):
    print(f"State {s:2d}: {np.round(Q_tables[s], 3)}")