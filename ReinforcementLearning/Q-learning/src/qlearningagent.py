import numpy as np
from typing import Optional, Union, Tuple
import gymnasium as gym
import random
import pickle

class QLearningAgent:
    """
    Tabular Q-Learning Agent.

    遵循 SB3 BaseAlgorithm 核心契约:
        - learn(total_timesteps)   : 主训练入口
        - predict(observation)     : 推理/评估入口
        - save(path) / load(path)  : 持久化
    """    
    def __init__(self, env: gym.Env, learning_rate: float = 0.1,
                 gamma: float = 0.95,
                 epsilon_decay: float = 0.9997,
                exploration_initial_eps: float = 1.0,
                exploration_final_eps: float = 0.05,
                verbose: int = 1,):
        """
        初始化表格型 Q-Learning 智能体。

        :param env: 交互的 Gym 环境。其观测空间应为离散/可离散化的（表格 Q 学习
            要求状态可索引），动作空间应为离散的 `gym.spaces.Discrete`。
        :type env: gym.Env
        :param learning_rate: Q 值更新步长 α（学习率），取值范围 (0, 1]。
            决定每次 TD 误差中吸收多少新信息，越大学习越快但越不稳定。
        :type learning_rate: float
        :param gamma: 折扣因子 γ，取值范围 [0, 1]。
            衡量未来奖励的相对重要性，越接近 1 越重视长期回报。
        :type gamma: float
        :param exploration_fraction: 从 `exploration_initial_eps` 线性衰减到
            `exploration_final_eps` 所用的训练时间比例（0~1）。例如 0.1 表示
            前 10% 的训练步数内完成 ε 的退火。
        :type exploration_fraction: float
        :param exploration_initial_eps: 训练开始时的探索率 ε（初始随机动作概率）。
            默认 1.0 表示开局完全随机探索。
        :type exploration_initial_eps: float
        :param exploration_final_eps: 训练结束（退火完成）后的最小探索率 ε。
            保留少量探索以避免后期陷入次优策略。
        :type exploration_final_eps: float
        :param verbose: 日志输出详细程度。0 静默，1 正常打印训练信息，>=2 更详细。
        :type verbose: int
        """
        
        self.env = env
        self.verbose = verbose
        self.num_timesteps = 0

        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps

        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self._current_eps = exploration_initial_eps

    def predict(
        self,
        observation: Union[int, np.ndarray],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:             
        """
        deterministic 参数用于控制动作选择是否包含随机性
        True 表示确定性动作，可复现，False表示可以探索动作用于训练。

        Returns:
            action: np.ndarray (SB3 约定返回 ndarray 而非 int)
            state:  None (tabular 方法无隐藏状态)
        """                   
        if isinstance(observation, np.ndarray):
            obs = int(observation.item()) if observation.ndim == 0 else int(observation[0])
        else:
            obs = int(observation)

        # obs 就是 Q值中提到的state

        if deterministic:
            # 只利用训练好的
            # 在当前状态下选择使得Q值最大的动作。Q-learning 算法中 ε-greedy 贪心选取最优动作
            q_values = self.q_table[obs] # 取出当前状态 obs 下所有动作对应的 Q 值一维数组
            max_q = q_values.max()
            candidates = np.where(q_values == max_q)[0]
            action = int(np.random.choice(candidates))

            # 不使用np.argmax(q_values)是因为只会返回第一个出现的最大值下标，上面会公平选择每一个
        
        else:
            # ε-greedy 策略：探索 vs 利用
            random_num = np.random.rand() # 1. 随机抽取一个[0,1) 的数字
            if random_num < self._current_eps:
                action = random.randrange(self.n_actions)   # 2. 随机数小于epsilon则在动作空间中随机选择一个
            else:
                # 利用
                q_values = self.q_table[obs] 
                max_q = q_values.max()
                candidates = np.where(q_values == max_q)[0]
                action = int(np.random.choice(candidates))
        return np.array([action]), state
    
    def learn(self,total_timesteps:int,   
                callback=None,# 预留 SB3 Callback 接口
                log_interval: int = 100,):
        """
        SB3 标准训练入口。按 total_timesteps 训练（而非 episodes）。
        返回 self（SB3 惯例，支持链式调用）。
        """        
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_count = 0

        for step in range(total_timesteps):

            action_arr, _ = self.predict(obs,deterministic=False)
            action = int(action_arr[0])

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            done = terminated or truncated

            # Q-Learning 更新 ★ 核心更新公式（背下来）★
            current_q = self.q_table[obs, action]
            best_next_q = self.q_table[next_obs].max() if not terminated else 0 # 如果到达了目标，则下一时刻的Q为0，如果没有结束则选择当前状态Q值里最大的
            td_target = reward + self.gamma * best_next_q
            td_error = td_target - current_q
            self.q_table[obs, action] += self.learning_rate * td_error

            episode_reward += reward
            self.num_timesteps += 1
                        
            # Episode 结束处理
            if done:
                episode_count += 1
                if self.verbose >= 1 and episode_count % log_interval == 0:
                    print(f"Timestep {self.num_timesteps}/{total_timesteps} | "
                          f"Ep {episode_count} | Reward: {episode_reward:.2f} | "
                          f"ε: {self._current_eps:.4f}")
                
                # 重置
                obs, info = self.env.reset()
                episode_reward = 0.0
                self._current_eps = max(self.exploration_final_eps, self._current_eps * self.epsilon_decay)
            else:
                obs = next_obs



        return self

    def save(self, path: str) -> None:
        """保存模型（SB3 使用 .zip，这里简化为 pickle）。"""
        data = {
            "q_table": self.q_table,
            "num_timesteps": self.num_timesteps,
            "_current_eps": self._current_eps,
            "config": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "epsilon_decay": self.epsilon_decay,
                "exploration_initial_eps": self.exploration_initial_eps,
                "exploration_final_eps": self.exploration_final_eps,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str, env=None) :
        """加载模型（SB3 惯例: load 是 classmethod，env 可选传入）。"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        config = data["config"]
        agent = cls(env=env, **config)
        agent.q_table = data["q_table"]
        agent.num_timesteps = data["num_timesteps"]
        agent._current_eps = data["_current_eps"]
        return agent

    # ══════════════════════════════════════════════
    #  工具方法
    # ══════════════════════════════════════════════

    def get_q(self, state: int, action: int) -> float:
        return float(self.q_table[state, action])

    def __repr__(self) -> str:
        return (
            f"TabularQLearning(states={self.n_states}, actions={self.n_actions}, "
            f"timesteps={self.num_timesteps}, ε={self._current_eps:.4f})"
        )   


if __name__ == '__main__':
    from environment import GridWorldEnv
    env = GridWorldEnv()
    agent = QLearningAgent(env)