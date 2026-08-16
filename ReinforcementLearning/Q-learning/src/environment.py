import gymnasium as gym
import numpy as np
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    """
    5x5 网格世界，从 (0,0) 走到 (4,4)
    符合 Gymnasium v0.29+ 标准接口
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()     
        self.size = 5
        self.start = (0, 0)
        self.goal = (4, 4)

        # ✅ 1. 定义动作空间和观测空间（必须）
        self.action_space = spaces.Discrete(4) # 0:up, 1:down, 2:left, 3:right
        self.observation_space = spaces.Discrete(self.size * self.size)  # 25

        # 动作映射表，    
        self._action_to_delta = {
            0: (-1, 0),   # up
            1: (1, 0),    # down
            2: (0, -1),   # left
            3: (0, 1),    # right
        }

        self.state = None
        self.render_mode = render_mode
    # ── 坐标转换工具方法 ──────────────────────
    def _to_obs(self, x: int, y: int) -> int:
        """(x, y) → 一维状态编号"""
        return x * self.size + y

    def _from_obs(self, obs: int) -> tuple:
        """一维状态编号 → (x, y)"""
        return divmod(obs, self.size)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        sx, sy = self.start
        self.state = self._to_obs(sx, sy)       # ✅ 存为 int

        info = {}
        if self.render_mode == "human":
            self.render()
        return self.state, info                  # ✅ 返回 int
    
    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}, expected 0-3")

        x, y = self._from_obs(self.state)       # ✅ 解码为坐标
        dx, dy = self._action_to_delta[action]

        new_x = np.clip(x + dx, 0, self.size - 1)
        new_y = np.clip(y + dy, 0, self.size - 1)
        self.state = self._to_obs(new_x, new_y)  # ✅ 编码回 int

        terminated = (new_x, new_y) == self.goal
        truncated = False
        reward = 10.0 if terminated else -1.0

        info = {"state_tuple": (new_x, new_y)}

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, info  # ✅ 返回 int

    def render(self):
        x, y = self._from_obs(self.state)       # ✅ 渲染时解码
        if self.render_mode == "ansi":
            grid = [['·' for _ in range(self.size)] for _ in range(self.size)]
            grid[x][y] = '🤖'
            gx, gy = self.goal
            grid[gx][gy] = '🎯'
            return '\n'.join(' '.join(row) for row in grid)
        elif self.render_mode == "human":
            print(self.render())

    def close(self):
        pass

if __name__ == '__main__':
    # 验证环境是否正确
    from gymnasium.utils.env_checker import check_env
    
    env = GridWorldEnv()
    check_env(env, warn=True)  # ✅ 自动检测所有接口合规性
    print("环境检查通过！")