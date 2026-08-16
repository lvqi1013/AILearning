import gymnasium as gym 
from environment import GridWorldEnv
from qlearningagent import QLearningAgent

# env = GridWorldEnv()
env = gym.make("FrozenLake-v1", is_slippery=False)
model = QLearningAgent(env)
model.learn(3000000)