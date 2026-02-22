import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np

class TradingEnv(gym.Env):
    """Custom Trading Environment that follows gym interface"""
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(TradingEnv, self).__init__()

        # Load data
        self.df = pd.read_csv('nvda_data.csv')

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # 0=Sell, 1=Hold, 2=Buy
        self.action_space = spaces.Discrete(3)

        # Observation is the current step's 'Close', 'RSI', 'MACD' values
        # We use a Box space for continuous values
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        self.current_step = 0

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.current_step = 0

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        # Calculate reward based on the action and price change
        current_price = self.df.iloc[self.current_step]['Close']
        next_price = self.df.iloc[self.current_step + 1]['Close']
        price_diff = next_price - current_price

        reward = 0.0
        if action == 2:  # Buy
            reward = price_diff
        elif action == 0:  # Sell
            reward = -price_diff
        # Hold (action 1) gives 0 reward

        self.current_step += 1

        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        observation = self._get_observation()
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        # Get the current observation
        obs = self.df.iloc[self.current_step][['Close', 'RSI', 'MACD']].values
        return obs.astype(np.float32)

    def render(self, mode='human'):
        # Optional: Implement rendering logic
        pass

    def close(self):
        pass
