import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np

class TradingEnv(gym.Env):
    """Custom Trading Environment that follows gym interface"""
    metadata = {'render_modes': ['human']}

    _data_cache = None

    def __init__(self, df=None):
        super(TradingEnv, self).__init__()

        # Load data
        if df is not None:
            self.df = df
        else:
            if TradingEnv._data_cache is None:
                TradingEnv._data_cache = pd.read_csv('nvda_data.csv').dropna().reset_index(drop=True)
            self.df = TradingEnv._data_cache

        self.obs_matrix = self.df[['Close', 'RSI', 'MACD']].values.astype(np.float32)
        
        # FIX 1: Initialize the _prices array that Jules forgot!
        self._prices = self.df['Close'].values 

        # Define action and observation space
        self.action_space = spaces.Discrete(3) # 0=Sell, 1=Hold, 2=Buy
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        current_price = self._prices[self.current_step]
        next_price = self._prices[self.current_step + 1]
        price_diff = next_price - current_price

        # FIX 2: Add the Transaction Fee logic that Jules missed
        transaction_fee_percent = 0.001 # 0.1% fee
        fee = 0.0
        
        reward = 0.0
        if action == 2:  # Buy
            fee = current_price * transaction_fee_percent
            reward = price_diff - fee
        elif action == 0:  # Sell
            fee = current_price * transaction_fee_percent
            reward = -price_diff - fee
        # Hold (action 1) gives 0 reward and pays 0 fee

        self.current_step += 1

        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        observation = self._get_observation()
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        return self.obs_matrix[self.current_step]

    def render(self, mode='human'):
        pass

    def close(self):
        pass
