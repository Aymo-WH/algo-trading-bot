import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import glob
import random

class TradingEnv(gym.Env):
    """Custom Trading Environment that follows gym interface"""
    metadata = {'render_modes': ['human']}

    def __init__(self, df=None):
        super(TradingEnv, self).__init__()

        self.window_size = 10

        # Load data
        if df is not None:
            self.dfs = [df]
            self.df = df
        else:
            # Find all CSV files in data/ folder
            data_files = glob.glob('data/*_data.csv')
            if not data_files:
                raise FileNotFoundError("No data files found in data/ directory matching pattern *_data.csv")

            self.dfs = []
            for file in data_files:
                # Load each file
                df_loaded = pd.read_csv(file).dropna().reset_index(drop=True)
                self.dfs.append(df_loaded)

            # Select a random DataFrame initially
            self.df = random.choice(self.dfs)

        # Update obs_matrix to include new features: Close, RSI, MACD, BBL, BBM, BBU, ATR
        self.obs_matrix = self.df[['Close', 'RSI', 'MACD', 'BBL', 'BBM', 'BBU', 'ATR']].values.astype(np.float32)
        
        self._prices = self.df['Close'].values 

        # Define action and observation space
        self.action_space = spaces.Discrete(3) # 0=Sell, 1=Hold, 2=Buy

        # Observation space is now a 2D Box (window_size, num_features)
        # num_features = 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, 7), dtype=np.float32
        )

        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly select a DataFrame for the new episode
        self.df = random.choice(self.dfs)

        # Rebuild observation matrix and prices for the chosen stock
        self.obs_matrix = self.df[['Close', 'RSI', 'MACD', 'BBL', 'BBM', 'BBU', 'ATR']].values.astype(np.float32)
        self._prices = self.df['Close'].values

        self.current_step = 0
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # The decision is made based on the window ending at current_step + window_size - 1
        decision_idx = self.current_step + self.window_size - 1

        current_price = self._prices[decision_idx]
        next_price = self._prices[decision_idx + 1]
        price_diff = next_price - current_price

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

        # Check if we can form the next window AND have a next price for the next step
        # Next observation covers [current_step, current_step + window_size)
        # Next step would need _prices[current_step + window_size] to calculate reward
        # So we need current_step + window_size < len(self.df)

        terminated = (self.current_step + self.window_size) >= len(self.df)
        truncated = False
        observation = self._get_observation()
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        return self.obs_matrix[self.current_step : self.current_step + self.window_size]

    def render(self, mode='human'):
        pass

    def close(self):
        pass
