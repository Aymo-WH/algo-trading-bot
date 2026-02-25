import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import glob
import random
import os

class TradingEnv(gym.Env):
    """Custom Trading Environment that follows gym interface"""
    metadata = {'render_modes': ['human']}

    def __init__(self, df=None, is_discrete=False, data_dir='data/', transaction_fee_percent=0.001):
        super(TradingEnv, self).__init__()

        self.is_discrete = is_discrete
        self.transaction_fee_percent = transaction_fee_percent
        self.window_size = 10

        # Load data
        if df is not None:
            self.dfs = [df]
            self.df = df
        else:
            # Find all CSV files in data/ folder
            pattern = os.path.join(data_dir, '*_data.csv')
            data_files = glob.glob(pattern)
            if not data_files:
                raise FileNotFoundError(f"No data files found in {data_dir} directory matching pattern *_data.csv")

            self.dfs = []
            for file in data_files:
                # Load each file
                df_loaded = pd.read_csv(file).dropna().reset_index(drop=True)
                self.dfs.append(df_loaded)

            # Select a random DataFrame initially
            self.df = random.choice(self.dfs)

        # Update obs_matrix to include new features: Close, RSI, MACD, Sentiment_Score, BB_Upper, BB_Lower, ATR
        self.obs_matrix = self.df[['Close', 'RSI', 'MACD', 'Sentiment_Score', 'BB_Upper', 'BB_Lower', 'ATR']].values.astype(np.float32)
        
        self._prices = self.df['Close'].values 

        # Define action and observation space
        if self.is_discrete:
            self.action_space = spaces.Discrete(5)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space is now a 2D Box (window_size, num_features)
        # num_features = 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, 7), dtype=np.float32
        )

        self.initial_balance = 10000.0
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly select a DataFrame for the new episode
        self.df = random.choice(self.dfs)

        # Rebuild observation matrix and prices for the chosen stock
        self.obs_matrix = self.df[['Close', 'RSI', 'MACD', 'Sentiment_Score', 'BB_Upper', 'BB_Lower', 'ATR']].values.astype(np.float32)
        self._prices = self.df['Close'].values

        if options and 'start_step' in options:
            self.current_step = options['start_step']
        else:
            # Generate random start step ensuring we have enough data for at least one step + window
            max_step = len(self.df) - self.window_size - 1
            self.current_step = random.randint(0, max_step) if max_step > 0 else 0

        self.cash = self.initial_balance
        self.shares_held = 0

        # Explicit resets requested
        self.balance = 10000.0
        self.net_worth = 10000.0
        self.total_fees = 0.0

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # The decision is made based on the window ending at current_step + window_size - 1
        decision_idx = self.current_step + self.window_size - 1
        current_price = self._prices[decision_idx]
        
        # Calculate portfolio value before action
        prev_val = self.cash + (self.shares_held * current_price)
        
        # Interpret action
        if self.is_discrete:
            # Map discrete actions to percentages
            # 0 -> -1.0 (Sell 100%)
            # 1 -> -0.5 (Sell 50%)
            # 2 -> 0.0 (Hold)
            # 3 -> 0.5 (Buy 50%)
            # 4 -> 1.0 (Buy 100%)
            mapping = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
            act = mapping[int(action)]
        else:
            # Action is a 1D array from Box space, e.g., [0.5]
            act = float(action[0])
        
        step_fee = 0.0
        
        if act > 0: # Buy
            # Buy shares using that percentage of self.cash
            # We interpret "percentage of self.cash" as the gross amount leaving the wallet.
            amount_to_invest = self.cash * act
            fee = amount_to_invest * self.transaction_fee_percent
            net_investment = amount_to_invest - fee
            
            if net_investment > 0:
                shares_bought = net_investment / current_price
                self.cash -= amount_to_invest
                self.shares_held += shares_bought
                step_fee = fee
                
        elif act < 0: # Sell
            # Sell that percentage of self.shares_held
            fraction = abs(act)
            shares_sold = self.shares_held * fraction
            
            gross_proceeds = shares_sold * current_price
            fee = gross_proceeds * self.transaction_fee_percent
            net_proceeds = gross_proceeds - fee
            
            self.cash += net_proceeds
            self.shares_held -= shares_sold
            step_fee = fee

        self.current_step += 1

        # Calculate new portfolio value at the new step
        new_decision_idx = self.current_step + self.window_size - 1
        new_price = self._prices[new_decision_idx]
        new_val = self.cash + (self.shares_held * new_price)
        
        reward = new_val - prev_val

        # Check termination
        terminated = (self.current_step + self.window_size) >= len(self.df)
        truncated = False
        
        # Bankruptcy check
        if new_val < 1000:
            terminated = True
            
        observation = self._get_observation()
        info = {'step_fee': step_fee}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        return self.obs_matrix[self.current_step : self.current_step + self.window_size]

    def render(self, mode='human'):
        pass

    def close(self):
        pass
