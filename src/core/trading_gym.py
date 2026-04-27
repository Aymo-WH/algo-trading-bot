import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import glob
import random
import os
from core.utils import load_config

class TradingEnv(gym.Env):
    """
    A unified trading environment integrating Information-Driven features and the Triple-Barrier Method.

    This custom Gymnasium environment handles the core simulation for reinforcement learning agents.
    It accepts pre-processed data (FFD, Point-in-Time PCA, Dollar Bars) and native implements
    Marcos López de Prado's Triple-Barrier Method (Profit Taking, Stop Loss, and Time barriers)
    during each step step() execution.
    """
    """Custom Trading Environment that follows gym interface"""
    metadata = {'render_modes': ['human']}

    # Class-level cache to store loaded DataFrames keyed by data_dir
    _DATA_CACHE = {}

    def __init__(self, df=None, is_discrete=False, data_dir='data/', transaction_fee_percent=None, window_size=10, xgb_model_path=None):
        """
        Initializes the trading environment and pre-loads data into memory caches to O(1) step access.

        Args:
            df (pd.DataFrame, optional): Directly pass a specific ticker DataFrame.
            is_discrete (bool): If True, use discrete actions (e.g. for DQN). Defaults to False.
            data_dir (str): Directory containing preprocessed CSV files. Defaults to 'data/'.
            transaction_fee_percent (float, optional): Custom fee. Overrides config.json.
            window_size (int): Lookback sequence length for state observations. Defaults to 10.
            xgb_model_path (str, optional): Path to the XGBoost model to generate signal/probability.
        """
        """
        Initialize the Trading Environment.

        :param df: Pandas DataFrame containing historical data. If None, loads from data_dir.
        :param is_discrete: Boolean flag for discrete action space (True) or continuous (False).
        :param data_dir: Directory path to load data from if df is None.
        :param transaction_fee_percent: Transaction fee as a percentage of trade value (default None -> load from config or 0.001).
        :param window_size: Size of the observation window (default 10).
        """
        super(TradingEnv, self).__init__()

        self.is_discrete = is_discrete

        if transaction_fee_percent is None:
            config = load_config()
            self.transaction_fee_percent = config.get('transaction_fee_percent', 0.001)
        else:
            self.transaction_fee_percent = transaction_fee_percent

        self.window_size = window_size

        # Load XGBoost model if provided
        self.xgb_model = None
        if xgb_model_path and os.path.exists(xgb_model_path):
            import xgboost as xgb
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(xgb_model_path)

        # Load data
        if df is not None:
            self.dfs = [df]
        else:
            # Check cache first
            if data_dir in TradingEnv._DATA_CACHE:
                self.dfs = TradingEnv._DATA_CACHE[data_dir]
            else:
                # Find all CSV files in data/ folder
                pattern = os.path.join(data_dir, '*_data.csv')
                data_files = glob.glob(pattern)
                if not data_files:
                    raise FileNotFoundError(f"No data files found in {data_dir} directory matching pattern *_data.csv")

                self.dfs = []
                for file in data_files:
                    # Load each file
                    try:
                        df_loaded = pd.read_csv(file).dropna().reset_index(drop=True)
                        if len(df_loaded) < self.window_size + 2:
                            print(f"Skipping {file}: Empty or not enough rows after dropna.")
                            continue

                        required_columns = ['Close', 'Close_FFD', 'PCA_1', 'PCA_2', 'PCA_3', 'PCA_4']
                        # Validate columns
                        if not set(required_columns).issubset(df_loaded.columns):
                            print(f"Skipping {file}: Missing required columns. Found {df_loaded.columns}")
                            continue

                        # Validate data types
                        df_loaded[required_columns] = df_loaded[required_columns].astype(np.float32)

                        self.dfs.append(df_loaded)
                    except Exception as e:
                        print(f"Skipping {file}: Invalid data format ({e}).")
                        continue

                if not self.dfs:
                    raise ValueError(f"No valid data files found in {data_dir} directory.")

                # Update cache
                TradingEnv._DATA_CACHE[data_dir] = self.dfs

        # Precompute observation matrices and prices for all DataFrames
        self.precomputed_data = []
        for d in self.dfs:
            # Drop Sentiment_Score and PCA_5, now 5 core features
            obs = d[['Close_FFD', 'PCA_1', 'PCA_2', 'PCA_3', 'PCA_4']].values.astype(np.float32)
            prices = d['Close'].values
            atr = prices * 0.02 # Removed ATR dependency
            opt_pt = d['Optimal_PT'].values if 'Optimal_PT' in d.columns else np.full(len(d), 2.0)
            opt_sl = d['Optimal_SL'].values if 'Optimal_SL' in d.columns else np.full(len(d), 2.0)
            self.precomputed_data.append({'df': d, 'obs': obs, 'prices': prices, 'atr': atr, 'opt_pt': opt_pt, 'opt_sl': opt_sl})

        # Select a random DataFrame initially
        selected_data = random.choice(self.precomputed_data)
        self.df = selected_data['df']
        self.obs_matrix = selected_data['obs']
        self._prices = selected_data['prices']
        self._atr = selected_data['atr']
        self._opt_pt = selected_data['opt_pt']
        self._opt_sl = selected_data['opt_sl']

        # Define action and observation space
        if self.is_discrete:
            self.action_space = spaces.Discrete(5)
        else:
            # Action space outputs continuous bet size [0.0, 1.0]
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space is now a 1D Box with 4 features:
        # [xgb_signal, xgb_prob, rolling_volatility, portfolio_drawdown]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        self.obs_buffer = np.empty((4,), dtype=np.float32)
        self._forced_sell_action = np.array([0.0], dtype=np.float32)

        self.initial_balance = 10000.0
        self.current_step = 0
        self.episode_length = 90

        # Triple Barrier Parameters
        self.max_holding_bars = 15  # Vertical Time Barrier

        # Trade State Tracking
        self.entry_price = 0.0
        self.entry_step = 0
        self.entry_atr = 0.0

    def reset(self, seed=None, options=None):
        """
        Resets the environment state and randomly selects a new data chunk for the episode.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Dict to pass start_step.

        Returns:
            tuple: (observation array, info dict).
        """
        super().reset(seed=seed)

        # Randomly select a precomputed data entry for the new episode
        selected_data = random.choice(self.precomputed_data)
        self.df = selected_data['df']
        self.obs_matrix = selected_data['obs']
        self._prices = selected_data['prices']
        self._atr = selected_data['atr']
        self._opt_pt = selected_data['opt_pt']
        self._opt_sl = selected_data['opt_sl']

        if options and 'start_step' in options:
            self.current_step = options['start_step']
        else:
            # Generate random start step ensuring we have enough data for at least one step + window
            max_step = len(self.df) - self.window_size - 1
            self.current_step = random.randint(0, max_step) if max_step > 0 else 0

        self.start_step = self.current_step
        self.cash = self.initial_balance
        self.shares_held = 0

        self.entry_price = 0.0
        self.entry_step = 0
        self.entry_atr = 0.0

        # Explicit resets requested
        self.balance = 10000.0
        self.net_worth = 10000.0
        self.total_fees = 0.0

        # Populate initial buffer
        for i in range(self.window_size):
            self._get_single_observation(self.current_step + i, self.cash, self.shares_held, self.obs_buffer[i])

        observation = self.obs_buffer
        info = {}
        return observation, info

    def step(self, action):
        """
        Executes a trading step incorporating Marcos López de Prado's Triple-Barrier logic.

        The function checks if an upper (Profit Taking), lower (Stop Loss), or vertical (Time)
        barrier has been breached based on dynamic Optimal Trading Rules (OTR). If breached,
        it forcefully overrides the agent's action and liquidates the position. The reward
        uses a Sortino-style Risk-Adjusted proxy, heavily penalizing downside volatility.

        Args:
        action (int/float/np.ndarray): The agent's requested action.

        Returns:
        tuple: (observation, reward, terminated, truncated, info dict).
        """
        decision_idx = self.current_step + self.window_size - 1

        # STRICT BOUNDS CHECK
        forced_termination = False
        if decision_idx >= len(self._prices):
            decision_idx = len(self._prices) - 1  # Clamp to final row
            forced_termination = True             # Force episode to end

        current_price = self._prices[decision_idx]

        # Fetch Current ATR (fallback to a small percentage if ATR column is missing)
        current_atr = self._atr[decision_idx]

        current_pt_mult = self._opt_pt[decision_idx]
        current_sl_mult = self._opt_sl[decision_idx]

        forced_sell = False
        if self.shares_held > 0 and self.entry_price > 0:
            bars_held = self.current_step - self.entry_step
            upper_barrier = self.entry_price + (self.entry_atr * current_pt_mult)
            lower_barrier = self.entry_price - (self.entry_atr * current_sl_mult)

            if current_price >= upper_barrier:  # Take Profit (Upper Barrier)
                forced_sell = True
            elif current_price <= lower_barrier:  # Stop Loss (Lower Barrier)
                forced_sell = True
            elif bars_held >= self.max_holding_bars:  # Time Limit (Vertical Barrier)
                forced_sell = True

        # Override action to force a 100% sell if a barrier is breached
        if forced_sell:
            if self.is_discrete:
                action = 0  # Assuming 0 maps to -1.0 in your mapping dict
            else:
                action = self._forced_sell_action

        # ETF TRICK: Track pure mark-to-market before rebalancing
        prev_val = self.cash + (self.shares_held * current_price)

        if self.is_discrete:
            mapping = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
            act = mapping[int(action)]
        else:
            act = float(action[0])
            
        step_fee = 0.0
            
        # Calculate Rebalancing
        if act > 0:
            amount_to_invest = self.cash * act
            step_fee = amount_to_invest * self.transaction_fee_percent
            net_investment = amount_to_invest - step_fee
                
            if net_investment > 0 and current_price > 0:
                shares_bought = net_investment / current_price
                self.cash -= amount_to_invest
                self.shares_held += shares_bought
                    
                if self.entry_price == 0.0:  # Only set on initial entry, not scaling in
                    self.entry_price = current_price
                    self.entry_step = self.current_step
                    self.entry_atr = current_atr

        elif act < 0:
            fraction = abs(act)
            shares_sold = self.shares_held * fraction
            gross_proceeds = shares_sold * current_price
            step_fee = gross_proceeds * self.transaction_fee_percent
            net_proceeds = gross_proceeds - step_fee
                
            self.cash += net_proceeds
            self.shares_held -= shares_sold

            if self.shares_held <= 1e-6: # Account for float precision
                self.shares_held = 0.0
                self.entry_price = 0.0
                self.entry_step = 0
                self.entry_atr = 0.0
    
        self.current_step += 1
    
        new_decision_idx = self.current_step + self.window_size - 1
        if new_decision_idx >= len(self._prices):
            new_decision_idx = len(self._prices) - 1
            forced_termination = True

        new_price = self._prices[new_decision_idx]
            
        # ETF TRICK: Calculate reward based on pure asset growth, THEN explicitly deduct fee
        pure_new_val = self.cash + (self.shares_held * new_price)
        daily_return = (pure_new_val - prev_val) / prev_val if prev_val > 0 else 0
            
        if daily_return > 0:
            reward = daily_return * 100
        else:
            reward = (daily_return * 100) * 1.5
                
        # Hold Cash Penalty
        if self.shares_held == 0:
            reward -= 0.01
                
        # Explicit Negative Dividend (Prevents fake compounding)
        reward -= (step_fee / prev_val) * 100 if prev_val > 0 else 0
    
        terminated = (self.current_step - self.start_step >= self.episode_length) or \
                     ((self.current_step + self.window_size) >= len(self.df)) or \
                     forced_termination
        truncated = False
            
        if pure_new_val < 1000:
            terminated = True
                
        self.obs_buffer[:-1] = self.obs_buffer[1:]
        self._get_single_observation(self.current_step + self.window_size - 1, self.cash, self.shares_held, self.obs_buffer[-1])
    
        return self.obs_buffer, reward, terminated, truncated, {'step_fee': step_fee}

    def _get_single_observation(self, step_idx, cash, shares_held, target_array):
        if step_idx < len(self.obs_matrix):
            base_obs = self.obs_matrix[step_idx]
            current_price = self._prices[step_idx]
        else:
            base_obs = self.obs_matrix[-1]
            current_price = self._prices[-1]

        norm_cash = cash / self.initial_balance
        norm_holdings = (shares_held * current_price) / self.initial_balance

        n = base_obs.shape[0]
        target_array[:n] = base_obs
        target_array[n] = norm_cash
        target_array[n + 1] = norm_holdings

    def render(self, mode='human'):
        """
        Prints current step performance (Net Worth, Profit, Cash, Shares).

        Args:
            mode (str): Rendering mode. Defaults to 'human'.
        """
        if mode == 'human':
            # Calculate current price based on the current window position
            decision_idx = self.current_step + self.window_size - 1

            # Ensure we don't go out of bounds
            if decision_idx < len(self._prices):
                current_price = self._prices[decision_idx]
            else:
                current_price = self._prices[-1]

            net_worth = self.cash + (self.shares_held * current_price)
            profit = net_worth - self.initial_balance

            print(f"Step: {self.current_step}")
            print(f"Price: {current_price:.2f}")
            print(f"Cash: {self.cash:.2f}")
            print(f"Shares: {self.shares_held:.2f}")
            print(f"Net Worth: {net_worth:.2f}")
            print(f"Profit: {profit:.2f}")
            print("-" * 20)

    def close(self):
        pass
