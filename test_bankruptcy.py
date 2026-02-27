import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path so we can import trading_gym
sys.path.append(os.getcwd())

from trading_gym import TradingEnv

class TestBankruptcy(unittest.TestCase):
    def test_bankruptcy_trigger(self):
        # Create a DataFrame with 20 rows
        # We need enough data for window_size (10) + at least one step

        # indices 0-9: Price 100.0
        # indices 10-19: Price 1.0

        prices = [100.0] * 10 + [1.0] * 10

        data = {
            'Close': prices,
            'RSI': [50.0] * 20,
            'MACD': [0.0] * 20,
            'Sentiment_Score': [0.0] * 20,
            'BB_Upper': [110.0] * 20,
            'BB_Lower': [90.0] * 20,
            'ATR': [1.0] * 20
        }

        df = pd.DataFrame(data)

        # Initialize environment
        # transaction_fee_percent=0.0 to make calculation simpler
        env = TradingEnv(df=df, window_size=10, transaction_fee_percent=0.0)

        # Reset environment
        # start_step=0 means current_step=0.
        # decision_idx = 0 + 10 - 1 = 9. Price[9] = 100.0.
        obs, info = env.reset(options={'start_step': 0})

        # Action: Buy 100% (Action 4 for Discrete, or [1.0] for Continuous)
        # The env handles both based on is_discrete flag. Default is Continuous (Box).
        # trading_gym.py: is_discrete defaults to False.
        # Action space is Box(-1, 1).
        action = np.array([1.0], dtype=np.float32)

        # Step
        obs, reward, terminated, truncated, info = env.step(action)

        # Check if terminated due to bankruptcy
        self.assertTrue(terminated, "Environment should be terminated due to bankruptcy")

        # Calculate current portfolio value
        # current_step is now 1
        # decision_idx is 1 + 10 - 1 = 10. Price[10] = 1.0.
        current_price = 1.0
        portfolio_value = env.cash + (env.shares_held * current_price)

        self.assertLess(portfolio_value, 1000, "Portfolio value should be less than 1000")

if __name__ == '__main__':
    unittest.main()
