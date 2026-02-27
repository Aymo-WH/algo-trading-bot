import unittest
import pandas as pd
import numpy as np
import gymnasium as gym
from trading_gym import TradingEnv

class TestTradingEnvDiscreteActions(unittest.TestCase):
    def setUp(self):
        # Create a dummy DataFrame with required columns
        # We need enough data for window_size (default 10) + episode_length (90) + some buffer
        n_rows = 200
        self.df = pd.DataFrame({
            'Close': np.random.uniform(100, 200, n_rows).astype(np.float32),
            'RSI': np.random.uniform(0, 100, n_rows).astype(np.float32),
            'MACD': np.random.uniform(-5, 5, n_rows).astype(np.float32),
            'Sentiment_Score': np.random.uniform(-1, 1, n_rows).astype(np.float32),
            'BB_Upper': np.random.uniform(100, 200, n_rows).astype(np.float32),
            'BB_Lower': np.random.uniform(100, 200, n_rows).astype(np.float32),
            'ATR': np.random.uniform(0, 10, n_rows).astype(np.float32)
        })

        # Initialize environment with discrete actions
        self.env = TradingEnv(df=self.df, is_discrete=True)

    def test_discrete_actions(self):
        # Reset environment
        obs, info = self.env.reset()

        # Test all discrete actions (0 to 4)
        # 0: Sell 100%
        # 1: Sell 50%
        # 2: Hold
        # 3: Buy 50%
        # 4: Buy 100%
        actions = [0, 1, 2, 3, 4]

        for action in actions:
            print(f"Testing action: {action}")
            try:
                # We need to make sure we don't hit the end of the episode too quickly
                # so we just take one step per action, or maybe reset in between if needed.
                # But taking consecutive steps is fine as long as we don't exceed episode length.

                obs, reward, terminated, truncated, info = self.env.step(action)

                # Basic checks
                self.assertIsInstance(obs, np.ndarray)
                self.assertTrue(isinstance(reward, float) or isinstance(reward, np.floating))
                self.assertIsInstance(terminated, bool)
                self.assertIsInstance(truncated, bool)
                self.assertIsInstance(info, dict)

                # Check observation shape
                # Window size is 10, features is 9
                self.assertEqual(obs.shape, (10, 9))

            except Exception as e:
                self.fail(f"Action {action} raised exception: {e}")

if __name__ == '__main__':
    unittest.main()
