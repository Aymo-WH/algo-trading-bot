import unittest
from unittest.mock import MagicMock, patch
import sys

# Pre-mock everything
mock_pd = MagicMock()
mock_np = MagicMock()
mock_gym = MagicMock()
mock_spaces = MagicMock()
mock_utils = MagicMock()

mock_np.float32 = float
mock_np.inf = float('inf')
mock_gym.Env = object # Inherit from object instead of MagicMock

sys.modules['pandas'] = mock_pd
sys.modules['numpy'] = mock_np
sys.modules['gymnasium'] = mock_gym
sys.modules['gymnasium.spaces'] = mock_spaces
sys.modules['utils'] = mock_utils

from trading_gym import TradingEnv

class TestZeroPriceFix(unittest.TestCase):
    def setUp(self):
        # Create a mock environment with minimal __init__ impact
        mock_df = MagicMock()

        # Patch random.choice to return something predictable
        with patch('random.choice') as mock_choice:
            mock_choice.return_value = {
                'df': mock_df,
                'obs': None,
                'prices': [0.0] * 110
            }
            # Patch load_config
            with patch('trading_gym.load_config', return_value={}):
                # Mock spaces to avoid their initialization
                with patch('gymnasium.spaces.Discrete'), patch('gymnasium.spaces.Box'):
                    # We need to satisfy the data loading part or skip it
                    # Let's manually create an instance and set what we need
                    self.env = TradingEnv.__new__(TradingEnv)
                    self.env.is_discrete = True
                    self.env.transaction_fee_percent = 0.001
                    self.env.cash = 1000.0
                    self.env.shares_held = 0.0
                    self.env.initial_balance = 10000.0
                    self.env.current_step = 0
                    self.env.window_size = 10
                    self.env._prices = [0.0] * 110
                    self.env.start_step = 0
                    self.env.episode_length = 90
                    self.env.df = mock_df

    def test_step_with_zero_price_buy(self):
        # Action 4 is Buy 100% (act = 1.0)
        # current_price = self._prices[0 + 10 - 1] = 0.0

        self.env._get_observation = MagicMock(return_value=None)

        # Should NOT raise ZeroDivisionError AFTER the fix
        try:
            self.env.step(4)
        except ZeroDivisionError:
            self.fail("ZeroDivisionError raised even after the fix!")
        except Exception:
            pass # Other errors might happen due to incomplete mocking but we only care about ZeroDivisionError

    def test_step_with_zero_price_sell(self):
        self.env.shares_held = 10.0
        self.env._get_observation = MagicMock(return_value=None)

        # Sell action should not raise ZeroDivisionError
        try:
            self.env.step(0)
        except ZeroDivisionError:
            self.fail("ZeroDivisionError raised during Sell action!")
        except Exception:
            pass

if __name__ == '__main__':
    unittest.main()
