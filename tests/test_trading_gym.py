import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import unittest
import numpy as np
import pandas as pd
import shutil
from core.trading_gym import TradingEnv
from gymnasium import spaces

class TestTradingGym(unittest.TestCase):
    def setUp(self):
        TradingEnv._DATA_CACHE.clear()
        self.test_dir = 'test_data_gym'
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)

        self.valid_df = pd.DataFrame({
            'Close': np.random.rand(20),
            'Close_FFD': np.random.rand(20),
            'PCA_1': np.random.rand(20),
            'PCA_2': np.random.rand(20),
            'PCA_3': np.random.rand(20),
            'PCA_4': np.random.rand(20)
        })
        self.valid_df.to_csv(os.path.join(self.test_dir, 'valid_data.csv'), index=False)

    def tearDown(self):
        TradingEnv._DATA_CACHE.clear()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_observation_space_shape(self):
        env = TradingEnv(data_dir=self.test_dir)

        # Verify that observation space is a Box and has shape == (3,)
        self.assertIsInstance(env.observation_space, spaces.Box)
        self.assertEqual(env.observation_space.shape, (3,), "Observation space shape must be exactly (3,) per V2 specs.")

if __name__ == '__main__':
    unittest.main()
