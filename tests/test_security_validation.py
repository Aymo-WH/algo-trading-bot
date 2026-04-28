import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import unittest
import sys

# Remove the bad mocks that broke actual test logic: test_security_validation actually relies on pandas and numpy

import os
import shutil
from core.trading_gym import TradingEnv
import pandas as pd
import numpy as np

class TestTradingEnvSecurity(unittest.TestCase):
    def clean_directory(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        TradingEnv._DATA_CACHE.clear()

    def setUp(self):
        TradingEnv._DATA_CACHE.clear()
        self.test_dir = 'test_data_security'
        self.clean_directory()

        # valid dataframe
        self.valid_df = pd.DataFrame({
            'Close': np.random.rand(20),
            'Close_FFD': np.random.rand(20),
            'PCA_1': np.random.rand(20),
            'PCA_2': np.random.rand(20),
            'PCA_3': np.random.rand(20),
            'PCA_4': np.random.rand(20)
        })

    def tearDown(self):
        TradingEnv._DATA_CACHE.clear()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_missing_columns_recovery(self):
        self.clean_directory()
        self.valid_df.to_csv(os.path.join(self.test_dir, 'valid_data.csv'), index=False)
        bad_df = self.valid_df.drop(columns=['PCA_1'])
        bad_df.to_csv(os.path.join(self.test_dir, 'missing_col_data.csv'), index=False)

        try:
            env = TradingEnv(data_dir=self.test_dir)
            self.assertEqual(len(env.dfs), 1)
        except Exception as e:
            self.fail(f"TradingEnv crashed on bad file: {e}")

    def test_bad_types_recovery(self):
        self.clean_directory()
        self.valid_df.to_csv(os.path.join(self.test_dir, 'valid_data.csv'), index=False)
        bad_df = self.valid_df.copy()
        bad_df['Close'] = "NotANumber"
        bad_df.to_csv(os.path.join(self.test_dir, 'bad_type_data.csv'), index=False)

        try:
            env = TradingEnv(data_dir=self.test_dir)
            self.assertEqual(len(env.dfs), 1)
        except Exception as e:
            self.fail(f"TradingEnv crashed on bad file: {e}")

    def test_all_bad_files(self):
        self.clean_directory()
        bad_df = self.valid_df.drop(columns=['PCA_1'])
        bad_df.to_csv(os.path.join(self.test_dir, 'missing_col_data.csv'), index=False)

        # Catch either ValueError or FileNotFoundError
        with self.assertRaises((ValueError, FileNotFoundError)):
            env = TradingEnv(data_dir=self.test_dir)

    def test_type_conversion(self):
        self.clean_directory()
        df_float64 = pd.DataFrame({
            'Close': np.array([100.0] * 12, dtype=np.float64),
            'Close_FFD': np.array([0.0] * 12, dtype=np.float64),
            'PCA_1': np.array([50.0] * 12, dtype=np.float64),
            'PCA_2': np.array([0.0] * 12, dtype=np.float64),
            'PCA_3': np.array([110.0] * 12, dtype=np.float64),
            'PCA_4': np.array([90.0] * 12, dtype=np.float64)
        })
        df_float64.to_csv(os.path.join(self.test_dir, 'float64_data.csv'), index=False)

        env = TradingEnv(data_dir=self.test_dir)

        self.assertEqual(len(env.dfs), 1)
        loaded_df = env.dfs[0]
        required_columns = ['Close', 'Close_FFD', 'PCA_1', 'PCA_2', 'PCA_3', 'PCA_4']
        for col in required_columns:
            self.assertEqual(loaded_df[col].dtype, np.float32)

    def test_observation_space_1d(self):
        self.clean_directory()
        df_float64 = pd.DataFrame({
            'Close': np.array([100.0] * 12, dtype=np.float64),
            'Close_FFD': np.array([0.0] * 12, dtype=np.float64),
            'PCA_1': np.array([50.0] * 12, dtype=np.float64),
            'PCA_2': np.array([0.0] * 12, dtype=np.float64),
            'PCA_3': np.array([110.0] * 12, dtype=np.float64),
            'PCA_4': np.array([90.0] * 12, dtype=np.float64)
        })
        df_float64.to_csv(os.path.join(self.test_dir, 'float64_data.csv'), index=False)

        env = TradingEnv(data_dir=self.test_dir)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (4,), "Observation space must be strictly 1D shape (4,)")

        obs, _, _, _, _ = env.step(env.action_space.sample() if hasattr(env, 'action_space') and env.action_space else 0)
        self.assertEqual(obs.shape, (4,), "Observation space must be strictly 1D shape (4,) after step")

if __name__ == '__main__':
    unittest.main()
