import unittest
import pandas as pd
import numpy as np
import os
import shutil
from trading_gym import TradingEnv

class TestTradingEnvSecurity(unittest.TestCase):
    def clean_directory(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        TradingEnv._DATA_CACHE.clear()

    def setUp(self):
        self.test_dir = 'test_data_security'
        self.clean_directory()

        # valid dataframe
        self.valid_df = pd.DataFrame({
            'Close': np.random.rand(20),
            'RSI': np.random.rand(20),
            'MACD': np.random.rand(20),
            'Sentiment_Score': np.random.rand(20),
            'BB_Upper': np.random.rand(20),
            'BB_Lower': np.random.rand(20),
            'ATR': np.random.rand(20)
        })

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        TradingEnv._DATA_CACHE.clear()

    def test_missing_columns_recovery(self):
        self.clean_directory()
        self.valid_df.to_csv(os.path.join(self.test_dir, 'valid_data.csv'), index=False)
        bad_df = self.valid_df.drop(columns=['RSI'])
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
        bad_df = self.valid_df.drop(columns=['RSI'])
        bad_df.to_csv(os.path.join(self.test_dir, 'missing_col_data.csv'), index=False)

        # Catch either ValueError or FileNotFoundError
        with self.assertRaises((ValueError, FileNotFoundError)):
            env = TradingEnv(data_dir=self.test_dir)

    def test_type_conversion(self):
        self.clean_directory()
        df_float64 = pd.DataFrame({
            'Close': np.array([100.0, 101.0], dtype=np.float64),
            'RSI': np.array([50.0, 51.0], dtype=np.float64),
            'MACD': np.array([0.0, 0.1], dtype=np.float64),
            'Sentiment_Score': np.array([0.5, 0.6], dtype=np.float64),
            'BB_Upper': np.array([110.0, 111.0], dtype=np.float64),
            'BB_Lower': np.array([90.0, 91.0], dtype=np.float64),
            'ATR': np.array([1.0, 1.1], dtype=np.float64)
        })
        df_float64.to_csv(os.path.join(self.test_dir, 'float64_data.csv'), index=False)

        env = TradingEnv(data_dir=self.test_dir)

        self.assertEqual(len(env.dfs), 1)
        loaded_df = env.dfs[0]
        required_columns = ['Close', 'RSI', 'MACD', 'Sentiment_Score', 'BB_Upper', 'BB_Lower', 'ATR']
        for col in required_columns:
            self.assertEqual(loaded_df[col].dtype, np.float32)

if __name__ == '__main__':
    unittest.main()
