import sys
import unittest
from unittest.mock import MagicMock, patch
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import data_factory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import data_factory

class TestDataFactorySecurity(unittest.TestCase):

    @patch('data_factory.load_config')
    @patch('data_factory.yf.download')
    @patch('data_factory.SentimentIntensityAnalyzer')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    @patch('data_factory.download_nltk_data')
    def test_fetch_data_sanitization(self, mock_download_nltk, mock_to_csv, mock_makedirs, mock_sia_cls, mock_download, mock_load_config):
        print("Starting security test...")

        # Configure config with various ticker types:
        # 1. Malicious path traversal
        # 2. Standard ticker
        # 3. Index ticker (starts with ^)
        mock_load_config.return_value = {
            'tickers': ['../malicious', 'NVDA', '^GSPC'],
            'start_date': '2020-01-01',
            'end_date': '2020-01-10',
            'train_start_date': '2020-01-01',
            'train_end_date': '2020-01-05',
            'test_start_date': '2020-01-06'
        }

        # Create a real but small pandas DataFrame so we don't need to mock all of pandas operations
        dates = pd.date_range(start='2020-01-01', periods=30)
        np.random.seed(42) # For reproducibility
        df = pd.DataFrame({
            'Open': np.random.rand(30) * 100,
            'High': np.random.rand(30) * 100,
            'Low': np.random.rand(30) * 100,
            'Close': np.random.rand(30) * 100,
            'Volume': np.random.randint(1000, 10000, size=30)
        }, index=dates)

        # Mock yfinance return
        # yfinance returns a MultiIndex columns if group_by='ticker' is used with multiple tickers
        # However, if data_factory does `data[ticker]`, we can mock `data` as a dict-like object
        class MockDataWrapper:
            def __init__(self, df):
                self.df = df
                # Make columns NOT a MultiIndex to take the fallback path in data_factory
                # OR make it a MultiIndex and mock __getitem__
                self.columns = pd.MultiIndex.from_tuples([('Close', 'NVDA')])
            def __getitem__(self, key):
                return self.df.copy()

        mock_data = MockDataWrapper(df)
        mock_download.return_value = mock_data

        # Mock SentimentIntensityAnalyzer instance to avoid NLTK downloads/processing
        mock_sia_instance = MagicMock()
        mock_sia_instance.polarity_scores.return_value = {'compound': 0.5}
        mock_sia_cls.return_value = mock_sia_instance

        # Run the function
        try:
            data_factory.fetch_data()
        except Exception as e:
            self.fail(f"Exception during execution: {e}")

        # Check results
        print("\nChecking file paths passed to to_csv:")

        saved_paths = []
        for call in mock_to_csv.call_args_list:
            args, kwargs = call
            if args:
                saved_paths.append(args[0])
                print(f" - {args[0]}")

        # Verification

        # 1. Malicious Ticker '../malicious' should be sanitized to 'malicious'
        self.assertIn('data/train/malicious_data.csv', saved_paths, "Sanitized malicious ticker not saved to correct path")
        self.assertIn('data/test/malicious_data.csv', saved_paths, "Sanitized malicious ticker not saved to correct path")

        # Ensure NO traversal
        for path in saved_paths:
            self.assertNotIn('../', path, "Path traversal detected in saved file path")

        # 2. Standard Ticker 'NVDA' should be present
        self.assertIn('data/train/NVDA_data.csv', saved_paths, "Valid ticker NVDA not saved")

        # 3. Index Ticker '^GSPC' should be present
        self.assertIn('data/train/^GSPC_data.csv', saved_paths, "Index ticker ^GSPC not saved (likely rejected by regex)")

if __name__ == '__main__':
    unittest.main()
