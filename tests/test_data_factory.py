import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_factory import construct_dollar_bars, fetch_data, rolling_sadf_np

class TestDataFactory(unittest.TestCase):

    @unittest.mock.patch('data_factory.yf')
    @unittest.mock.patch('data_factory.os.makedirs')
    @unittest.mock.patch('data_factory.pd.DataFrame.to_csv')
    @unittest.mock.patch('data_factory.joblib.dump')
    @unittest.mock.patch('core.utils.load_config')
    def test_microstructural_features_integrated(self, mock_load_config, mock_dump, mock_to_csv, mock_makedirs, mock_yfinance):
        # Mock config
        mock_load_config.return_value = {
            'tickers': ['TEST'],
            'transaction_fee_percent': 0.001,
            'data_window_days': 730
        }

        # Create a mock dataframe that yfinance will return
        dates = pd.date_range(start='2020-01-01', periods=500, freq='h')
        mock_df = pd.DataFrame({
            'Open': np.linspace(10, 20, 500),
            'High': np.linspace(10.5, 20.5, 500),
            'Low': np.linspace(9.5, 19.5, 500),
            'Close': np.linspace(10.2, 20.2, 500) + np.random.normal(0, 0.1, 500),
            'Volume': np.random.randint(100, 1000, 500).astype(float)
        }, index=dates)

        # Make yfinance return this df
        mock_yfinance.download.return_value = mock_df

        # We also need to mock `get_rolling_barriers` and `adfuller` to avoid complex test setups
        with unittest.mock.patch('data_factory.get_rolling_barriers') as mock_barriers:
            mock_barriers_df = pd.DataFrame({
                'Optimal_PT': [2.0]*500,
                'Optimal_SL': [2.0]*500
            }, index=mock_df.index) # index won't perfectly match dollar bars but that's ok, fillna handles it
            mock_barriers.return_value = mock_barriers_df

            with unittest.mock.patch('data_factory.adfuller') as mock_adf:
                mock_adf.return_value = (0, 0.01) # fake p-value < 0.05

                # Mock construct_dollar_bars to just return the df without compressing to avoid size issues
                with unittest.mock.patch('data_factory.construct_dollar_bars', return_value=mock_df):

                    # Capture the dataframe right before PCA is applied
                    captured_df = None
                    original_dropna = pd.DataFrame.dropna

                    def mock_dropna(self_df, *args, **kwargs):
                        nonlocal captured_df
                        captured_df = self_df
                        return original_dropna(self_df, *args, **kwargs)

                    with unittest.mock.patch('pandas.DataFrame.dropna', side_effect=mock_dropna, autospec=True):
                        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{"tickers": ["TEST"], "transaction_fee_percent": 0.001, "data_window_days": 730}')):
                            fetch_data('dummy_path.json')

                        # Assertions on the captured dataframe
                        self.assertIsNotNone(captured_df)
                        self.assertIn('VPIN', captured_df.columns)
                        self.assertIn('Amihud_Illiq', captured_df.columns)
                        self.assertIn('Kyles_Lambda', captured_df.columns)
                        self.assertIn('SADF', captured_df.columns)

    def test_construct_dollar_bars(self):
        # Construct dummy DataFrame with a DatetimeIndex and 300 rows
        dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(300)]

        # Set the first 210 rows to Close=10 and Volume=10
        # Set the remaining 90 rows to Close=10 and Volume=1
        closes = [10.0] * 300
        volumes = [10.0] * 210 + [1.0] * 90

        # Open, High, Low can be arbitrary but should make sense relative to Close
        opens = [9.5] * 300
        highs = [10.5] * 300
        lows = [9.0] * 300

        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        }, index=dates)

        # For the first 210 rows, Dollar Volume = 100 each. Total = 21,000.
        # M for the first 210 rows is calculated on row 209 (0-indexed) as 21,000 / 300 = 70.
        # Since M is backfilled, the threshold is 70 for the early rows.
        # With Dollar Volume = 100 per row, every single row from 0 to 209 generates a bar.
        # Then for the next 90 rows, Dollar Volume = 10 each.
        # M slowly decreases as the 210-row window drops the 100s and adds the 10s.
        # It takes several rows to accumulate enough Dollar Volume to hit the threshold.
        # We can predict the exact number of bars or at least ensure it's specifically calculating right.

        # Call the function
        result_df = construct_dollar_bars(df)

        # Assertions

        # We can calculate the expected number of bars.
        # For the first 210 rows (DV=100), M is 70 for the first 210 rows due to bfill and constant window sum.
        # Since DV > M (100 > 70), each of these 210 rows triggers exactly 1 bar.
        # For the remaining 90 rows (DV=10), M decreases.
        # M starts at 70 and drops.
        # But we know that exactly 210 bars are produced from the first 210 rows.
        # The remaining 90 rows have total DV = 900.
        # M during these rows starts at 69.7 and goes down to ~43 by the end.
        # So it takes ~5-7 rows to trigger a bar. We expect roughly 900 / ~50 ≈ 18 bars from the last 90 rows.
        # Total bars should be around 228.

        # Check that it compressed (fewer rows than input)
        self.assertLess(len(result_df), len(df), "Resulting DataFrame should have fewer rows than the input.")
        self.assertGreater(len(result_df), 0, "Resulting DataFrame should not be empty.")
        self.assertEqual(len(result_df), 225, "Expected exactly 225 dollar bars based on the mock data volume accumulation.")

        # Check columns are exactly ['Open', 'High', 'Low', 'Close', 'Volume']
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.assertListEqual(list(result_df.columns), expected_columns)

        # Check index name is 'Date'
        self.assertEqual(result_df.index.name, 'Date')

    def test_empty_dataframe(self):
        # Empty DataFrame with correct columns
        df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        result_df = construct_dollar_bars(df)

        # Result should be an empty DataFrame
        self.assertTrue(result_df.empty, "Resulting DataFrame should be empty.")

        # Columns might be preserved depending on pandas version/behavior,
        # but the main thing is that it handles the empty case without error
        if not result_df.empty:
            self.assertEqual(len(result_df), 0)

if __name__ == '__main__':
    unittest.main()
