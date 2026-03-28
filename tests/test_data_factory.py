import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_factory import construct_dollar_bars

class TestDataFactory(unittest.TestCase):

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
