import unittest
import pandas as pd
import numpy as np
from utils import flatten_multiindex_columns

class TestUtils(unittest.TestCase):
    def test_flatten_multiindex_columns(self):
        # Create a MultiIndex DataFrame similar to yfinance structure
        # Level 0: Attribute (Close, Open)
        # Level 1: Ticker (NVDA)
        arrays = [
            ['Close', 'Open'],
            ['NVDA', 'NVDA']
        ]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['Price', 'Ticker'])
        df = pd.DataFrame(np.random.randn(3, 2), columns=index)

        # Verify it is MultiIndex
        self.assertIsInstance(df.columns, pd.MultiIndex)

        # Flatten
        df = flatten_multiindex_columns(df)

        # Verify it is NOT MultiIndex
        self.assertNotIsInstance(df.columns, pd.MultiIndex)
        self.assertEqual(list(df.columns), ['Close', 'Open'])

    def test_standard_columns(self):
        # Create a standard DataFrame
        df = pd.DataFrame(np.random.randn(3, 2), columns=['Close', 'Open'])

        # Flatten (should be no-op)
        df = flatten_multiindex_columns(df)

        # Verify columns are unchanged
        self.assertEqual(list(df.columns), ['Close', 'Open'])

if __name__ == '__main__':
    unittest.main()
