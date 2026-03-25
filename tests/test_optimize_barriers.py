import unittest
import numpy as np
import pandas as pd
from core.optimize_barriers import estimate_ou_parameters

class TestOptimizeBarriers(unittest.TestCase):

    def test_estimate_ou_parameters_happy_path(self):
        """
        Tests that estimate_ou_parameters correctly computes phi and sigma
        for a valid, sufficiently long sequence of varying prices.
        """
        # Create a series of varying prices (e.g., a simple sine wave or trend)
        prices = pd.Series(np.linspace(100, 110, 50) + np.sin(np.linspace(0, 10, 50)))

        phi, sigma = estimate_ou_parameters(prices, window=5)

        # Verify types and that they are not NaNs
        self.assertIsInstance(phi, float)
        self.assertIsInstance(sigma, float)
        self.assertFalse(np.isnan(phi))
        self.assertFalse(np.isnan(sigma))
        self.assertGreater(sigma, 0.0)

    def test_estimate_ou_parameters_short_series(self):
        """
        Tests that estimate_ou_parameters safely returns (np.nan, np.nan)
        if the series length is less than or equal to the rolling window.
        This leads to fewer than 2 valid points after dropping NaNs.
        """
        # Create a series shorter than the window
        prices = pd.Series([100.0, 101.0, 102.0])
        phi, sigma = estimate_ou_parameters(prices, window=5)

        self.assertTrue(np.isnan(phi))
        self.assertTrue(np.isnan(sigma))

    def test_estimate_ou_parameters_constant_prices(self):
        """
        Tests that estimate_ou_parameters safely returns (np.nan, np.nan)
        when prices are constant, as this causes zero variance in X and
        would lead to division by zero.
        """
        prices = pd.Series([100.0] * 50)
        phi, sigma = estimate_ou_parameters(prices, window=5)

        self.assertTrue(np.isnan(phi))
        self.assertTrue(np.isnan(sigma))

    def test_estimate_ou_parameters_with_nans(self):
        """
        Tests that estimate_ou_parameters correctly drops NaNs and still
        computes valid values if enough valid data points remain.
        """
        # Create prices with NaNs scattered
        prices_array = np.linspace(100, 110, 50) + np.sin(np.linspace(0, 10, 50))
        prices_array[10:15] = np.nan
        prices = pd.Series(prices_array)

        phi, sigma = estimate_ou_parameters(prices, window=5)

        self.assertIsInstance(phi, float)
        self.assertIsInstance(sigma, float)
        self.assertFalse(np.isnan(phi))
        self.assertFalse(np.isnan(sigma))
        self.assertGreater(sigma, 0.0)

if __name__ == '__main__':
    unittest.main()
