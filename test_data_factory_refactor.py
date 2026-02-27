import unittest
import pandas as pd
import numpy as np
import data_factory

class TestDataFactoryRefactor(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame with enough data for indicators
        dates = pd.date_range(start='2020-01-01', periods=100)
        self.df = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(105, 205, 100),
            'Low': np.random.uniform(95, 195, 100)
        }, index=dates)

        # Ensure High >= Close and Low <= Close for sanity (though math works regardless)
        self.df['High'] = np.maximum(self.df['High'], self.df['Close'])
        self.df['Low'] = np.minimum(self.df['Low'], self.df['Close'])

    def test_calculate_rsi(self):
        # Expected Logic (from original code)
        df_expected = self.df.copy()
        delta = df_expected['Close'].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss
        expected_rsi = 100 - (100 / (1 + rs))
        expected_rsi.name = 'RSI'

        # Actual Logic (using new function)
        if hasattr(data_factory, 'calculate_rsi'):
            df_actual = self.df.copy()
            df_actual = data_factory.calculate_rsi(df_actual)
            pd.testing.assert_series_equal(df_actual['RSI'], expected_rsi, rtol=1e-5)
        else:
            self.skipTest("calculate_rsi not yet implemented")

    def test_calculate_macd(self):
        # Expected Logic
        df_expected = self.df.copy()
        exp1 = df_expected['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_expected['Close'].ewm(span=26, adjust=False).mean()
        expected_macd = exp1 - exp2
        expected_macd.name = 'MACD'
        expected_signal = expected_macd.ewm(span=9, adjust=False).mean()
        expected_signal.name = 'Signal_Line'

        # Actual Logic
        if hasattr(data_factory, 'calculate_macd'):
            df_actual = self.df.copy()
            df_actual = data_factory.calculate_macd(df_actual)
            pd.testing.assert_series_equal(df_actual['MACD'], expected_macd, rtol=1e-5)
            pd.testing.assert_series_equal(df_actual['Signal_Line'], expected_signal, rtol=1e-5)
        else:
            self.skipTest("calculate_macd not yet implemented")

    def test_calculate_bollinger_bands(self):
        # Expected Logic
        df_expected = self.df.copy()
        sma_20 = df_expected['Close'].rolling(window=20).mean()
        std_20 = df_expected['Close'].rolling(window=20).std()
        expected_upper = sma_20 + 2 * std_20
        expected_upper.name = 'BB_Upper'
        expected_lower = sma_20 - 2 * std_20
        expected_lower.name = 'BB_Lower'

        # Actual Logic
        if hasattr(data_factory, 'calculate_bollinger_bands'):
            df_actual = self.df.copy()
            df_actual = data_factory.calculate_bollinger_bands(df_actual)
            pd.testing.assert_series_equal(df_actual['BB_Upper'], expected_upper, rtol=1e-5)
            pd.testing.assert_series_equal(df_actual['BB_Lower'], expected_lower, rtol=1e-5)
        else:
            self.skipTest("calculate_bollinger_bands not yet implemented")

    def test_calculate_atr(self):
        # Expected Logic
        df_expected = self.df.copy()
        high_low = df_expected['High'] - df_expected['Low']
        high_close = (df_expected['High'] - df_expected['Close'].shift()).abs()
        low_close = (df_expected['Low'] - df_expected['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        expected_atr = tr.rolling(window=14).mean()
        expected_atr.name = 'ATR'

        # Actual Logic
        if hasattr(data_factory, 'calculate_atr'):
            df_actual = self.df.copy()
            df_actual = data_factory.calculate_atr(df_actual)
            pd.testing.assert_series_equal(df_actual['ATR'], expected_atr, rtol=1e-5)
        else:
            self.skipTest("calculate_atr not yet implemented")

    def test_extract_ticker_data_multiindex(self):
        # Simulate MultiIndex DataFrame from yfinance (Ticker, Attribute)
        # Note: yfinance group_by='ticker' results in top level Ticker
        arrays = [
            ['NVDA', 'NVDA', 'AAPL', 'AAPL'],
            ['Close', 'Open', 'Close', 'Open']
        ]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['Ticker', 'Price'])
        data = pd.DataFrame(np.random.randn(5, 4), columns=index)

        if hasattr(data_factory, 'extract_ticker_data'):
            df_nvda = data_factory.extract_ticker_data(data, 'NVDA')
            self.assertIsInstance(df_nvda, pd.DataFrame)
            self.assertIn('Close', df_nvda.columns)
            self.assertIn('Open', df_nvda.columns)
            # Should have dropped the ticker level
            self.assertFalse(isinstance(df_nvda.columns, pd.MultiIndex))
        else:
            self.skipTest("extract_ticker_data not yet implemented")

    def test_extract_ticker_data_flat(self):
        # Simulate Flat DataFrame (single ticker or already processed)
        data = pd.DataFrame({
            'Close': np.random.randn(5),
            'Open': np.random.randn(5)
        })

        if hasattr(data_factory, 'extract_ticker_data'):
            df_res = data_factory.extract_ticker_data(data, 'NVDA') # Ticker arg ignored if structure is flat/fallback
            pd.testing.assert_frame_equal(df_res, data)
        else:
            self.skipTest("extract_ticker_data not yet implemented")

if __name__ == '__main__':
    unittest.main()
