import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the current directory to sys.path so we can import data_factory
sys.path.append(os.getcwd())

# Mock missing dependencies
sys.modules['yfinance'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['nltk'] = MagicMock()
sys.modules['nltk.sentiment'] = MagicMock()
sys.modules['nltk.sentiment.vader'] = MagicMock()

import data_factory

class TestGetMockSentiment(unittest.TestCase):
    def setUp(self):
        self.mock_sia = MagicMock()
        # Default mock for polarity_scores
        self.mock_sia.polarity_scores.return_value = {'compound': 0.5}

    def test_get_mock_sentiment_returns_float(self):
        """Test that the function returns a float."""
        result = data_factory.get_mock_sentiment(self.mock_sia)
        self.assertIsInstance(result, float)

    def test_get_mock_sentiment_range(self):
        """Test that the function returns a value within [-1.0, 1.0] over multiple runs."""
        for _ in range(100):
            result = data_factory.get_mock_sentiment(self.mock_sia)
            self.assertGreaterEqual(result, -1.0)
            self.assertLessEqual(result, 1.0)

    @patch('data_factory.random.choice')
    @patch('data_factory.random.uniform')
    def test_get_mock_sentiment_logic(self, mock_uniform, mock_choice):
        """Test the core logic: headline selection, sentiment scoring, and noise addition."""
        mock_choice.return_value = "Company reports record earnings."
        mock_uniform.return_value = 0.05
        self.mock_sia.polarity_scores.return_value = {'compound': 0.5}

        result = data_factory.get_mock_sentiment(self.mock_sia)

        # Verify sia.polarity_scores was called with the chosen headline
        self.mock_sia.polarity_scores.assert_called_with("Company reports record earnings.")
        # Verify the result is score + noise (0.5 + 0.05 = 0.55)
        self.assertAlmostEqual(result, 0.55)

    @patch('data_factory.random.uniform')
    def test_get_mock_sentiment_clipping_upper(self, mock_uniform):
        """Test upper bound clipping."""
        mock_uniform.return_value = 0.1
        self.mock_sia.polarity_scores.return_value = {'compound': 0.95}
        # 0.95 + 0.1 = 1.05 -> should be clipped to 1.0
        result = data_factory.get_mock_sentiment(self.mock_sia)
        self.assertEqual(result, 1.0)

    @patch('data_factory.random.uniform')
    def test_get_mock_sentiment_clipping_lower(self, mock_uniform):
        """Test lower bound clipping."""
        mock_uniform.return_value = -0.1
        self.mock_sia.polarity_scores.return_value = {'compound': -0.95}
        # -0.95 - 0.1 = -1.05 -> should be clipped to -1.0
        result = data_factory.get_mock_sentiment(self.mock_sia)
        self.assertEqual(result, -1.0)

if __name__ == '__main__':
    unittest.main()
