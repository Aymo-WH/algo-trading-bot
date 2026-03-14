import unittest
from unittest.mock import MagicMock, patch, call
import os
import sys

# Add parent directory to path to import data_factory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock missing dependencies
sys.modules['numpy'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['yfinance'] = MagicMock()
sys.modules['nltk'] = MagicMock()
sys.modules['nltk.sentiment'] = MagicMock()
sys.modules['nltk.sentiment.vader'] = MagicMock()
sys.modules['statsmodels'] = MagicMock()
sys.modules['statsmodels.tsa'] = MagicMock()
sys.modules['statsmodels.tsa.stattools'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.stats'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.decomposition'] = MagicMock()
sys.modules['sklearn.preprocessing'] = MagicMock()

import data_factory

class TestMockSentimentBatch(unittest.TestCase):
    def setUp(self):
        # We need to use a local import or re-mock for EACH test to avoid state bleeding
        # because of how we're mocking numpy
        self.sia = MagicMock()
        self.sia.polarity_scores.return_value = {'compound': 0.5}
        data_factory._MOCK_HEADLINE_SCORES = None

    @patch('data_factory.np')
    def test_get_mock_sentiment_batch_logic(self, mock_np):
        """
        Tests the logic of get_mock_sentiment_batch by mocking numpy functions
        and verifying they are called correctly.
        """
        n = 10
        # Setup mock returns
        # Important: the return values MUST behave like what the code expects (e.g. support addition)
        # Since we're using MagicMocks, they might return other MagicMocks.
        # Let's use simple values and side_effects.

        mock_np.array.return_value = [0.5] * len(data_factory.MOCK_HEADLINES)

        # When selected_scores + noise happens, it will be a mock addition.
        # To test the result, we can define the behavior of the addition or just
        # check that the correct calls were made.

        mock_scores = MagicMock()
        mock_np.random.choice.return_value = mock_scores

        mock_noise = MagicMock()
        mock_np.random.uniform.return_value = mock_noise

        mock_final = MagicMock()
        mock_scores.__add__.return_value = mock_final

        mock_clipped = MagicMock()
        mock_np.clip.return_value = mock_clipped

        results = data_factory.get_mock_sentiment_batch(n, self.sia)

        # Verify result is the clipped mock
        self.assertEqual(results, mock_clipped)

        # Verify numpy calls
        mock_np.random.choice.assert_called_once()
        mock_np.random.uniform.assert_called_once_with(-0.1, 0.1, size=n)
        mock_np.clip.assert_called_once_with(mock_final, -1.0, 1.0)

    @patch('data_factory.np')
    def test_get_mock_sentiment_batch_zero(self, mock_np):
        n = 0
        mock_np.array.return_value = [0.5]

        mock_scores = MagicMock()
        mock_np.random.choice.return_value = mock_scores

        mock_noise = MagicMock()
        mock_np.random.uniform.return_value = mock_noise

        mock_final = MagicMock()
        mock_scores.__add__.return_value = mock_final

        mock_clipped = MagicMock()
        mock_np.clip.return_value = mock_clipped

        results = data_factory.get_mock_sentiment_batch(n, self.sia)
        self.assertEqual(results, mock_clipped)

        mock_np.random.choice.assert_called_once_with(mock_np.array.return_value, size=0)

if __name__ == '__main__':
    unittest.main()
