import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies before importing data_factory
mock_np = MagicMock()
sys.modules['numpy'] = mock_np
sys.modules['pandas'] = MagicMock()
sys.modules['yfinance'] = MagicMock()
sys.modules['nltk'] = MagicMock()
sys.modules['nltk.sentiment'] = MagicMock()
sys.modules['nltk.sentiment.vader'] = MagicMock()
sys.modules['utils'] = MagicMock()

import data_factory

class TestDataFactory(unittest.TestCase):
    def setUp(self):
        # Reset the global cache before each test
        data_factory._MOCK_HEADLINE_SCORES = None
        mock_np.reset_mock()

    def test_get_mock_sentiment_batch(self):
        """
        Verify that get_mock_sentiment_batch generates a batch of scores
        of the requested size and uses the caching mechanism.
        """
        n = 5
        sia = MagicMock()
        sia.polarity_scores.return_value = {'compound': 0.5}

        # Mocking the return of np.clip to verify the flow
        mock_np.clip.return_value = "mock_output"

        # First call: verifies logic and caching initiation
        result = data_factory.get_mock_sentiment_batch(n, sia)

        self.assertEqual(result, "mock_output")
        self.assertEqual(mock_np.array.call_count, 1) # Caching: np.array called once
        self.assertEqual(mock_np.random.choice.call_args[1]['size'], n)
        self.assertEqual(mock_np.random.uniform.call_args[1]['size'], n)

        # Second call: verifies caching
        data_factory.get_mock_sentiment_batch(n, sia)
        self.assertEqual(mock_np.array.call_count, 1) # Still 1 due to cache

if __name__ == '__main__':
    unittest.main()
