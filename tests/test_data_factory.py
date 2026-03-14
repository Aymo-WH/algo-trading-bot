import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies before importing data_factory
# This is a project-wide convention to handle the absence of these libraries in the sandbox environment.
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
        # Reset mock calls
        mock_np.reset_mock()

    def test_get_mock_sentiment_batch_logic(self):
        """
        Verify the core logic of get_mock_sentiment_batch:
        1. Samples from cached scores
        2. Adds uniform noise
        3. Clips the result
        """
        n = 5
        sia = MagicMock()
        sia.polarity_scores.return_value = {'compound': 0.5}

        # Setup mocks to return specific values to trace the flow
        mock_selected = MagicMock(name="selected_scores")
        mock_noise = MagicMock(name="noise")
        mock_sum = MagicMock(name="sum")
        mock_clipped = MagicMock(name="clipped")

        mock_np.random.choice.return_value = mock_selected
        mock_np.random.uniform.return_value = mock_noise
        mock_selected.__add__.return_value = mock_sum
        mock_np.clip.return_value = mock_clipped

        result = data_factory.get_mock_sentiment_batch(n, sia)

        # 1. Verify sampling
        mock_np.random.choice.assert_called_once()
        args, kwargs = mock_np.random.choice.call_args
        self.assertEqual(kwargs.get('size'), n)

        # 2. Verify noise generation
        mock_np.random.uniform.assert_called_once_with(-0.1, 0.1, size=n)

        # 3. Verify addition (noise added to selected scores)
        mock_selected.__add__.assert_called_once_with(mock_noise)

        # 4. Verify clipping
        mock_np.clip.assert_called_once_with(mock_sum, -1.0, 1.0)

        # 5. Verify the final result is the clipped value
        self.assertEqual(result, mock_clipped)

    def test_get_mock_sentiment_batch_caching(self):
        """
        Verify that _get_cached_scores implements caching.
        """
        n = 5
        sia = MagicMock()
        sia.polarity_scores.return_value = {'compound': 0.5}

        # First call: should compute scores
        data_factory.get_mock_sentiment_batch(n, sia)
        self.assertEqual(mock_np.array.call_count, 1)

        # Second call: should use cached scores
        data_factory.get_mock_sentiment_batch(n, sia)
        self.assertEqual(mock_np.array.call_count, 1)

    def test_get_mock_sentiment_batch_size(self):
        """
        Verify that the function requests the correct batch size 'n' from all operations.
        """
        sia = MagicMock()
        sia.polarity_scores.return_value = {'compound': 0.1}

        for n in [1, 10, 100]:
            mock_np.random.choice.reset_mock()
            mock_np.random.uniform.reset_mock()

            data_factory.get_mock_sentiment_batch(n, sia)

            self.assertEqual(mock_np.random.choice.call_args[1]['size'], n)
            self.assertEqual(mock_np.random.uniform.call_args[1]['size'], n)

if __name__ == '__main__':
    unittest.main()
