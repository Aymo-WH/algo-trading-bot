import sys
from unittest.mock import MagicMock, patch, mock_open



import unittest
import utils

class TestConfigCaching(unittest.TestCase):
    def setUp(self):
        # Reset the cache before each test
        utils._CONFIG_CACHE = None

    def test_load_config_caching(self):
        config_data = '{"transaction_fee_percent": 0.005}'

        with patch("builtins.open", mock_open(read_data=config_data)) as mocked_file:
            # First call should read from file
            config1 = utils.load_config()
            self.assertEqual(config1.get('transaction_fee_percent'), 0.005)
            mocked_file.assert_called_once_with('config.json', 'r')

            # Second call should use cache and NOT read from file
            mocked_file.reset_mock()
            config2 = utils.load_config()
            self.assertEqual(config2.get('transaction_fee_percent'), 0.005)
            mocked_file.assert_not_called()

            self.assertIs(config1, config2)

    def test_load_config_file_not_found_caching(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            # First call fails to find file, should return empty dict
            config1 = utils.load_config()
            self.assertEqual(config1, {})

            # Second call should still return empty dict from cache without trying to open file
            with patch("builtins.open") as mocked_file:
                config2 = utils.load_config()
                self.assertEqual(config2, {})
                mocked_file.assert_not_called()

if __name__ == '__main__':
    unittest.main()
