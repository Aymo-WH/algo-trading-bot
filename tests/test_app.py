import sys
import os
from unittest.mock import MagicMock, patch, mock_open
import unittest
import json

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Mock dependencies before importing app
mock_gr = MagicMock()
mock_gr.themes.Monochrome.return_value = MagicMock()
# Make gr.Dataframe.update return the 'value' keyword argument for easier assertion
def mock_update(value=None):
    return value
mock_gr.Dataframe.update.side_effect = mock_update

sys.modules["gradio"] = mock_gr
sys.modules["evaluate_agents"] = MagicMock()

# Now import update_tickers from app
from app import update_tickers

class TestApp(unittest.TestCase):
    def test_update_tickers_success(self):
        config_data = {"tickers": ["AAPL", "MSFT"]}
        config_json = json.dumps(config_data)

        with patch("builtins.open", mock_open(read_data=config_json)):
            rows = update_tickers("Tech Equities")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], ["AAPL", 100, 101, "2%", "-2%", "1 day"])
        self.assertEqual(rows[1], ["MSFT", 100, 101, "2%", "-2%", "1 day"])

    def test_update_tickers_error_path_file_not_found(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            rows = update_tickers("Tech Equities")

        self.assertEqual(rows, [])

    def test_update_tickers_error_path_json_decode_error(self):
        with patch("builtins.open", mock_open(read_data="invalid json")):
            # Simulate JSONDecodeError when json.load is called
            with patch("json.load", side_effect=json.JSONDecodeError("msg", "doc", 0)):
                rows = update_tickers("Tech Equities")

        self.assertEqual(rows, [])

    def test_update_tickers_missing_key(self):
        config_data = {"not_tickers": ["AAPL"]}
        config_json = json.dumps(config_data)

        with patch("builtins.open", mock_open(read_data=config_json)):
            rows = update_tickers("Tech Equities")

        self.assertEqual(rows, [])

if __name__ == "__main__":
    unittest.main()
