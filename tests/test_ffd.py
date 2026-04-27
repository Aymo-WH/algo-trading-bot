import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

class TestFFD(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Localized mocking to avoid global module poisoning
        cls.mocks = {
            "numpy": MagicMock(),
            "pandas": MagicMock(),
            "joblib": MagicMock(),
            "yfinance": MagicMock(),
            "nltk": MagicMock(),
            "nltk.sentiment.vader": MagicMock(),
            "sklearn.decomposition": MagicMock(),
            "sklearn.preprocessing": MagicMock(),
            "statsmodels.tsa.stattools": MagicMock(),
            "scipy": MagicMock(),
            "scipy.stats": MagicMock(),
            "core.optimize_barriers": MagicMock(),
        }

        # Setup numpy specific mock behavior
        cls.last_arr_mock = None
        def mock_array(data):
            arr = MagicMock()
            arr.tolist.return_value = data
            arr.reshape.return_value = arr
            arr.__getitem__.side_effect = lambda x: data[x]
            cls.last_arr_mock = arr
            return arr
        cls.mocks["numpy"].array.side_effect = mock_array

        # Start patches
        cls.patchers = [patch.dict("sys.modules", cls.mocks)]
        for p in cls.patchers:
            p.start()

        # Import after mocking
        global get_weights_ffd
        from data_factory import get_weights_ffd

    @classmethod
    def tearDownClass(cls):
        # Stop patches
        for p in cls.patchers:
            p.stop()

    def test_get_weights_ffd_d1(self):
        # d=1.0 should give [1, -1] which reversed is [-1, 1]
        weights = get_weights_ffd(1.0)
        self.mocks["numpy"].array.assert_any_call([-1.0, 1.0])

    def test_get_weights_ffd_d0(self):
        # d=0.0 should give [1] which reversed is [1]
        weights = get_weights_ffd(0.0)
        self.mocks["numpy"].array.assert_any_call([1.0])

    def test_get_weights_ffd_threshold(self):
        # d=0.5, thres=0.1 -> w = [1.0, -0.5, -0.125]. Reversed: [-0.125, -0.5, 1.0]
        get_weights_ffd(0.5, thres=0.1)
        self.mocks["numpy"].array.assert_any_call([-0.125, -0.5, 1.0])

    def test_output_shape_mock(self):
        # Verify that reshape was called to make it a column vector
        weights = get_weights_ffd(0.5)
        self.last_arr_mock.reshape.assert_called_with(-1, 1)

if __name__ == "__main__":
    unittest.main()
