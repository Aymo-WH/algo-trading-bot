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
            "sklearn.decomposition": MagicMock(),
            "sklearn.preprocessing": MagicMock(),
            "statsmodels.tsa.stattools": MagicMock(),
            "scipy": MagicMock(),
            "scipy.stats": MagicMock(),
            "core.optimize_barriers": MagicMock(),
            "numba": MagicMock(),
        }

        # Start patches
        # We must keep numpy in sys.modules because np.testing uses it internally.
        # So we mock everything EXCEPT numpy here to allow exact output matching.
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
        # Stop patch temporarily to allow native numpy import
        for p in self.patchers: p.stop()
        import numpy as np
        weights = get_weights_ffd(1.0)
        np.testing.assert_array_almost_equal(weights, np.array([[-1.0], [1.0]]))
        for p in self.patchers: p.start()

    def test_get_weights_ffd_d0(self):
        for p in self.patchers: p.stop()
        import numpy as np
        weights = get_weights_ffd(0.0)
        np.testing.assert_array_almost_equal(weights, np.array([[1.0]]))
        for p in self.patchers: p.start()

    def test_get_weights_ffd_threshold(self):
        for p in self.patchers: p.stop()
        import numpy as np
        weights = get_weights_ffd(0.5, thres=0.1)
        np.testing.assert_array_almost_equal(weights, np.array([[-0.125], [-0.5], [1.0]]))
        for p in self.patchers: p.start()

    def test_output_shape_mock(self):
        for p in self.patchers: p.stop()
        weights = get_weights_ffd(0.5)
        self.assertEqual(weights.shape[1], 1)
        for p in self.patchers: p.start()

if __name__ == "__main__":
    unittest.main()
