import sys
import os
from unittest.mock import MagicMock

# Simple mock to avoid recursion and heavy dependencies
class SimpleMock:
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return SimpleMock()
    def __call__(self, *args, **kwargs):
        return SimpleMock()

mock_modules = [
    "yfinance", "pandas", "numpy", "nltk", "nltk.sentiment.vader",
    "sklearn.decomposition", "sklearn.preprocessing", "joblib",
    "statsmodels.tsa.stattools", "scipy", "scipy.stats", "core.optimize_barriers"
]
for mod in mock_modules:
    sys.modules[mod] = SimpleMock()

# Now we can import fetch_data
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from data_factory import fetch_data
except ImportError as e:
    print(f"Still failed to import: {e}")
    sys.exit(1)

def test_security():
    print("Testing Security Fix for Path Traversal in data_factory.py (Unit Test)...")

    from io import StringIO

    test_cases = [
        ("../README.md", "Path Traversal Attempt"),
        ("/etc/passwd", "Absolute Path Traversal"),
        ("config/../README.md", "Complex Traversal"),
    ]

    for path, desc in test_cases:
        print(f"\n[TEST] {desc} ({path})")
        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            fetch_data(path)
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        if "[ERROR] Security: Configuration path" in output:
            print(f"PASS: {desc} correctly blocked.")
        else:
            print(f"FAIL: {desc} NOT blocked. Output:\n{output}")

    print("\n[TEST] Legitimate config (config/config_phase1.json)")
    captured_output = StringIO()
    sys.stdout = captured_output
    try:
        # We use a path that exists to avoid FileNotFoundError which might happen after security check
        fetch_data("config/config_phase1.json")
    except Exception as e:
        # We expect it might fail later due to mocks, that's fine
        pass
    finally:
        sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    if "[ERROR] Security:" in output:
        print(f"FAIL: Legitimate path blocked. Output:\n{output}")
    else:
        print("PASS: Legitimate path not blocked by security check.")

if __name__ == "__main__":
    test_security()
