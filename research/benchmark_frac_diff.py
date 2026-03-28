import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data_factory import frac_diff_ffd

def run_benchmark():
    # Generate some sample data: 5 tickers, 5000 days of data
    n_days = 5000
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
    data = np.random.randn(n_days, len(tickers)).cumsum(axis=0) + 100
    df = pd.DataFrame(data, columns=tickers, index=pd.date_range('2000-01-01', periods=n_days))

    # Add some NaNs to test the np.isfinite check
    for col in df.columns:
        mask = np.random.choice([True, False], size=n_days, p=[0.01, 0.99])
        df.loc[mask, col] = np.nan

    print(f"Starting benchmark with {n_days} rows and {len(tickers)} columns...")

    start_time = time.time()
    result = frac_diff_ffd(df, d=0.4)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Original frac_diff_ffd took: {duration:.4f} seconds")
    return duration

if __name__ == "__main__":
    run_benchmark()
