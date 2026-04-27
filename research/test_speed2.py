import time
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.core.optimize_barriers as ob
from research.benchmark import evaluate_barriers_optimized

if __name__ == "__main__":
    np.random.seed(42)
    paths = np.random.normal(0, 1, size=(100000, 15))
    sigma = 1.0
    pt_grid = np.arange(0.5, 3.25, 0.25)
    sl_grid = np.arange(0.5, 3.25, 0.25)

    start_time = time.time()
    for _ in range(50):
        evaluate_barriers_optimized(paths, sigma, pt_grid, sl_grid)
    end_time = time.time()
    print(f"Benchmark Optimized Time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    for _ in range(50):
        ob.evaluate_barriers(paths, sigma, pt_grid, sl_grid)
    end_time = time.time()
    print(f"Optimize_barriers Time: {end_time - start_time:.4f} seconds")
