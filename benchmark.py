import time
import numpy as np
from src.core.optimize_barriers import evaluate_barriers

# create synthetic paths
np.random.seed(42)
paths = np.random.normal(0, 1, size=(100000, 15))
sigma = 1.0
pt_grid = np.arange(0.5, 3.25, 0.25)
sl_grid = np.arange(0.5, 3.25, 0.25)

start_time = time.time()
for _ in range(10): # run multiple times
    evaluate_barriers(paths, sigma, pt_grid, sl_grid)
end_time = time.time()

print(f"Baseline Time: {end_time - start_time:.4f} seconds")
