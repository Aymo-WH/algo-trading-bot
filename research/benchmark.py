import time
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.core.optimize_barriers as ob

def evaluate_barriers_optimized(paths: np.ndarray, sigma: float, pt_grid: np.ndarray, sl_grid: np.ndarray):
    num_paths, length = paths.shape
    best_pt = None
    best_sl = None
    max_sharpe = -np.inf

    P = len(pt_grid)
    S = len(sl_grid)

    pt_levels = pt_grid * sigma
    first_pt_hits = np.empty((P, num_paths), dtype=int)

    # Pre-allocate the padded boolean array ONCE
    # shape (num_paths, length + 1)
    # last column is always True
    hit_padded = np.empty((num_paths, length + 1), dtype=bool)
    hit_padded[:, -1] = True

    for i, pt_level in enumerate(pt_levels):
        # Write directly into the view of the first `length` columns
        hit_padded[:, :-1] = paths >= pt_level
        first_pt_hits[i] = np.argmax(hit_padded, axis=1)

    sl_levels = -sl_grid * sigma
    first_sl_hits = np.empty((S, num_paths), dtype=int)

    for j, sl_level in enumerate(sl_levels):
        hit_padded[:, :-1] = paths <= sl_level
        first_sl_hits[j] = np.argmax(hit_padded, axis=1)

    row_indices = np.arange(num_paths)

    for i, pt in enumerate(pt_grid):
        pt_level = pt_levels[i]
        first_pt_hit = first_pt_hits[i]

        for j, sl in enumerate(sl_grid):
            sl_level = sl_levels[j]
            first_sl_hit = first_sl_hits[j]

            first_hit_idx = np.minimum(first_pt_hit, first_sl_hit)
            exit_idx = np.minimum(first_hit_idx, length - 1)
            exit_pnls = paths[row_indices, exit_idx]

            hit_mask = first_hit_idx < length
            if hit_mask.any():
                hit_pt_at_exit = hit_mask & (exit_pnls >= pt_level)
                hit_sl_at_exit = hit_mask & (exit_pnls <= sl_level)
                exit_pnls = np.where(
                    hit_pt_at_exit, pt_level,
                    np.where(
                        hit_sl_at_exit, sl_level, exit_pnls
                    )
                )

            std_pnl = np.std(exit_pnls)
            if std_pnl > 0:
                mean_pnl = np.mean(exit_pnls)
                sharpe = mean_pnl / std_pnl
            else:
                sharpe = 0.0

            if sharpe > max_sharpe:
                max_sharpe = sharpe
                best_pt = pt
                best_sl = sl

    return best_pt, best_sl, max_sharpe

if __name__ == "__main__":
    # create synthetic paths
    np.random.seed(42)
    paths = np.random.normal(0, 1, size=(100000, 15))
    sigma = 1.0
    pt_grid = np.arange(0.5, 3.25, 0.25)
    sl_grid = np.arange(0.5, 3.25, 0.25)

    start_time = time.time()
    for _ in range(50):
        evaluate_barriers_optimized(paths, sigma, pt_grid, sl_grid)
    end_time = time.time()
    print(f"Optimized Time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    for _ in range(50):
        ob.evaluate_barriers(paths, sigma, pt_grid, sl_grid)
    end_time = time.time()
    print(f"Original Time: {end_time - start_time:.4f} seconds")