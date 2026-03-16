import numpy as np
import pandas as pd
import yfinance as yf

def estimate_ou_parameters(prices: pd.Series, window: int = 15):
    """
    Estimates the Ornstein-Uhlenbeck parameters using OLS regression.
    """
    # Define Y as the current prices (P_t)
    Y = prices.values

    # Define Z as a rolling moving average representing the long-term mean
    Z = prices.rolling(window=window).mean().values

    # Define X as the lagged deviations (P_{t-1} - Z_{t-1})
    P_prev = prices.shift(1).values
    Z_prev = pd.Series(Z).shift(1).values
    X = P_prev - Z_prev

    # Drop NaNs created by rolling and shift
    valid = ~np.isnan(Y) & ~np.isnan(Z) & ~np.isnan(X)
    Y = Y[valid]
    Z = Z[valid]
    X = X[valid]

    # Calculate the Speed of Mean Reversion: phi = cov(Y, X) / cov(X, X)
    cov_y_x = np.cov(Y, X)[0, 1]
    cov_x_x = np.cov(X)
    phi = cov_y_x / cov_x_x

    # Calculate the Residuals: xi = Y - Z - (phi * X)
    xi = Y - Z - (phi * X)

    # Calculate Volatility: sigma = sqrt(cov(xi, xi))
    cov_xi_xi = np.cov(xi)
    sigma = np.sqrt(cov_xi_xi)

    return phi, sigma


def generate_synthetic_paths(phi: float, sigma: float, num_paths: int = 100000, length: int = 15):
    """
    Generates synthetic price paths using the Ornstein-Uhlenbeck process.
    """
    # Initialize paths array: num_paths rows, length columns
    paths = np.zeros((num_paths, length))

    # Initialize all paths with an entry price of 0 (which is already 0 from np.zeros)

    # Iteratively generate the next step
    for t in range(1, length):
        noise = np.random.normal(0, 1, size=num_paths)
        # p_t = (1 - phi) * 0 + phi * p_{t-1} + sigma * N(0, 1)
        paths[:, t] = phi * paths[:, t-1] + sigma * noise

    return paths


def evaluate_barriers(paths: np.ndarray, sigma: float, pt_grid: np.ndarray, sl_grid: np.ndarray):
    """
    Evaluates a grid of PT and SL multipliers on the synthetic paths.
    """
    num_paths, length = paths.shape
    best_pt = None
    best_sl = None
    max_sharpe = -np.inf

    # Pre-compute maxima and minima to speed up early exit checking if possible,
    # but since paths are 15 steps long, vectorization over paths is better.

    # A path terminates immediately if it hits PT * sigma or -SL * sigma

    for pt in pt_grid:
        for sl in sl_grid:
            pt_level = pt * sigma
            sl_level = -sl * sigma

            # Find the first step where the path hits the PT or SL barrier
            # Create boolean masks for hits
            hit_pt = paths >= pt_level
            hit_sl = paths <= sl_level
            hit_any = hit_pt | hit_sl

            # Find the index of the first hit for each path
            # argmax returns the first index of True. If all are False, it returns 0.
            # We can use this, but we need to handle paths that never hit.

            # Add a dummy True at the end (column index `length`) to handle no-hit paths
            # We'll use the last step value if no hit occurred
            hit_any_padded = np.hstack([hit_any, np.ones((num_paths, 1), dtype=bool)])
            first_hit_idx = np.argmax(hit_any_padded, axis=1)

            # Now we extract the PnL at the exit step.
            # If a path never hit (first_hit_idx == length), it exits at length - 1.
            exit_idx = np.minimum(first_hit_idx, length - 1)

            # Get the exit PnL
            # We need to extract the value from each row at the corresponding exit_idx
            row_indices = np.arange(num_paths)
            exit_pnls = paths[row_indices, exit_idx]

            # If it hit the PT barrier, we cap the PnL at pt_level.
            # If it hit the SL barrier, we cap the PnL at sl_level.
            # Wait, the prompt says "terminates immediately if it hits PT * sigma (Profit), -SL * sigma (Loss)".
            # In real trading, if it crosses the barrier, you exit AT the barrier price (or worse/better depending on slippage, but usually we just use the barrier value).
            # The prompt says: "A path terminates immediately if it hits PT * sigma (Profit), -SL * sigma (Loss), or reaches the end of the 15 steps."
            # We'll assign the barrier values to the ones that hit it.

            # Did it hit? first_hit_idx < length
            hit_mask = first_hit_idx < length

            # Which one did it hit first?
            # It could hit both in the same step, but we check PT first or SL first.
            # Usually, we just use the value at that step. But to be exact to the prompt:
            # Let's just clip the exit PnLs to the barriers for those that hit.
            # Wait, if `paths[r, c] >= pt_level` is the first hit, the exact exit PnL might just be `pt_level` (limit order)
            # or it might be `paths[r, c]` (market order).
            # Let's just use `paths[row_indices, exit_idx]` and clip it to the barriers.
            # Actually, standard is to use the actual barrier levels. Let's use the barrier levels for hits.
            exit_pnls = np.where(
                hit_mask & hit_pt[row_indices, exit_idx],
                pt_level,
                np.where(
                    hit_mask & hit_sl[row_indices, exit_idx],
                    sl_level,
                    exit_pnls
                )
            )

            mean_pnl = np.mean(exit_pnls)
            std_pnl = np.std(exit_pnls)

            # Avoid division by zero
            if std_pnl > 0:
                sharpe = mean_pnl / std_pnl
            else:
                sharpe = 0.0

            if sharpe > max_sharpe:
                max_sharpe = sharpe
                best_pt = pt
                best_sl = sl

    return best_pt, best_sl, max_sharpe


if __name__ == "__main__":
    print("Fetching AAPL data...")
    # Fetch historical data for AAPL
    aapl_data = yf.download('AAPL', start='2016-01-01', progress=False)

    # Extract 'Close' prices. Handle potential MultiIndex columns from yfinance.
    if isinstance(aapl_data.columns, pd.MultiIndex):
        prices = aapl_data['Close'].iloc[:, 0] if isinstance(aapl_data['Close'], pd.DataFrame) else aapl_data['Close']
    else:
        prices = aapl_data['Close']

    print("Estimating O-U Parameters...")
    phi, sigma = estimate_ou_parameters(prices, window=15)
    print(f"Estimated phi: {phi:.6f}, sigma: {sigma:.6f}")

    print("Generating Synthetic Paths...")
    paths = generate_synthetic_paths(phi, sigma, num_paths=100000, length=15)

    print("Running Grid Search...")
    pt_grid = np.arange(0.5, 3.25, 0.25)
    sl_grid = np.arange(0.5, 3.25, 0.25)

    best_pt, best_sl, max_sharpe = evaluate_barriers(paths, sigma, pt_grid, sl_grid)

    print(f"Optimal PT Multiplier: {best_pt:.2f}")
    print(f"Optimal SL Multiplier: {best_sl:.2f}")
    print(f"Max Expected Sharpe Ratio: {max_sharpe:.6f}")
