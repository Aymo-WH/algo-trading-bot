import numpy as np
import pandas as pd
import yfinance as yf

def estimate_ou_parameters(prices: pd.Series, window: int = 15):
    """
    Estimates the Ornstein-Uhlenbeck (O-U) parameters using Ordinary Least Squares (OLS) regression.

    The O-U process mathematically models the mean-reverting behavior of asset prices.
    It calculates the speed of mean reversion (phi) and volatility (sigma) by comparing
    lagged deviations of price from a rolling long-term mean.

    Args:
        prices (pd.Series): Historical price series.
        window (int): The rolling window size to define the long-term mean. Defaults to 15.

    Returns:
        tuple: (phi, sigma) representing the speed of mean reversion and volatility.
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
    Generates synthetic price paths using the calibrated Ornstein-Uhlenbeck (O-U) process.

    Using the parameters phi (mean reversion) and sigma (volatility), this function
    simulates thousands of possible future price trajectories. These paths are used
    to evaluate Marcos López de Prado's Optimal Trading Rules (OTR) without overfitting
    to the single realized historical path.

    Args:
        phi (float): Speed of mean reversion from O-U estimation.
        sigma (float): Volatility of the residuals from O-U estimation.
        num_paths (int): Number of synthetic paths to simulate. Defaults to 100000.
        length (int): The number of steps (bars) in each path. Defaults to 15.

    Returns:
        np.ndarray: A 2D array of shape (num_paths, length) containing simulated prices.
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


def get_rolling_barriers(price_series: pd.Series, window: int = 60, step: int = 20) -> pd.DataFrame:
    """
    Computes rolling optimal profit-taking (PT) and stop-loss (SL) barriers.

    This implements Marcos López de Prado's dynamic barrier optimization. By rolling a
    lookback window over the price series, it repeatedly estimates the O-U parameters
    and determines the point-in-time Optimal Trading Rules (PT and SL multipliers) that
    maximize the expected Sharpe Ratio over synthetic paths.

    Args:
        price_series (pd.Series): The historical price data.
        window (int): Lookback window size for estimating the O-U process. Defaults to 60.
        step (int): The frequency (in bars) at which the barriers are re-optimized. Defaults to 20.

    Returns:
        pd.DataFrame: A DataFrame containing 'Optimal_PT' and 'Optimal_SL' multipliers for each step.
    """
    # Create an empty DataFrame to hold the results
    result_df = pd.DataFrame(index=price_series.index, columns=['Optimal_PT', 'Optimal_SL'], dtype=float)

    pt_grid = np.arange(0.5, 3.25, 0.25)
    sl_grid = np.arange(0.5, 3.25, 0.25)

    n_samples = len(price_series)

    for start_idx in range(0, n_samples, step):
        # We need to look back `window` days from the current step index.
        # But wait, the prompt says "Every step days ... look back at the last window days".
        # So at index `i`, we look at `price_series.iloc[i-window:i]`.
        end_idx = start_idx + step
        if start_idx < window:
            # We can't compute for the first `window` days, so we skip and let it be NaN
            continue

        # The slice of data we use to optimize the barriers for the NEXT `step` days
        # is the LAST `window` days. So from `start_idx - window` to `start_idx`.
        slice_prices = price_series.iloc[start_idx - window : start_idx]

        # If the slice has enough valid data
        if len(slice_prices.dropna()) >= 15:
            try:
                phi, sigma = estimate_ou_parameters(slice_prices, window=15)
                # Ensure valid parameters before generating paths
                if np.isnan(phi) or np.isnan(sigma) or sigma <= 0:
                    continue

                paths = generate_synthetic_paths(phi, sigma, num_paths=10000, length=15)
                best_pt, best_sl, _ = evaluate_barriers(paths, sigma, pt_grid, sl_grid)

                # Assign to the upcoming step chunk
                # Note: We assign it to `start_idx` up to `end_idx`
                assign_end = min(end_idx, n_samples)
                result_df.iloc[start_idx:assign_end, result_df.columns.get_loc('Optimal_PT')] = best_pt
                result_df.iloc[start_idx:assign_end, result_df.columns.get_loc('Optimal_SL')] = best_sl
            except Exception as e:
                # If estimation fails (e.g. division by zero in covariance), just skip this window
                pass

    # Forward fill missing values
    result_df = result_df.ffill()

    return result_df


def evaluate_barriers(paths: np.ndarray, sigma: float, pt_grid: np.ndarray, sl_grid: np.ndarray):
    """
    Evaluates a grid of PT and SL multipliers against synthetic O-U paths to find the optimal barriers.

    According to Marcos López de Prado's Optimal Trading Rules, the optimal barriers are those
    that maximize the Sharpe Ratio of the resulting PnL distribution when applied to the
    simulated synthetic paths.

    Args:
        paths (np.ndarray): 2D array of synthetic price paths.
        sigma (float): Estimated volatility used to scale the multipliers.
        pt_grid (np.ndarray): Grid of candidate profit-taking multipliers.
        sl_grid (np.ndarray): Grid of candidate stop-loss multipliers.

    Returns:
        tuple: (best_pt, best_sl, max_sharpe) the optimal multipliers and their expected Sharpe Ratio.
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
