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

    # Need at least 2 points to calculate covariance
    if len(X) < 2:
        return np.nan, np.nan

    # Calculate the Speed of Mean Reversion: phi = cov(Y, X) / cov(X, X)
    cov_y_x = np.cov(Y, X)[0, 1]
    cov_x_x = np.cov(X)

    # Avoid division by zero if variance of X is zero
    if cov_x_x == 0.0 or np.isnan(cov_x_x):
        return np.nan, np.nan

    phi = cov_y_x / cov_x_x

    # Calculate the Residuals: xi = Y - Z - (phi * X)
    xi = Y - Z - (phi * X)

    # Calculate Volatility: sigma = sqrt(cov(xi, xi))
    cov_xi_xi = np.cov(xi)
    sigma = np.sqrt(cov_xi_xi)

    return float(phi), float(sigma)


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

    P = len(pt_grid)
    S = len(sl_grid)

    # Pre-calculate boolean hit masks and hit indices for each grid point
    # We do this individually to avoid large memory allocations and caching issues
    pt_levels = pt_grid * sigma
    first_pt_hits = np.empty((P, num_paths), dtype=int)

    # Pre-allocate padded array to avoid reallocation in loops
    hit_padded = np.ones((num_paths, length + 1), dtype=bool)

    for i, pt_level in enumerate(pt_levels):
        hit_padded[:, :-1] = paths >= pt_level
        # The padded True at the end handles paths that never hit
        first_pt_hits[i] = np.argmax(hit_padded, axis=1)

    sl_levels = -sl_grid * sigma
    first_sl_hits = np.empty((S, num_paths), dtype=int)
    for j, sl_level in enumerate(sl_levels):
        hit_padded[:, :-1] = paths <= sl_level
        first_sl_hits[j] = np.argmax(hit_padded, axis=1)

    row_indices = np.arange(num_paths)

    # A path terminates immediately if it hits PT * sigma or -SL * sigma

    for i, pt in enumerate(pt_grid):
        pt_level = pt_levels[i]
        first_pt_hit = first_pt_hits[i]

        for j, sl in enumerate(sl_grid):
            sl_level = sl_levels[j]
            first_sl_hit = first_sl_hits[j]

            # Find the overall first hit
            first_hit_idx = np.minimum(first_pt_hit, first_sl_hit)

            # Cap exit indices at length - 1 (for paths that didn't hit)
            exit_idx = np.minimum(first_hit_idx, length - 1)

            # Extract PnLs at the exit step
            exit_pnls = paths[row_indices, exit_idx]

            # Adjust PnL based on barrier hit
            hit_mask = first_hit_idx < length

            # Only perform np.where if we actually have hits
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

            # Avoid division by zero
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
