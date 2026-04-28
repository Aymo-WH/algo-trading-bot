import pandas as pd
import numpy as np
import yfinance as yf
from core.trading_gym import TradingEnv
from core.meta_agent import MetaAgent
import glob
import os
import warnings
from core.utils import flatten_multiindex_columns, load_agent

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
MODELS_DIR = "models/"
DATA_DIR = "data/test/"
INITIAL_CAPITAL = 10000.0

# Global cache for S&P 500 benchmark data to prevent redundant downloads
_GSPC_CACHE = None
_GSPC_CACHE_PARTS = []

def _consolidate_gspc_cache():
    """
    Consolidates accumulated S&P 500 data parts into the main cache.
    Using list-based accumulation and a single concat is more efficient than
    repeatedly calling pd.concat in a loop.
    """
    global _GSPC_CACHE, _GSPC_CACHE_PARTS
    if not _GSPC_CACHE_PARTS:
        return _GSPC_CACHE

    if _GSPC_CACHE is not None:
        _GSPC_CACHE = pd.concat([_GSPC_CACHE] + _GSPC_CACHE_PARTS)
    else:
        _GSPC_CACHE = pd.concat(_GSPC_CACHE_PARTS)

    _GSPC_CACHE = _GSPC_CACHE.drop_duplicates().sort_index()
    _GSPC_CACHE_PARTS = []
    return _GSPC_CACHE

def calculate_cagr(start_value, end_value, start_date, end_date):
    """
    Calculates the Compound Annual Growth Rate (CAGR).

    CAGR measures the annualized return rate of an investment assuming profits are reinvested.

    Args:
        start_value (float): Initial portfolio value.
        end_value (float): Final portfolio value.
        start_date (datetime): Timestamp of the first bar.
        end_date (datetime): Timestamp of the last bar.

    Returns:
        float: The calculated CAGR as a percentage. Returns 0 if calculation is invalid.
    """
    days = (end_date - start_date).days
    if days <= 0: return 0.0
    years = days / 365.25
    if start_value <= 0: return 0.0 # Avoid division by zero or log of negative
    if end_value <= 0: return -1.0 # Total loss
    return (end_value / start_value) ** (1 / years) - 1

def evaluate_model_on_stock(model, df, stock_name, is_discrete, start_steps):
    """
    Evaluates a trained model on a specific stock's DataFrame over predefined chronological steps.

    This function initializes a TradingEnv, loads the model, and forces the environment to reset
    to specific indices. It returns a dictionary of performance metrics including Total Return,
    Sharpe Ratio, Max Drawdown, and Win Rate.

    Args:
        model: A loaded stable-baselines3 model (DQN/PPO) or MetaAgent.
        df (pd.DataFrame): The specific stock's historical data.
        stock_name (str): Ticker symbol.
        is_discrete (bool): Flag indicating if the agent expects discrete actions.
        start_steps (list): Fixed chronological starting indices for valid PBO calculation.

    Returns:
        dict: Performance metrics averaged across all evaluated steps.
    """

    # If the model is not the MetaAgent and not XGB, we still need to provide the XGB model
    # to the environment so it can construct the observation if PPO expects it.
    xgb_path = "models/xgb_trading_bot.json"

    # Initialize Environment with specific DF
    env = TradingEnv(df=df, is_discrete=is_discrete, xgb_model_path=xgb_path)

    # Ensure episode length aligns with our non-overlapping windows
    env.episode_length = len(df) // 5

    roi_list = []
    cagr_list = []
    profit_list = []
    trades_list = []
    fees_list = []
    all_daily_returns = []

    # Iterate over the pre-defined start steps
    for start_step in start_steps:
        obs, _ = env.reset(options={'start_step': start_step})

        episode_profit = 0.0
        episode_trades = 0
        episode_fees = 0.0

        # Track portfolio value for daily returns
        prev_portfolio_value = INITIAL_CAPITAL

        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Check if model is XGBoost directly
            if hasattr(model, 'predict_proba'):
                # xgb_signal is obs[0], we don't need predict here if we just output signal,
                # but if we evaluate XGBoost directly (which live_inference simulates), we output action.
                # The model variable might be XGBoost, which expects features.
                # Actually, our env returns [signal, prob, vol, dd]
                # If we evaluate standalone XGBoost, the action is just the signal
                xgb_signal = obs[0]
                action = np.array([np.sign(xgb_signal)])
            else:
                # Handle MetaAgent or PPO
                obs_for_prediction = obs
                if hasattr(model, 'observation_space') and hasattr(model.observation_space, 'shape') and model.observation_space.shape[-1] < obs.shape[-1]:
                    obs_for_prediction = obs[..., :model.observation_space.shape[-1]]

                action, _states = model.predict(obs_for_prediction, deterministic=True)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Count trades based on fee
            fee = info.get('step_fee', 0.0)
            if fee > 0.01:
                episode_trades += 1
                episode_fees += fee

            # Accumulate reward
            episode_profit += reward

            # Calculate daily return for this step
            current_portfolio_value = INITIAL_CAPITAL + episode_profit
            daily_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
            all_daily_returns.append(daily_return)
            prev_portfolio_value = current_portfolio_value

            obs = next_obs

        # Calculate Episode Metrics
        final_value = INITIAL_CAPITAL + episode_profit
        roi = (episode_profit / INITIAL_CAPITAL) * 100

        # Calculate CAGR for this episode
        # Start date is at start_step
        # End date is the current step (end of episode)
        ep_start_date = df['Date'].iloc[start_step]
        safe_end_step = min(env.current_step, len(df) - 1)
        ep_end_date = df['Date'].iloc[safe_end_step]
        cagr = calculate_cagr(INITIAL_CAPITAL, final_value, ep_start_date, ep_end_date) * 100

        roi_list.append(roi)
        cagr_list.append(cagr)
        profit_list.append(episode_profit)
        trades_list.append(episode_trades)
        fees_list.append(episode_fees)

    # Average Metrics
    avg_roi = np.mean(roi_list)
    avg_cagr = np.mean(cagr_list)
    total_net_profit = np.sum(profit_list)
    total_trades = np.sum(trades_list)
    total_fees = np.sum(fees_list)

    return {
        "Net Profit": total_net_profit,
        "ROI": avg_roi,
        "CAGR": avg_cagr,
        "Trades": total_trades,
        "Fees": total_fees,
        "Start Date": df['Date'].iloc[start_steps[0]], # Representative
        "End Date": df['Date'].iloc[-1],
        "daily_returns": all_daily_returns
    }

def evaluate_model(dqn_model, ppo_model, ticker):
    """
    Orchestrates the evaluation of multiple models (Hold, DQN, PPO, MetaAgent) for a ticker.

    This function compares standalone base models against the combined Meta-Labeling architecture.
    It iterates over 5 non-overlapping fixed windows to generate evaluation metrics, which are
    then printed in a comparative table.

    Args:
        dqn_model (stable_baselines3.DQN): Trained directional agent.
        ppo_model (stable_baselines3.PPO): Trained sizing agent.
        ticker (str): The stock ticker to evaluate.
    """
    # Load specific dataframe
    data_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data for {ticker} not found in {DATA_DIR}")

    df = pd.read_csv(data_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # Evaluate across 5 non-overlapping fixed windows
    step_size = len(df) // 5
    steps = [i * step_size for i in range(5)]

    meta_agent = MetaAgent(dqn_model=dqn_model, ppo_model=ppo_model, step_size=0.10)

    # Run Evaluation
    metrics = evaluate_model_on_stock(meta_agent, df, ticker, False, steps)

    return None, None, metrics["daily_returns"]

def get_benchmark_sp500(start_date, end_date, sp500_df=None):
    """
    Retrieves S&P 500 benchmark performance with optimized caching and slicing.

    Args:
        start_date (datetime): Beginning of the evaluation window.
        end_date (datetime): End of the evaluation window.
        sp500_df (pd.DataFrame, optional): A pre-fetched benchmark DataFrame.
    """
    global _GSPC_CACHE, _GSPC_CACHE_PARTS
    try:
        # If no explicit DataFrame is provided, check if our cache covers it.
        # Consolidate parts first if we are relying on the global cache.
        if sp500_df is None:
            source_df = _consolidate_gspc_cache()
        else:
            source_df = sp500_df

        # Download if we don't have a source or if dates are out of bounds
        is_covered = (
            source_df is not None and
            not source_df.empty and
            source_df.index[0] <= start_date and
            source_df.index[-1] >= end_date
        )

        if not is_covered:
            # Download missing portion (with small buffer)
            new_data = yf.download("^GSPC", start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)
            new_data = flatten_multiindex_columns(new_data)

            # Append to parts for efficient batch consolidation later
            _GSPC_CACHE_PARTS.append(new_data)
            source_df = _consolidate_gspc_cache()

        if source_df is None or source_df.empty:
            return 0.0, 0.0

        # O(log N) slicing using searchsorted for DatetimeIndex
        start_idx = source_df.index.searchsorted(start_date)
        end_idx = source_df.index.searchsorted(end_date)
        sp500 = source_df.iloc[start_idx:end_idx]

        if sp500.empty:
            return 0.0, 0.0

        # Ensure Close exists
        if 'Close' not in sp500.columns:
            return 0.0, 0.0

        start_price = sp500['Close'].iloc[0]
        end_price = sp500['Close'].iloc[-1]

        roi = ((end_price - start_price) / start_price) * 100
        cagr = calculate_cagr(start_price, end_price, start_date, end_date) * 100

        return roi, cagr
    except Exception as e:
        print(f"Error fetching S&P 500: {e}")
        return 0.0, 0.0

def main(active_tickers=None):
    """
    Command-line execution flow for evaluating agents against baseline hold strategies.
    """
    print("Starting Evaluation...")

    # Find Models
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.zip")) + glob.glob(os.path.join(MODELS_DIR, "*.json"))
    if not model_files:
        print("No models found in models/")
        return

    # Find Data
    if active_tickers:
        data_files = [os.path.join(DATA_DIR, f"{ticker}_data.csv") for ticker in active_tickers]
        data_files = [f for f in data_files if os.path.exists(f)]
    else:
        data_files = glob.glob(os.path.join(DATA_DIR, "*_data.csv"))

    if not data_files:
        print("No data found in data/")
        return

    results = []
    sp500_cache = {}

    # Pre-generate start steps for each stock to ensure fair comparison across models
    stock_start_steps = {}
    stock_dfs = {}
    stock_sp500_benchmarks = {}
    stock_bh_benchmarks = {}

    # Store all dates to fetch global S&P 500 data once
    all_start_dates = []
    all_end_dates = []
    unique_date_pairs = set()

    for data_path in data_files:
        stock_name = os.path.basename(data_path).replace("_data.csv", "")
        df = pd.read_csv(data_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        stock_dfs[stock_name] = df

        # Evaluate across 5 non-overlapping fixed windows
        step_size = len(df) // 5
        steps = [i * step_size for i in range(5)]
        stock_start_steps[stock_name] = steps

        # Collect dates for S&P 500 optimization
        for s in steps:
            s_date = df['Date'].iloc[s]
            # End index is the end of the window
            e_idx = min(s + step_size, len(df) - 1)
            e_date = df['Date'].iloc[e_idx]
            all_start_dates.append(s_date)
            all_end_dates.append(e_date)
            unique_date_pairs.add((s_date, e_date))

    # Fetch global S&P 500 data if dates are available
    global_sp500_df = None
    if all_start_dates:
        global_min_date = min(all_start_dates)
        global_max_date = max(all_end_dates)

        global _GSPC_CACHE
        # Add a 7-day buffer to ensure weekends and holidays are covered
        global_min_date_buffered = global_min_date - pd.Timedelta(days=7)
        global_max_date_buffered = global_max_date + pd.Timedelta(days=7)

        # Check if cache covers the required range
        is_covered = (
            _GSPC_CACHE is not None and
            not _GSPC_CACHE.empty and
            _GSPC_CACHE.index[0] <= global_min_date_buffered and
            _GSPC_CACHE.index[-1] >= global_max_date_buffered
        )

        if not is_covered:
            try:
                print(f"Fetching S&P 500 benchmark data from {global_min_date_buffered.date()} to {global_max_date_buffered.date()}...")
                new_data = yf.download("^GSPC", start=global_min_date_buffered, end=global_max_date_buffered, progress=False)
                new_data = flatten_multiindex_columns(new_data)

                # Efficient accumulation
                _GSPC_CACHE_PARTS.append(new_data)
                _consolidate_gspc_cache()

            except Exception as e:
                print(f"Error pre-fetching S&P 500 data: {e}")

        global_sp500_df = _GSPC_CACHE

    # Pre-calculate unique S&P 500 benchmark pairs to avoid redundant slicing
    if global_sp500_df is not None:
        for s_date, e_date in unique_date_pairs:
            if (s_date, e_date) not in sp500_cache:
                sp500_cache[(s_date, e_date)] = get_benchmark_sp500(s_date, e_date, sp500_df=global_sp500_df)

    # Calculate Benchmarks using optimized or cached approach
    for stock_name, steps in stock_start_steps.items():
        df = stock_dfs[stock_name]
        step_size = len(df) // 5
        sp_rois = []
        sp_cagrs = []
        for s in steps:
            s_date = df['Date'].iloc[s]

            # End index is the end of the window
            e_idx = min(s + step_size, len(df) - 1)
            e_date = df['Date'].iloc[e_idx]

            # Use pre-calculated cache
            r, c = sp500_cache.get((s_date, e_date), (0.0, 0.0))
            sp_rois.append(r)
            sp_cagrs.append(c)

        stock_sp500_benchmarks[stock_name] = (np.mean(sp_rois), np.mean(sp_cagrs))

        # Also pre-calculate Buy & Hold for the stock
        bh_rois = []
        for s in steps:
            start_price = df['Close'].iloc[s]
            e_idx = min(s + step_size, len(df) - 1)
            end_price = df['Close'].iloc[e_idx]
            bh_rois.append(((end_price - start_price) / start_price) * 100)

        stock_bh_benchmarks[stock_name] = np.mean(bh_rois)


    # Load Standalone Models
    models_to_eval = {}
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace(".zip", "")
        try:
            model = load_agent(model_path)
            models_to_eval[model_name] = model
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

    # Load and Instantiate MetaAgent if both XGBoost and PPO are present
    xgb_model = next((m for name, m in models_to_eval.items() if "xgb" in name.lower()), None)
    ppo_model = next((m for name, m in models_to_eval.items() if "ppo" in name.lower()), None)

    if xgb_model and ppo_model:
        meta_agent = MetaAgent(xgb_model=xgb_model, ppo_model=ppo_model, step_size=0.10)
        models_to_eval["meta_agent"] = meta_agent

    for model_name, model in models_to_eval.items():
        # print(f"Evaluating Model: {model_name}")

        for stock_name, df in stock_dfs.items():
            start_steps = stock_start_steps[stock_name]

            # All models now use continuous env in phase 2 setup
            is_discrete = False

            # Run Evaluation
            metrics = evaluate_model_on_stock(model, df, stock_name, is_discrete, start_steps)

            # Benchmarks (Buy & Hold) over the fixed non-overlapping windows
            bh_roi = stock_bh_benchmarks[stock_name]

            sp500_roi, sp500_cagr = stock_sp500_benchmarks[stock_name]

            results.append({
                "Agent": model_name,
                "Stock": stock_name,
                "Trades": metrics["Trades"],
                "Fees ($)": round(metrics["Fees"], 2),
                "Net Profit ($)": round(metrics["Net Profit"], 2),
                "ROI (%)": round(metrics["ROI"], 2),
                "CAGR (%)": round(metrics["CAGR"], 2),
                "vs B&H ROI (%)": round(metrics["ROI"] - bh_roi, 2),
                "vs SP500 ROI (%)": round(metrics["ROI"] - sp500_roi, 2)
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No results generated.")
        return

    # Sort by Agent and Stock
    results_df = results_df.sort_values(by=["Agent", "Stock"])

    print("\n" + "="*120)
    print("STRATEGIC EVALUATION MATRIX")
    print("="*120)
    # Reorder columns
    cols = ["Agent", "Stock", "Net Profit ($)", "ROI (%)", "CAGR (%)", "Trades", "Fees ($)", "vs B&H ROI (%)", "vs SP500 ROI (%)"]

    # Print formatted string
    print(results_df[cols].to_string(index=False))
    print("="*120)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_phase1.json')
    args = parser.parse_args()

    import json
    with open(args.config, 'r') as f:
        config = json.load(f)
        active_tickers = config.get("tickers", [])

    if active_tickers:
        main(active_tickers)
    else:
        main()
