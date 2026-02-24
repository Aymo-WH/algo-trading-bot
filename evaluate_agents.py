import pandas as pd
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO, DQN
from trading_gym import TradingEnv
import glob
import os
import warnings
import random

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
MODELS_DIR = "models/"
DATA_DIR = "data/test/"
INITIAL_CAPITAL = 10000.0
TRANSACTION_FEE = 0.001 # 0.1%

def load_agent(model_path):
    if "ppo" in model_path.lower():
        return PPO.load(model_path)
    elif "dqn" in model_path.lower():
        return DQN.load(model_path)
    else:
        raise ValueError(f"Unknown model type for {model_path}")

def calculate_cagr(start_value, end_value, start_date, end_date):
    days = (end_date - start_date).days
    if days <= 0: return 0.0
    years = days / 365.25
    if start_value <= 0: return 0.0 # Avoid division by zero or log of negative
    if end_value <= 0: return -1.0 # Total loss
    return (end_value / start_value) ** (1 / years) - 1

def evaluate_model_on_stock(model, df, stock_name, is_discrete, start_steps):
    # Ensure Date is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # Initialize Environment with specific DF
    env = TradingEnv(df=df, is_discrete=is_discrete)

    roi_list = []
    cagr_list = []
    profit_list = []
    trades_list = []
    fees_list = []

    # Iterate over the pre-defined start steps
    for start_step in start_steps:
        obs, _ = env.reset(options={'start_step': start_step})

        episode_profit = 0.0
        episode_trades = 0
        episode_fees = 0.0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Count trades based on fee
            fee = info.get('step_fee', 0.0)
            if fee > 0:
                episode_trades += 1
                episode_fees += fee

            # Accumulate reward
            episode_profit += reward

            obs = next_obs

        # Calculate Episode Metrics
        final_value = INITIAL_CAPITAL + episode_profit
        roi = (episode_profit / INITIAL_CAPITAL) * 100

        # Calculate CAGR for this episode
        # Start date is at start_step
        # End date is last date of DF (as per env termination logic)
        ep_start_date = df['Date'].iloc[start_step]
        ep_end_date = df['Date'].iloc[-1]
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
        "End Date": df['Date'].iloc[-1]
    }

def get_benchmark_sp500(start_date, end_date):
    # Fetch S&P 500 data
    try:
        # Download usually returns index as Date
        sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)

        if sp500.empty:
            return 0.0, 0.0

        if isinstance(sp500.columns, pd.MultiIndex):
            try:
                sp500.columns = sp500.columns.droplevel(1)
            except:
                pass

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

def get_buy_and_hold(df):
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]

    roi = ((end_price - start_price) / start_price) * 100

    start_date = pd.to_datetime(df['Date'].iloc[0])
    end_date = pd.to_datetime(df['Date'].iloc[-1])
    cagr = calculate_cagr(start_price, end_price, start_date, end_date) * 100

    return roi, cagr

def main():
    print("Starting Evaluation...")

    # Find Models
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.zip"))
    if not model_files:
        print("No models found in models/")
        return

    # Find Data
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

    for data_path in data_files:
        stock_name = os.path.basename(data_path).replace("_data.csv", "")
        df = pd.read_csv(data_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        stock_dfs[stock_name] = df

        # Generate 5 random start steps
        # Ensure start_step fits within dataframe minus window_size
        # Env window_size is hardcoded to 10 in trading_gym.py
        window_size = 10
        max_step = len(df) - window_size - 1

        if max_step > 0:
             steps = [random.randint(0, max_step) for _ in range(5)]
        else:
             steps = [0] * 5

        stock_start_steps[stock_name] = steps

        # Calculate S&P 500 Benchmark for these 5 windows
        sp_rois = []
        sp_cagrs = []
        for s in steps:
            s_date = df['Date'].iloc[s]
            e_date = df['Date'].iloc[-1]

            # Check cache
            date_key = (s_date, e_date)
            if date_key not in sp500_cache:
                 sp500_cache[date_key] = get_benchmark_sp500(s_date, e_date)

            r, c = sp500_cache[date_key]
            sp_rois.append(r)
            sp_cagrs.append(c)

        stock_sp500_benchmarks[stock_name] = (np.mean(sp_rois), np.mean(sp_cagrs))


    for model_path in model_files:
        model_name = os.path.basename(model_path).replace(".zip", "")
        # print(f"Evaluating Model: {model_name}")

        try:
            model = load_agent(model_path)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        for stock_name, df in stock_dfs.items():
            start_steps = stock_start_steps[stock_name]
            is_discrete = "dqn" in model_name.lower()

            # Run Evaluation
            metrics = evaluate_model_on_stock(model, df, stock_name, is_discrete, start_steps)

            # Benchmarks (Buy & Hold) - average over same windows?
            # Original code did BH on full DF. But to be fair, we should average BH over the same 5 windows.
            # However, prompt only asked for S&P 500 benchmark using exact same 5 windows.
            # I will assume BH should also be fair.
            bh_rois = []
            for s in start_steps:
                 # BH logic: (End - Start) / Start
                 start_price = df['Close'].iloc[s]
                 end_price = df['Close'].iloc[-1]
                 bh_rois.append(((end_price - start_price) / start_price) * 100)

            bh_roi = np.mean(bh_rois)

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
    print("UNIFIED GLADIATOR LEADERBOARD")
    print("="*120)
    # Reorder columns
    cols = ["Agent", "Stock", "Net Profit ($)", "ROI (%)", "CAGR (%)", "Trades", "Fees ($)", "vs B&H ROI (%)", "vs SP500 ROI (%)"]

    # Print formatted string
    print(results_df[cols].to_string(index=False))
    print("="*120)

if __name__ == "__main__":
    main()
