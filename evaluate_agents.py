import pandas as pd
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO, DQN
from trading_gym import TradingEnv
import glob
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
MODELS_DIR = "models/"
DATA_DIR = "data/"
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

def evaluate_model_on_stock(model, df, stock_name):
    # Ensure Date is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # Initialize Environment with specific DF
    env = TradingEnv(df=df)
    obs, _ = env.reset()

    portfolio_value = INITIAL_CAPITAL
    current_position = 1 # 0=Short, 1=Flat, 2=Long. Start Flat.

    # Tracking
    trades = 0
    fees_paid = 0.0

    terminated = False
    truncated = False

    portfolio_values = []

    # Iterate through the environment
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        action = int(action) # Ensure int

        # Capture state before step
        prev_step = env.current_step

        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Determine Trade Logic (Stateful)
        # Check for position change
        if action != current_position:
            # We are changing position.
            # Calculate fees for exiting old position (if any) and entering new (if any).

            # Exit old position fee?
            if current_position != 1: # Was Long or Short
                # Closing position incurs fee
                fee = portfolio_value * TRANSACTION_FEE
                portfolio_value -= fee
                fees_paid += fee
                trades += 1 # Count as a trade (exit)

            # Enter new position fee?
            if action != 1: # Going Long or Short
                # Opening position incurs fee
                fee = portfolio_value * TRANSACTION_FEE
                portfolio_value -= fee
                fees_paid += fee
                trades += 1 # Count as a trade (entry)

            # Update Position
            current_position = action

        # Calculate Daily Return based on Position held *during* the step
        # env.current_step is now incremented
        curr_step = env.current_step

        if curr_step < len(env._prices):
             price_t = env._prices[prev_step]
             price_t1 = env._prices[curr_step]

             # Calculate Daily Return
             daily_return = 0.0
             gross_return = (price_t1 - price_t) / price_t

             if current_position == 2: # Long
                 daily_return = gross_return
             elif current_position == 0: # Short
                 daily_return = -gross_return
             elif current_position == 1: # Flat
                 daily_return = 0.0

             # Update Portfolio
             portfolio_value *= (1 + daily_return)
             portfolio_values.append(portfolio_value)

        obs = next_obs

    # Calculate Metrics
    net_profit = portfolio_value - INITIAL_CAPITAL
    roi = (net_profit / INITIAL_CAPITAL) * 100

    start_date = df['Date'].iloc[0]
    end_date = df['Date'].iloc[-1]

    cagr = calculate_cagr(INITIAL_CAPITAL, portfolio_value, start_date, end_date) * 100

    return {
        "Net Profit": net_profit,
        "ROI": roi,
        "CAGR": cagr,
        "Trades": trades,
        "Fees": fees_paid,
        "Final Value": portfolio_value,
        "Start Date": start_date,
        "End Date": end_date
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

    for model_path in model_files:
        model_name = os.path.basename(model_path).replace(".zip", "")
        # print(f"Evaluating Model: {model_name}")

        try:
            model = load_agent(model_path)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        for data_path in data_files:
            stock_name = os.path.basename(data_path).replace("_data.csv", "")
            # print(f"  Testing on {stock_name}...")

            # Load Data
            df = pd.read_csv(data_path)

            # Run Evaluation
            metrics = evaluate_model_on_stock(model, df, stock_name)

            # Benchmarks
            bh_roi, bh_cagr = get_buy_and_hold(df)

            start_date = metrics["Start Date"]
            end_date = metrics["End Date"]

            # S&P 500 Benchmark
            # Key by date tuple (use string format to be hashable if needed, or tuple)
            date_key = (start_date, end_date)
            if date_key not in sp500_cache:
                sp500_cache[date_key] = get_benchmark_sp500(start_date, end_date)
            sp500_roi, sp500_cagr = sp500_cache[date_key]

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
