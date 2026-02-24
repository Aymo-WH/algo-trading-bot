import gradio as gr
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO, DQN
from trading_gym import TradingEnv
import os
import numpy as np
import glob
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
MODELS_DIR = "models/"
DATA_DIR = "data/"
INITIAL_CAPITAL = 10000.0

# Cache for S&P 500 data
sp500_cache = {}

def load_agent(model_name):
    """Loads the specified agent model."""
    # Ensure model name includes path if not already provided, or assume it's in models/
    # The dropdown provides just the name 'ppo_trading_bot'
    model_path = os.path.join(MODELS_DIR, model_name)
    if "ppo" in model_name.lower():
        return PPO.load(model_path)
    elif "dqn" in model_name.lower():
        return DQN.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def calculate_cagr(start_val, end_val, start_date, end_date):
    """Calculates Compound Annual Growth Rate."""
    days = (end_date - start_date).days
    if days <= 0: return 0.0
    years = days / 365.25
    if start_val <= 0: return 0.0
    if end_val <= 0: return -1.0
    return (end_val / start_val) ** (1 / years) - 1

def get_sp500_benchmark(start_date, end_date):
    """Fetches S&P 500 benchmark data and calculates CAGR."""
    # Use string representation of dates for cache key to avoid hashing issues with different datetime types if any
    date_key = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if date_key in sp500_cache:
        return sp500_cache[date_key]

    try:
        # Download S&P 500 data
        sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
        if sp500.empty:
            return 0.0

        # Handle MultiIndex if present
        if isinstance(sp500.columns, pd.MultiIndex):
             try:
                sp500.columns = sp500.columns.droplevel(1)
             except:
                pass

        if 'Close' not in sp500.columns:
            return 0.0

        start_price = sp500['Close'].iloc[0]
        end_price = sp500['Close'].iloc[-1]

        cagr = calculate_cagr(start_price, end_price, start_date, end_date) * 100
        sp500_cache[date_key] = cagr
        return cagr
    except Exception as e:
        print(f"Error fetching S&P 500: {e}")
        return 0.0

def run_simulation(agent_model_name, tickers, window_size, num_simulations):
    """Runs the simulation loop."""
    if not tickers:
        return pd.DataFrame(columns=['Ticker', 'Agent', 'Window Size', 'Simulations', 'Total Net Profit', 'Total Trades', 'Agent CAGR', 'SP500 CAGR', 'Win Rate'])

    try:
        model = load_agent(agent_model_name)
    except Exception as e:
        return pd.DataFrame([{"Error": f"Failed to load model: {str(e)}"}])

    results = []
    is_discrete = "dqn" in agent_model_name.lower()

    for ticker in tickers:
        file_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_csv(file_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])

            # Initialize environment with specific window size
            env = TradingEnv(df=df, is_discrete=is_discrete, window_size=window_size)

            ticker_net_profits = []
            ticker_trades = 0
            ticker_agent_cagrs = []
            ticker_sp500_cagrs = []

            wins = 0

            for _ in range(num_simulations):
                obs, _ = env.reset()

                # Determine dates based on environment state
                start_idx = env.start_step
                end_idx = start_idx + window_size

                # Ensure we have dates
                if start_idx < len(env.df) and end_idx <= len(env.df):
                    start_date = env.df['Date'].iloc[start_idx]
                    end_date = env.df['Date'].iloc[min(end_idx - 1, len(env.df)-1)]
                else:
                    # Fallback if indices are weird (should not happen with correct logic)
                    start_date = env.df['Date'].iloc[0]
                    end_date = env.df['Date'].iloc[-1]

                done = False
                truncated = False
                episode_profit = 0.0
                episode_trades = 0

                while not (done or truncated):
                    action, _ = model.predict(obs, deterministic=True)

                    # Count trades
                    if is_discrete:
                         if int(action) != 1: # 1 is Hold
                             episode_trades += 1
                    else:
                         if abs(float(action[0])) > 0.01: # Threshold for hold
                             episode_trades += 1

                    obs, reward, done, truncated, info = env.step(action)
                    episode_profit += reward

                ticker_net_profits.append(episode_profit)
                ticker_trades += episode_trades

                if episode_profit > 0:
                    wins += 1

                # Agent CAGR for this episode
                final_val = INITIAL_CAPITAL + episode_profit
                agent_cagr = calculate_cagr(INITIAL_CAPITAL, final_val, start_date, end_date) * 100
                ticker_agent_cagrs.append(agent_cagr)

                # SP500 CAGR
                sp500_cagr = get_sp500_benchmark(start_date, end_date)
                ticker_sp500_cagrs.append(sp500_cagr)

            # Aggregate results for this ticker
            total_net_profit = sum(ticker_net_profits)
            avg_agent_cagr = np.mean(ticker_agent_cagrs) if ticker_agent_cagrs else 0.0
            avg_sp500_cagr = np.mean(ticker_sp500_cagrs) if ticker_sp500_cagrs else 0.0
            win_rate = (wins / num_simulations) * 100

            results.append({
                "Ticker": ticker,
                "Agent": agent_model_name,
                "Window Size": window_size,
                "Simulations": num_simulations,
                "Total Net Profit ($)": round(total_net_profit, 2),
                "Total Trades": ticker_trades,
                "Agent CAGR (%)": round(avg_agent_cagr, 2),
                "SP500 CAGR (%)": round(avg_sp500_cagr, 2),
                "Win Rate (%)": round(win_rate, 2)
            })

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    return pd.DataFrame(results)

# Scan for tickers
csv_files = glob.glob(os.path.join(DATA_DIR, "*_data.csv"))
tickers_available = [os.path.basename(f).replace("_data.csv", "") for f in csv_files]
tickers_available.sort()

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Algorithmic Trading Dashboard")
    gr.Markdown("Select an agent model, tickers, and simulation parameters to compare performance against the S&P 500.")

    with gr.Row():
        with gr.Column():
            agent_dropdown = gr.Dropdown(
                choices=['ppo_trading_bot', 'dqn_trading_bot'],
                label="Select Agent Model",
                value='ppo_trading_bot'
            )
            ticker_checkbox = gr.CheckboxGroup(
                choices=tickers_available,
                label="Select Tickers",
                value=tickers_available[:1] if tickers_available else []
            )
            window_slider = gr.Slider(
                minimum=30,
                maximum=365,
                step=1,
                label="Window Size (Days)",
                value=90
            )
            sim_slider = gr.Slider(
                minimum=1,
                maximum=50,
                step=1,
                label="Number of Random Simulations",
                value=5
            )
            run_btn = gr.Button("Run Simulation", variant="primary")

    with gr.Row():
        output_df = gr.DataFrame(label="Simulation Results")

    run_btn.click(
        fn=run_simulation,
        inputs=[agent_dropdown, ticker_checkbox, window_slider, sim_slider],
        outputs=output_df
    )

if __name__ == "__main__":
    demo.launch(share=True)
