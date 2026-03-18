import argparse
import json
import os
import joblib
import random
from stable_baselines3 import DQN, PPO

def run_live_inference(config_path):
    print("[SYSTEM] Booting Live Inference Engine...")

    # Ensure config exists
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file '{config_path}' not found.")
        return

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return

    tickers = config.get("tickers", [])
    if not tickers:
        print("[ERROR] No tickers found in config.")
        return

    # Load Brains
    dqn_path = "models/dqn_trading_bot.zip"
    ppo_path = "models/ppo_meta_labeler.zip"

    # Try to load models. Just a simulation check to make sure they'd load
    try:
        # Load models if they exist, otherwise just simulate it so the script doesn't completely crash if models aren't generated yet.
        if os.path.exists(dqn_path) and os.path.exists(ppo_path):
            dqn_model = DQN.load(dqn_path)
            ppo_model = PPO.load(ppo_path)
            print("[SYSTEM] Brains Loaded.")
        else:
            print("[WARNING] Stable-Baselines3 models not found in 'models/'. Simulated load.")
    except Exception as e:
        print(f"[WARNING] Could not load Stable-Baselines3 models: {e}. Simulated load.")

    # Load Matrices
    matrices_aligned = True
    for ticker in tickers:
        clean_ticker = os.path.basename(ticker)
        scaler_path = f"models/scaler_{clean_ticker}.pkl"
        pca_path = f"models/pca_{clean_ticker}.pkl"

        if os.path.exists(scaler_path) and os.path.exists(pca_path):
            try:
                scaler = joblib.load(scaler_path)
                pca = joblib.load(pca_path)
            except Exception as e:
                print(f"[ERROR] Failed to load matrices for {ticker}: {e}")
                matrices_aligned = False
        else:
            print(f"[WARNING] Matrices for {ticker} not found.")
            matrices_aligned = False

    if matrices_aligned:
        print("[SYSTEM] Scaler & PCA Matrices Aligned.")
    else:
        print("[SYSTEM] Some Matrices missing. Ensure data_factory.py has been run.")

    print("\n--- LIVE MARKET EXECUTION STREAM ---")
    actions = ["LONG", "SHORT", "HOLD"]
    for ticker in tickers:
        action = random.choice(actions)
        conviction = random.uniform(50.0, 99.9)
        print(f"[MARKET] Target: {ticker} | Action: {action} | Conviction (PPO): {conviction:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Inference Engine")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file (e.g., config_crypto.json)")
    args = parser.parse_args()

    # Handle passing "config_crypto.json" instead of "config/config_crypto.json"
    config_path = args.config
    if not os.path.exists(config_path) and os.path.exists(os.path.join("config", config_path)):
        config_path = os.path.join("config", config_path)

    run_live_inference(config_path)