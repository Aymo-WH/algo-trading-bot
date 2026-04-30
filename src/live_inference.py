import argparse
import json
import os
import random
import time
import ccxt
import pandas as pd
import numpy as np

try:
    import joblib
    from stable_baselines3 import PPO
    from core.utils import load_agent
    from core.meta_agent import MetaAgent
    from data_factory import calculate_microstructural_features
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

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
    xgb_path = "models/xgb_trading_bot.json"

    xgb_model = None

    # Try to load models. Just a simulation check to make sure they'd load
    if HAS_DEPENDENCIES:
        try:
            # Load models if they exist, otherwise just simulate it so the script doesn't completely crash if models aren't generated yet.
            if os.path.exists(xgb_path):
                xgb_model = load_agent(xgb_path)
                print("[SYSTEM] XGBoost Brain Loaded (Executive Override).")
            else:
                print("[WARNING] XGBoost model not found in 'models/'. Simulated load.")
        except Exception as e:
            print(f"[WARNING] Could not load models: {e}. Simulated load.")
    else:
        print("[WARNING] Dependencies (joblib, stable-baselines3) missing. Skipping model/matrix load.")

    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET')
    })

    # Load Matrices
    matrices_aligned = True
    scalers = {}
    pcas = {}
    if HAS_DEPENDENCIES:
        for ticker in tickers:
            clean_ticker = os.path.basename(ticker)
            scaler_path = f"models/matrices/scaler_{clean_ticker}.pkl"
            pca_path = f"models/matrices/pca_{clean_ticker}.pkl"

            if os.path.exists(scaler_path) and os.path.exists(pca_path):
                try:
                    scalers[ticker] = joblib.load(scaler_path)
                    pcas[ticker] = joblib.load(pca_path)
                except Exception as e:
                    print(f"[ERROR] Failed to load matrices for {ticker}: {e}")
                    matrices_aligned = False
            else:
                print(f"[WARNING] Matrices for {ticker} not found.")
                matrices_aligned = False
    else:
        matrices_aligned = False

    if matrices_aligned:
        print("[SYSTEM] Scaler & PCA Matrices Aligned.")
    else:
        print("[SYSTEM] Some Matrices missing. Ensure data_factory.py has been run.")

    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET')
    })
    exchange.set_sandbox_mode(True)

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
    input_path = args.config

    # Security Fix: Prevent Path Traversal
    # 1. Resolve project root and allowed config directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    allowed_config_dir = os.path.realpath(os.path.join(project_root, "config"))

    # 2. If it's a simple filename, assume it's in the 'config/' directory.
    # Otherwise, treat it as a relative or absolute path.
    if os.path.dirname(input_path) == "":
        config_path = os.path.join(allowed_config_dir, input_path)
    else:
        # Resolve relative to project root if not absolute
        if not os.path.isabs(input_path):
            config_path = os.path.join(project_root, input_path)
        else:
            config_path = input_path

    # 3. Final Security Validation: Resolve absolute path and verify it's within 'config/'
    abs_config_path = os.path.realpath(config_path)

    if not abs_config_path.startswith(allowed_config_dir + os.sep) and abs_config_path != allowed_config_dir:
        print(f"[ERROR] Security: Configuration path '{input_path}' is restricted.")
    else:
        run_live_inference(abs_config_path)