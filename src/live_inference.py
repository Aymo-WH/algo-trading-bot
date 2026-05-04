import argparse
import json
import os
import sys
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
import joblib
import ccxt
from dotenv import load_dotenv

load_dotenv() 

# Silence the harmless scikit-learn version warnings for a clean terminal
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn")

# Tell Python to look in the current folder for our local files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_factory import calculate_microstructural_features

def run_live_inference(config_path):
    print("[SYSTEM] Booting Live Inference Engine...")

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

    # Load Brain
    xgb_path = "models/xgb_trading_bot.json"
    
    if not os.path.exists(xgb_path):
        print(f"[ERROR] XGBoost model not found at {xgb_path}.")
        return

    print("[SYSTEM] Loading XGBoost Brain...")
    model = xgb.XGBClassifier()
    model.load_model(xgb_path)

    # Boot Exchange Connection
    print("[SYSTEM] Connecting to Binance Testnet...")
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET'),
        'enableRateLimit': True,
    })
    exchange.set_sandbox_mode(True) # Force Testnet

    print("\n--- LIVE MARKET EXECUTION STREAM ---")
    
    for ticker in tickers:
        print(f"Fetching latest data for {ticker}...")
        
        try:
            df = yf.download(ticker, period="60d", interval='1h', progress=False)
            if df.empty:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            if 'Close' in df.columns and isinstance(df['Close'], pd.DataFrame):
                df['Close'] = df['Close'].squeeze()
                
        except Exception as e:
            print(f"[ERROR] API failed for {ticker}: {e}")
            continue

        try:
            df = calculate_microstructural_features(df)
        except Exception as e:
            print(f"[ERROR] Math engine failed for {ticker}: {e}")
            continue

        df = df.dropna()
        if df.empty:
             print(f"[WARNING] Not enough history for {ticker}.")
             continue
            
        latest_data = df.iloc[-1:]
        
        # Isolate the quantitative features (strip OHLCV)
        raw_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        feature_cols = [col for col in latest_data.columns if col not in raw_cols]
        
        if len(feature_cols) == 0:
            print(f"[WARNING] No custom features found for {ticker}.")
            continue
            
        raw_features = latest_data[feature_cols].values

        # --- THE MATRIX FIX (Using your new matrices/ folder) ---
        clean_ticker = os.path.basename(ticker)
        scaler_path = f"models/matrices/scaler_{clean_ticker}.pkl"
        pca_path = f"models/matrices/pca_{clean_ticker}.pkl"

        if not os.path.exists(scaler_path) or not os.path.exists(pca_path):
            print(f"[WARNING] Missing matrices in models/matrices/. Skipping {ticker}.")
            continue
            
        # Load the specific matrices for this specific ticker
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)

        # Scale, then Compress (PCA to 4 components)
        scaled_features = scaler.transform(raw_features)
        pca_features = pca.transform(scaled_features)

        # Predict using the 4 PCA components
        prediction = model.predict(pca_features)[0]
        probabilities = model.predict_proba(pca_features)[0]
        confidence = max(probabilities)
        # -----------------------

        # Execution Logic
        amount = 0.01 
        
        if prediction == 2.0:
            action = "LONG"
        elif prediction == 0.0:
            action = "SHORT"
        else:
            action = "HOLD"

        print(f"[MARKET] Target: {ticker} | Action: {action} | Conviction: {confidence*100:.1f}%")
        
        # The Live Trigger
        if os.getenv("LIVE_TRADING") == "TRUE":
            
            # Since Binance doesn't trade traditional stocks (XLF, TQQQ), 
            # we proxy the testnet execution to BTC/USDT just to prove the API works.
            testnet_symbol = "BTC/USDT" 
            
            try:
                if action == "LONG":
                    print(f">> PROXY LIVE EXECUTION: BUY {amount} {testnet_symbol} (Signal from {ticker})")
                    order = exchange.create_market_buy_order(testnet_symbol, amount)
                    print(f">> SUCCESS: Order filled! Receipt ID: {order['id']}")
                    
                elif action == "SHORT":
                    print(f">> PROXY LIVE EXECUTION: SELL {amount} {testnet_symbol} (Signal from {ticker})")
                    order = exchange.create_market_sell_order(testnet_symbol, amount)
                    print(f">> SUCCESS: Order filled! Receipt ID: {order['id']}")
            except Exception as e:
                print(f">> [EXCHANGE ERROR]: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Inference Engine")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    input_path = args.config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, input_path) if not os.path.isabs(input_path) else input_path
    
    run_live_inference(os.path.realpath(config_path))