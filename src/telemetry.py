import time
import glob
import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, log_loss
from core.trading_gym import TradingEnv
from core.meta_agent import MetaAgent
from core.utils import load_agent

warnings.filterwarnings("ignore")

MODELS_DIR = "models/"
DATA_DIR = "data/test/"
INITIAL_CAPITAL = 10000.0

def run_telemetry(ticker):
    print(f"\nEvaluating Telemetry for {ticker}...")

    data_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    if not os.path.exists(data_path):
        print(f"Data for {ticker} not found.")
        return

    df = pd.read_csv(data_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # Load Models
    xgb_model = None
    ppo_model = None

    xgb_path = os.path.join(MODELS_DIR, "xgb_trading_bot.json")
    ppo_path = os.path.join(MODELS_DIR, "ppo_meta_labeler.zip")

    if os.path.exists(xgb_path):
        xgb_model = load_agent(xgb_path)
    if os.path.exists(ppo_path):
        ppo_model = load_agent(ppo_path)

    if xgb_model is None or ppo_model is None:
        print("Both XGBoost and PPO models must be present in models/ directory.")
        return

    meta_agent = MetaAgent(xgb_model=xgb_model, ppo_model=ppo_model, step_size=0.10)

    # -------------------------------------------------------------------------
    # RUN ISOLATED XGBoost (Flat 1-unit bet size)
    # -------------------------------------------------------------------------
    # We use a non-discrete env but without PPO, just outputting XGB signal
    env_xgb = TradingEnv(df=df, is_discrete=False, xgb_model_path=xgb_path)
    obs, _ = env_xgb.reset(options={'start_step': 0})

    xgb_y_true = []
    xgb_y_pred = []

    xgb_inference_times = []
    xgb_reconstruction_times = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        t0 = time.perf_counter()

        # XGBoost signal is the first element of observation
        xgb_signal = obs[0]
        action = np.array([np.sign(xgb_signal)])

        t1 = time.perf_counter()
        xgb_inference_times.append((t1 - t0) * 1000) # ms

        # Predict 1 (Long) if direction > 0, else 0 (Hold)
        predicted_class = 1 if xgb_signal > 0 else 0

        # Get next price to determine actual label
        current_step = env_xgb.current_step
        decision_idx = current_step + env_xgb.window_size - 1
        current_price = env_xgb._prices[decision_idx]

        t2 = time.perf_counter()
        next_obs, reward, terminated, truncated, info = env_xgb.step(action)
        t3 = time.perf_counter()
        xgb_reconstruction_times.append((t3 - t2) * 1000) # ms

        next_decision_idx = env_xgb.current_step + env_xgb.window_size - 1
        if next_decision_idx < len(env_xgb._prices):
            next_price = env_xgb._prices[next_decision_idx]
        else:
            next_price = env_xgb._prices[-1]

        # Ground truth: 1 if price went up, else 0
        actual_class = 1 if next_price > current_price else 0

        xgb_y_pred.append(predicted_class)
        xgb_y_true.append(actual_class)

        obs = next_obs

    # Calculate Isolated XGB Metrics
    xgb_cm = confusion_matrix(xgb_y_true, xgb_y_pred, labels=[0, 1])
    xgb_tn, xgb_fp, xgb_fn, xgb_tp = xgb_cm.ravel()

    xgb_recall = recall_score(xgb_y_true, xgb_y_pred, zero_division=0)
    xgb_accuracy = accuracy_score(xgb_y_true, xgb_y_pred)
    xgb_precision = precision_score(xgb_y_true, xgb_y_pred, zero_division=0)

    # -------------------------------------------------------------------------
    # RUN COMBINED SYSTEM (XGBoost + PPO Meta-Labeling)
    # -------------------------------------------------------------------------
    env_meta = TradingEnv(df=df, is_discrete=False, xgb_model_path=xgb_path)
    obs, _ = env_meta.reset(options={'start_step': 0})

    meta_y_true = []
    meta_y_pred = []
    meta_y_prob = []

    meta_inference_times = []
    meta_reconstruction_times = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        t0 = time.perf_counter()

        xgb_signal = obs[0]
        xgb_prob_val = obs[1]

        if xgb_signal == 0:
            final_action = np.array([0.0])
            prob_long = 0.0 # No conviction to go long
        else:
            ppo_act, _ = meta_agent.ppo.predict(obs, deterministic=True)
            raw_size = np.clip(ppo_act[0], 0.0, 1.0)
            m_discrete = np.round(raw_size / meta_agent.step_size) * meta_agent.step_size
            # In Phase 2, PPO output is just bet size and action space is continuous [0.0, 1.0].
            # TradingEnv will treat action > 0 as Long and action <= 0 as Sell based on `act > 0`.
            # To pass a valid action to the env, we just pass the positive size if it's a long signal,
            # or 0.0 if it's a short signal (which effectively sells existing position).
            # The MetaAgent predict logic does np.sign(direction) * m_discrete, returning negative actions for shorts.
            # We must map this for the Gym Box space. But wait, TradingEnv action_space is Box(0.0, 1.0).
            # Therefore we shouldn't pass negative actions!
            # If xgb_signal < 0, we just want to sell. A 0.0 action is sell.
            if xgb_signal < 0:
                final_action = np.array([0.0])
            else:
                final_action = np.array([m_discrete])

            # Probability that the current signal is a winning long
            if xgb_signal > 0:
                prob_long = xgb_prob_val
            else:
                prob_long = 0.0 # Short signals are ignored in long-only

        t1 = time.perf_counter()
        meta_inference_times.append((t1 - t0) * 1000)

        predicted_class = 1 if final_action[0] > 0 else 0

        current_step = env_meta.current_step
        decision_idx = current_step + env_meta.window_size - 1
        current_price = env_meta._prices[decision_idx]

        t2 = time.perf_counter()
        next_obs, reward, terminated, truncated, info = env_meta.step(final_action)
        t3 = time.perf_counter()
        meta_reconstruction_times.append((t3 - t2) * 1000)

        next_decision_idx = env_meta.current_step + env_meta.window_size - 1
        if next_decision_idx < len(env_meta._prices):
            next_price = env_meta._prices[next_decision_idx]
        else:
            next_price = env_meta._prices[-1]

        actual_class = 1 if next_price > current_price else 0

        meta_y_pred.append(predicted_class)
        meta_y_true.append(actual_class)
        meta_y_prob.append(prob_long)

        obs = next_obs

    # Calculate Combined System Metrics
    meta_cm = confusion_matrix(meta_y_true, meta_y_pred, labels=[0, 1])
    meta_tn, meta_fp, meta_fn, meta_tp = meta_cm.ravel()

    meta_precision = precision_score(meta_y_true, meta_y_pred, zero_division=0)
    meta_f1 = f1_score(meta_y_true, meta_y_pred, zero_division=0)

    try:
        meta_log_loss = log_loss(meta_y_true, meta_y_prob, labels=[0, 1])
    except ValueError:
        meta_log_loss = np.nan

    delta_precision = meta_precision - xgb_precision

    avg_inference = np.mean(meta_inference_times)
    avg_reconstruction = np.mean(meta_reconstruction_times)

    print("==================================================")
    print("1. DECOUPLED AGENT TELEMETRY REPORT")
    print("==================================================")
    print("Baseline XGBoost (Isolated)")
    print(f"Confusion Matrix: TP={xgb_tp}, TN={xgb_tn}, FP={xgb_fp}, FN={xgb_fn}")
    print(f"Recall:   {xgb_recall:.4f}")
    print(f"Accuracy: {xgb_accuracy:.4f}")
    print(f"Precision: {xgb_precision:.4f}")
    print("-" * 50)
    print("Combined System (XGBoost + PPO)")
    print(f"Confusion Matrix: TP={meta_tp}, TN={meta_tn}, FP={meta_fp}, FN={meta_fn}")
    print(f"Precision: {meta_precision:.4f}")
    print(f"F1-Score:  {meta_f1:.4f}")
    print(f"Negative Log-Loss: {-meta_log_loss:.4f} (Raw: {meta_log_loss:.4f})")
    print("-" * 50)
    print(f"The Delta (Precision Difference): {delta_precision:+.4f}")
    print("==================================================")
    print("2. EXECUTION LATENCY REPORT")
    print("==================================================")
    print(f"Average Inference Latency:      {avg_inference:.2f} ms")
    print(f"Average Reconstruction Latency: {avg_reconstruction:.2f} ms")
    print("==================================================\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_phase1.json')
    args = parser.parse_args()

    import json
    with open(args.config, 'r') as f:
        config = json.load(f)
        active_tickers = config.get("tickers", [])

    for ticker in active_tickers:
        run_telemetry(ticker)
