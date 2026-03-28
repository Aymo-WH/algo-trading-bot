import time
import glob
import os
import json
import warnings
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, log_loss
from core.trading_gym import TradingEnv
from core.meta_agent import MetaAgent

warnings.filterwarnings("ignore")

MODELS_DIR = "models/"
DATA_DIR = "data/test/"
INITIAL_CAPITAL = 10000.0

def load_agent(model_path):
    if "ppo" in model_path.lower():
        return PPO.load(model_path)
    elif "dqn" in model_path.lower():
        return DQN.load(model_path)
    else:
        raise ValueError(f"Unknown model type for {model_path}")

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
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.zip"))
    dqn_model = None
    ppo_model = None

    for mf in model_files:
        if "dqn" in mf.lower():
            dqn_model = load_agent(mf)
        elif "ppo" in mf.lower():
            ppo_model = load_agent(mf)

    if dqn_model is None or ppo_model is None:
        print("Both DQN and PPO models must be present in models/ directory.")
        return

    meta_agent = MetaAgent(dqn_model=dqn_model, ppo_model=ppo_model, step_size=0.10)

    # -------------------------------------------------------------------------
    # RUN ISOLATED DQN (Flat 1-unit bet size)
    # -------------------------------------------------------------------------
    env_dqn = TradingEnv(df=df, is_discrete=True)
    obs, _ = env_dqn.reset(options={'start_step': 0})

    dqn_y_true = []
    dqn_y_pred = []

    dqn_inference_times = []
    dqn_reconstruction_times = []

    terminated = False
    truncated = False

    # Mapping for DQN
    dqn_mapping = {0: -1, 1: -0.5, 2: 0, 3: 0.5, 4: 1}

    while not (terminated or truncated):
        # Match observation shape
        obs_for_pred = obs
        if dqn_model.observation_space.shape[-1] < obs.shape[-1]:
            obs_for_pred = obs[..., :dqn_model.observation_space.shape[-1]]

        t0 = time.perf_counter()
        action, _ = dqn_model.predict(obs_for_pred, deterministic=True)
        t1 = time.perf_counter()
        dqn_inference_times.append((t1 - t0) * 1000) # ms

        # Predict 1 (Long) if direction > 0, else 0 (Hold)
        direction = dqn_mapping[int(action)]
        predicted_class = 1 if direction > 0 else 0

        # Get next price to determine actual label
        current_step = env_dqn.current_step
        decision_idx = current_step + env_dqn.window_size - 1
        current_price = env_dqn._prices[decision_idx]

        t2 = time.perf_counter()
        next_obs, reward, terminated, truncated, info = env_dqn.step(action)
        t3 = time.perf_counter()
        dqn_reconstruction_times.append((t3 - t2) * 1000) # ms

        next_decision_idx = env_dqn.current_step + env_dqn.window_size - 1
        if next_decision_idx < len(env_dqn._prices):
            next_price = env_dqn._prices[next_decision_idx]
        else:
            next_price = env_dqn._prices[-1]

        # Ground truth: 1 if price went up, else 0
        actual_class = 1 if next_price > current_price else 0

        dqn_y_pred.append(predicted_class)
        dqn_y_true.append(actual_class)

        obs = next_obs

    # Calculate Isolated DQN Metrics
    dqn_cm = confusion_matrix(dqn_y_true, dqn_y_pred, labels=[0, 1])
    dqn_tn, dqn_fp, dqn_fn, dqn_tp = dqn_cm.ravel()

    dqn_recall = recall_score(dqn_y_true, dqn_y_pred, zero_division=0)
    dqn_accuracy = accuracy_score(dqn_y_true, dqn_y_pred)
    dqn_precision = precision_score(dqn_y_true, dqn_y_pred, zero_division=0)

    # -------------------------------------------------------------------------
    # RUN COMBINED SYSTEM (DQN + PPO Meta-Labeling)
    # -------------------------------------------------------------------------
    env_meta = TradingEnv(df=df, is_discrete=False)
    obs, _ = env_meta.reset(options={'start_step': 0})

    meta_y_true = []
    meta_y_pred = []
    meta_y_prob = []

    meta_inference_times = []
    meta_reconstruction_times = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        obs_for_pred = obs
        if meta_agent.observation_space.shape[-1] < obs.shape[-1]:
            obs_for_pred = obs[..., :meta_agent.observation_space.shape[-1]]

        t0 = time.perf_counter()
        # Custom prediction logic to extract probability for Log-Loss
        dqn_act, _ = meta_agent.dqn.predict(obs_for_pred, deterministic=True)
        direction = dqn_mapping[int(dqn_act)]

        if direction == 0:
            final_action = np.array([0.0])
            prob_long = 0.0 # No conviction to go long
        else:
            ppo_act, _ = meta_agent.ppo.predict(obs_for_pred, deterministic=True)
            raw_p = (ppo_act[0] + 1.0) / 2.0
            raw_p = np.clip(raw_p, 0.01, 0.99)

            z = (raw_p - 0.5) / np.sqrt(raw_p * (1.0 - raw_p))
            from scipy.stats import norm
            m = 2.0 * norm.cdf(z) - 1.0
            m_discrete = np.round(m / meta_agent.step_size) * meta_agent.step_size
            final_action = np.array([np.sign(direction) * m_discrete])

            # Probability that the current signal is a winning long
            if direction > 0:
                prob_long = raw_p
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

    delta_precision = meta_precision - dqn_precision

    avg_inference = np.mean(meta_inference_times)
    avg_reconstruction = np.mean(meta_reconstruction_times)

    print("==================================================")
    print("1. DECOUPLED AGENT TELEMETRY REPORT")
    print("==================================================")
    print("Baseline DQN (Isolated)")
    print(f"Confusion Matrix: TP={dqn_tp}, TN={dqn_tn}, FP={dqn_fp}, FN={dqn_fn}")
    print(f"Recall:   {dqn_recall:.4f}")
    print(f"Accuracy: {dqn_accuracy:.4f}")
    print(f"Precision: {dqn_precision:.4f}")
    print("-" * 50)
    print("Combined System (DQN + PPO)")
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
