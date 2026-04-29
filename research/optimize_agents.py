import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import argparse
import optuna
import pandas as pd
import numpy as np
import json
import torch
import random
from stable_baselines3.common.callbacks import BaseCallback

from train_agent import train_xgb, train_ppo
from evaluate_agents import evaluate_model
from core.pbo_validator import PBOValidator

# Global matrix to store the out-of-sample return series for every trial
TRIAL_RETURNS_MATRIX = {}

def set_global_seed(seed=42):
    """
    Sets the global seed for deterministic reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def objective(trial, timesteps, ticker):
    # 1. Sample Hyperparameters using Log-Uniform distributions
    xgb_lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    xgb_max_depth = trial.suggest_int("max_depth", 3, 10)

    ppo_lr = trial.suggest_float("ppo_lr", 1e-5, 1e-3, log=True)
    ppo_clip = trial.suggest_float("ppo_clip", 0.1, 0.4)
    ppo_ent = trial.suggest_float("ppo_ent", 0.0001, 0.01, log=True)

    print(f"\n--- Trial {trial.number} ({ticker}) ---")

    # 2. Train Models with these specific hyperparameters
    xgb_model = train_xgb(ticker=ticker, learning_rate=xgb_lr, max_depth=xgb_max_depth)
    ppo_model = train_ppo(ticker=ticker, total_timesteps=timesteps, ppo_lr=ppo_lr, ppo_clip=ppo_clip, ppo_ent=ppo_ent)

    # 3. Evaluate Out-Of-Sample
    _, _, oos_returns = evaluate_model(xgb_model, ppo_model, ticker=ticker)

    # 4. Save the chronological return path to our global dictionary
    TRIAL_RETURNS_MATRIX[f"Trial_{trial.number}"] = oos_returns

    # 5. Calculate fitness (Sharpe Ratio proxy)
    if np.std(oos_returns) == 0:
        return -1.0
    sharpe = np.mean(oos_returns) / np.std(oos_returns)
    return sharpe

def run_optimization(n_trials=20, timesteps=10000, config_path='config/config_phase1.json', specific_ticker=None):
    set_global_seed(42)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        active_tickers = [specific_ticker] if specific_ticker else config.get("tickers", [])
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        return

    if not active_tickers:
        print("No tickers found in config.")
        return

    for ticker in active_tickers:
        print(f"\n{'='*40}\nOptimizing for ticker: {ticker}\n{'='*40}")
        global TRIAL_RETURNS_MATRIX
        TRIAL_RETURNS_MATRIX = {} # reset matrix per ticker

        # Create the study
        study = optuna.create_study(
            direction="maximize", 
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )

        # FIX: The execution engine is restored here!
        study.optimize(lambda trial: objective(trial, timesteps, ticker), n_trials=n_trials)

        print(f"\nBest Hyperparameters for {ticker}:", study.best_params)

        # Re-train the models using the best discovered hyperparameters to 'restore' the engine
        print(f"\n[SYSTEM] Restoring execution engine for {ticker}...")

        best_xgb = train_xgb(
            ticker=ticker,
            learning_rate=study.best_params.get("learning_rate"),
            max_depth=study.best_params.get("max_depth")
        )
        best_ppo = train_ppo(
            ticker=ticker,
            total_timesteps=timesteps,
            ppo_lr=study.best_params.get("ppo_lr"),
            ppo_clip=study.best_params.get("ppo_clip"),
            ppo_ent=study.best_params.get("ppo_ent")
        )

        # Save to the specific paths expected by live_inference.py
        os.makedirs("models", exist_ok=True)
        best_xgb.save_model("models/xgb_trading_bot.json")
        best_ppo.save("models/ppo_meta_labeler")
        print(f"✅ Execution engine restored: Best models for {ticker} saved to models/.")

        # === CALCULATE PBO USING THE TRUE CSCV MATRIX ===
        performance_matrix = pd.DataFrame(TRIAL_RETURNS_MATRIX)

        if not performance_matrix.empty:
            print(f"\nCalculating PBO across all Optuna Trials for {ticker}...")
            validator = PBOValidator(performance_matrix, num_partitions=4)
            pbo_score, _ = validator.calculate_pbo()

            print(f"✅ True Probability of Backtest Overfitting (PBO) for {ticker}: {pbo_score * 100:.2f}%")
        else:
            print(f"⚠️ Could not calculate PBO for {ticker} as the performance matrix is empty.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha Search Engine")
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--timesteps', type=int, default=10000)
    parser.add_argument('--config', type=str, default='config/config_phase1.json')
    parser.add_argument('--ticker', type=str, default=None, help='Specific ticker to run in parallel mode')
    args = parser.parse_args()

    run_optimization(n_trials=args.trials, timesteps=args.timesteps, config_path=args.config, specific_ticker=args.ticker)