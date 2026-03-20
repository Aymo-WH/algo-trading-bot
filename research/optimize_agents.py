import argparse
import optuna
import pandas as pd
import numpy as np
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_agent import train_dqn, train_ppo
from evaluate_agents import evaluate_model
from core.pbo_validator import PBOValidator

# Global matrix to store the out-of-sample return series for every trial
TRIAL_RETURNS_MATRIX = {}

def objective(trial, timesteps, ticker):
    # 1. Sample Hyperparameters using Log-Uniform distributions
    dqn_lr = trial.suggest_float("dqn_lr", 1e-5, 1e-3, log=True)
    dqn_target_update = trial.suggest_int("dqn_target_update", 1000, 10000)

    ppo_lr = trial.suggest_float("ppo_lr", 1e-5, 1e-3, log=True)
    ppo_clip = trial.suggest_float("ppo_clip", 0.1, 0.4)
    ppo_ent = trial.suggest_float("ppo_ent", 0.0001, 0.01, log=True)

    print(f"\n--- Trial {trial.number} ({ticker}) ---")

    # 2. Train Models with these specific hyperparameters (shortened timesteps for search speed)
    # Note: In a real architecture, train_dqn and train_ppo need to accept kwargs
    dqn_model = train_dqn(ticker=ticker, total_timesteps=timesteps, dqn_lr=dqn_lr, dqn_target_update=dqn_target_update)
    ppo_model = train_ppo(ticker=ticker, total_timesteps=timesteps, ppo_lr=ppo_lr, ppo_clip=ppo_clip, ppo_ent=ppo_ent)

    # 3. Evaluate Out-Of-Sample
    # evaluate_model must return the chronological sequence of portfolio returns
    _, _, oos_returns = evaluate_model(dqn_model, ppo_model, ticker=ticker)

    # 4. Save the chronological return path to our global dictionary (The N trials)
    TRIAL_RETURNS_MATRIX[f"Trial_{trial.number}"] = oos_returns

    # 5. Calculate fitness (Sharpe Ratio proxy)
    if np.std(oos_returns) == 0:
        return -1.0
    sharpe = np.mean(oos_returns) / np.std(oos_returns)
    return sharpe

def run_optimization(n_trials=20, timesteps=10000, config_path='config/config.json'):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        tickers = config.get("tickers", [])
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        return

    if not tickers:
        print("No tickers found in config.")
        return

    for ticker in tickers:
        print(f"\n{'='*40}\nOptimizing for ticker: {ticker}\n{'='*40}")
        global TRIAL_RETURNS_MATRIX
        TRIAL_RETURNS_MATRIX = {} # reset matrix per ticker

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
        study.optimize(lambda trial: objective(trial, timesteps, ticker), n_trials=n_trials)

        print(f"\nBest Hyperparameters for {ticker}:", study.best_params)

        # === CALCULATE PBO USING THE TRUE CSCV MATRIX ===
        # Convert dictionary to TxN DataFrame
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
    parser.add_argument('--config', type=str, default='config/config.json')
    args = parser.parse_args()

    run_optimization(n_trials=args.trials, timesteps=args.timesteps, config_path=args.config)
