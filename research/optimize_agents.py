import argparse
import optuna
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_agent import train_dqn, train_ppo
from evaluate_agents import evaluate_model
from pbo_validator import PBOValidator

# Global matrix to store the out-of-sample return series for every trial
TRIAL_RETURNS_MATRIX = {}

def objective(trial, timesteps):
    # 1. Sample Hyperparameters using Log-Uniform distributions
    dqn_lr = trial.suggest_float("dqn_lr", 1e-5, 1e-3, log=True)
    dqn_target_update = trial.suggest_int("dqn_target_update", 1000, 10000)

    ppo_lr = trial.suggest_float("ppo_lr", 1e-5, 1e-3, log=True)
    ppo_clip = trial.suggest_float("ppo_clip", 0.1, 0.4)
    ppo_ent = trial.suggest_float("ppo_ent", 0.0001, 0.01, log=True)

    print(f"\n--- Trial {trial.number} ---")

    # 2. Train Models with these specific hyperparameters (shortened timesteps for search speed)
    # Note: In a real architecture, train_dqn and train_ppo need to accept kwargs
    dqn_model = train_dqn(ticker='AAPL', total_timesteps=timesteps, dqn_lr=dqn_lr, dqn_target_update=dqn_target_update)
    ppo_model = train_ppo(ticker='AAPL', total_timesteps=timesteps, ppo_lr=ppo_lr, ppo_clip=ppo_clip, ppo_ent=ppo_ent)

    # 3. Evaluate Out-Of-Sample
    # evaluate_model must return the chronological sequence of portfolio returns
    _, _, oos_returns = evaluate_model(dqn_model, ppo_model, ticker='AAPL')

    # 4. Save the chronological return path to our global dictionary (The N trials)
    TRIAL_RETURNS_MATRIX[f"Trial_{trial.number}"] = oos_returns

    # 5. Calculate fitness (Sharpe Ratio proxy)
    if np.std(oos_returns) == 0:
        return -1.0
    sharpe = np.mean(oos_returns) / np.std(oos_returns)
    return sharpe

def run_optimization(n_trials=20, timesteps=10000):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
    study.optimize(lambda trial: objective(trial, timesteps), n_trials=n_trials)

    print("\nBest Hyperparameters:", study.best_params)

    # === CALCULATE PBO USING THE TRUE CSCV MATRIX ===
    # Convert dictionary to TxN DataFrame
    performance_matrix = pd.DataFrame(TRIAL_RETURNS_MATRIX)

    print("\nCalculating PBO across all Optuna Trials...")
    validator = PBOValidator(performance_matrix, num_partitions=4)
    pbo_score, _ = validator.calculate_pbo()

    print(f"✅ True Probability of Backtest Overfitting (PBO): {pbo_score * 100:.2f}%")

if __name__ == "__main__":
    print("🧠 Welcome to the Alpha Search Engine")
    try:
        user_trials = int(input("Enter number of Optuna trials (e.g., 50): "))
        user_timesteps = int(input("Enter RL timesteps per trial (e.g., 50000): "))
    except ValueError:
        print("Invalid input. Defaulting to 10 trials / 10000 timesteps.")
        user_trials, user_timesteps = 10, 10000

    # Ensure run_optimization passes user_timesteps down to the train_dqn/train_ppo functions
    run_optimization(n_trials=user_trials, timesteps=user_timesteps)
