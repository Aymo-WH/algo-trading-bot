import argparse
import os
import pandas as pd
from core.trading_gym import TradingEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import multiprocessing
import torch
import random
import numpy as np
from numba import njit

@njit
def _compute_tbm_labels_njit(prices, atr, opt_pt, opt_sl, max_holding_bars):
    labels = np.zeros(len(prices))
    for i in range(len(prices)):
        entry_price = prices[i]
        entry_atr = atr[i]
        upper_barrier = entry_price + (entry_atr * opt_pt[i])
        lower_barrier = entry_price - (entry_atr * opt_sl[i])

        hit = 0
        for j in range(1, max_holding_bars + 1):
            if i + j >= len(prices):
                break
            curr_price = prices[i + j]
            if curr_price >= upper_barrier:
                hit = 1
                break
            elif curr_price <= lower_barrier:
                hit = -1
                break
        labels[i] = hit
    return labels

def set_global_seed(seed=42):
    """
    Sets the global seed for deterministic reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    """
    Parses CLI arguments for training RL agents.

    Returns:
        argparse.Namespace: CLI arguments including model type, timesteps, and data dir.
    """
    parser = argparse.ArgumentParser(description="Train a trading agent.")
    parser.add_argument(
        "--model",
        type=str,
        default="ppo",
        choices=["ppo", "xgb"],
        help="Model to train (ppo or xgb). Default: ppo"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300000,
        help="Total timesteps to train. Default: 300000"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/train/",
        help="Directory containing training data. Default: data/train/"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the model. If not provided, defaults to models/{model}_trading_bot"
    )
    return parser.parse_args()

def validate_path(path_str, arg_name):
    """
    Validates that a given path is within the current working directory.

    This function mitigates Path Traversal vulnerabilities by resolving absolute paths
    and enforcing that they cannot traverse above the designated base directory.

    Args:
        path_str (str): The provided path.
        arg_name (str): Argument name for error logging.

    Returns:
        str: The safely resolved path.
    """
    if path_str is None:
        return None
    abs_path = os.path.abspath(path_str)
    base_dir = os.path.abspath('.')

    # Check if the resolved path starts with the base directory
    # Adding os.sep ensures that '/base/dir2' doesn't match '/base/dir'
    if not abs_path.startswith(base_dir + os.sep) and abs_path != base_dir:
        raise ValueError(f"Security Error: {arg_name} '{path_str}' traverses outside the base directory.")

    return path_str

def compute_tbm_labels(df, max_holding_bars=15):
    """
    Computes labels using the Triple-Barrier Method.
    Label 1 if Take Profit is hit first.
    Label -1 if Stop Loss is hit first.
    Label 0 if Time barrier is hit first or no barrier is hit.
    We strictly avoid look-ahead bias by only looking at future prices relative to the current step `i`.
    """
    prices = df['Close'].values
    atr = prices * 0.02 # Assuming fallback ATR calculation used in gym

    opt_pt = df['Optimal_PT'].values if 'Optimal_PT' in df.columns else np.full(len(df), 2.0)
    opt_sl = df['Optimal_SL'].values if 'Optimal_SL' in df.columns else np.full(len(df), 2.0)

    # Use the optimized njit function
    labels = _compute_tbm_labels_njit(prices, atr, opt_pt, opt_sl, max_holding_bars)

    return labels

def main():
    """
    Command-line execution flow for standalone model training.
    """
    set_global_seed(42)
    args = parse_args()

    # Validate paths
    args.data_dir = validate_path(args.data_dir, "--data_dir")
    args.save_path = validate_path(args.save_path, "--save_path")

    # Determine save path if not provided
    if args.save_path is None:
        save_path = f"models/{args.model}_trading_bot"
        if args.model == "xgb":
            save_path += ".json"
    else:
        save_path = args.save_path

    print(f"Training {args.model.upper()} agent for {args.timesteps} timesteps...")
    print(f"Data directory: {args.data_dir}")
    print(f"Model will be saved to: {save_path}")

    if args.model == "ppo":
        n_cpu = max(1, multiprocessing.cpu_count() - 1)
        print(f"Igniting {n_cpu} parallel environments for PPO training...")

        env = make_vec_env(
            TradingEnv,
            n_envs=n_cpu,
            seed=42,
            env_kwargs={"is_discrete": False, "data_dir": args.data_dir, "xgb_model_path": "models/xgb_trading_bot.json"},
            vec_env_cls=SubprocVecEnv
        )
        env.action_space.seed(42)
        model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.01, seed=42)
        model.learn(total_timesteps=args.timesteps)
        model.save(save_path)
        print("Training complete and model saved.")

    elif args.model == "xgb":
        import xgboost as xgb
        import glob
        pattern = os.path.join(args.data_dir, '*_data.csv')
        data_files = glob.glob(pattern)
        if not data_files:
            raise FileNotFoundError(f"No data files found in {args.data_dir} directory.")

        all_features = []
        all_labels = []

        feature_cols = ['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4']

        for file in data_files:
            df = pd.read_csv(file).dropna().reset_index(drop=True)
            if len(df) < 20:
                continue

            if not set(feature_cols).issubset(df.columns):
                continue

            labels = compute_tbm_labels(df)
            features = df[feature_cols].values

            # Map labels (-1, 0, 1) to (0, 1, 2)
            mapped_labels = labels + 1

            all_features.append(features)
            all_labels.append(mapped_labels)

        if not all_features:
            raise ValueError("No valid data loaded for XGBoost.")

        X = np.vstack(all_features)
        y = np.concatenate(all_labels)

        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            seed=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )

        print("Fitting XGBoost model on TBM labels...")
        model.fit(X, y)

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        model.save_model(save_path)
        print("XGBoost training complete and model saved.")

if __name__ == "__main__":
    main()

def train_xgb(ticker, **kwargs):
    """
    Trains an XGBoost Classifier on a specific ticker using Triple-Barrier labels.
    """
    set_global_seed(42)
    import xgboost as xgb

    data_dir = "data/train/"
    ticker_file = os.path.join(data_dir, f"{os.path.basename(ticker)}_data.csv")

    if not os.path.exists(ticker_file):
        raise FileNotFoundError(f"Data file for {ticker} not found.")

    df = pd.read_csv(ticker_file).dropna().reset_index(drop=True)
    feature_cols = ['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4']

    if not set(feature_cols).issubset(df.columns):
        raise ValueError(f"Missing required features in {ticker_file}.")

    labels = compute_tbm_labels(df)
    X = df[feature_cols].values
    y = labels + 1 # shift -1, 0, 1 to 0, 1, 2

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        seed=42,
        **kwargs
    )

    model.fit(X, y)
    return model

def train_ppo(ticker, total_timesteps=300000, **kwargs):
    """
    Trains a PPO (Proximal Policy Optimization) agent on a specific ticker.

    PPO is a policy gradient method adept at handling continuous action spaces.
    In the Meta-Labeling architecture, this acts as the secondary model predicting bet size.

    Args:
        ticker (str): The specific stock to train on.
        total_timesteps (int): Number of steps to train. Defaults to 300000.
        **kwargs: Extensible hyperparameter overrides (e.g. learning rate, clip range, entropy coef).

    Returns:
        stable_baselines3.PPO: The trained secondary agent.
    """
    set_global_seed(42)

    learning_rate = kwargs.get('ppo_lr', 3e-4)
    clip_range = kwargs.get('ppo_clip', 0.2)
    ent_coef = kwargs.get('ppo_ent', 0.08)

    is_discrete = False
    data_dir = "data/train/"

    ticker_file = os.path.join(data_dir, f"{ticker}_data.csv")
    if os.path.exists(ticker_file):
        df = pd.read_csv(ticker_file)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        env = make_vec_env(
            TradingEnv,
            n_envs=max(1, multiprocessing.cpu_count() - 1),
            seed=42,
            env_kwargs={"df": df, "is_discrete": False, "xgb_model_path": "models/xgb_trading_bot.json"},
            vec_env_cls=SubprocVecEnv
        )
    else:
        env = make_vec_env(
            TradingEnv,
            n_envs=max(1, multiprocessing.cpu_count() - 1),
            seed=42,
            env_kwargs={"is_discrete": False, "data_dir": data_dir, "xgb_model_path": "models/xgb_trading_bot.json"},
            vec_env_cls=SubprocVecEnv
        )

    env.action_space.seed(42)

    model = PPO("MlpPolicy", env, verbose=0, learning_rate=learning_rate, clip_range=clip_range, ent_coef=ent_coef, seed=42)
    model.learn(total_timesteps=total_timesteps)
    return model
