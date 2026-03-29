import argparse
import os
import pandas as pd
from core.trading_gym import TradingEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import multiprocessing

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
        choices=["ppo", "dqn"],
        help="Model to train (ppo or dqn). Default: ppo"
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

def main():
    """
    Command-line execution flow for standalone model training.
    """
    args = parse_args()

    # Validate paths
    args.data_dir = validate_path(args.data_dir, "--data_dir")
    args.save_path = validate_path(args.save_path, "--save_path")

    # Determine save path if not provided
    if args.save_path is None:
        save_path = f"models/{args.model}_trading_bot"
    else:
        save_path = args.save_path

    print(f"Training {args.model.upper()} agent for {args.timesteps} timesteps...")
    print(f"Data directory: {args.data_dir}")
    print(f"Model will be saved to: {save_path}")

    # Initialize the environment and model
    # DQN requires discrete actions, PPO can handle both but usually continuous
    # Based on original scripts: PPO -> is_discrete=False, DQN -> is_discrete=True

    if args.model == "ppo":
        # Use vectorized environment for PPO to improve training speed
        # Determine number of CPUs to use
        n_cpu = max(1, multiprocessing.cpu_count() - 1)
        print(f"Igniting {n_cpu} parallel environments for PPO training...")

        # We use DummyVecEnv by default (when vec_env_cls is not specified) as it is often faster
        # for simple environments due to lower overhead than SubprocVecEnv.
        # Benchmarks on this environment showed DummyVecEnv ~370 FPS vs SubprocVecEnv ~220 FPS vs Single Env ~215 FPS.
        env = make_vec_env(
            TradingEnv,
            n_envs=n_cpu,
            env_kwargs={"is_discrete": False, "data_dir": args.data_dir},
            vec_env_cls=SubprocVecEnv
        )
        model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.01)

    elif args.model == "dqn":
        # DQN in SB3 doesn't support vector envs in the same way for efficiency gains usually
        # and requires discrete actions
        is_discrete = True
        env = TradingEnv(is_discrete=is_discrete, data_dir=args.data_dir)
        # Original DQN script used target_update_interval=500
        model = DQN("MlpPolicy", env, verbose=1, target_update_interval=500)

    # Command the model to learn
    model.learn(total_timesteps=args.timesteps)

    # Save the trained model
    model.save(save_path)
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()

def train_dqn(ticker, total_timesteps=10000, **kwargs):
    """
    Trains a DQN (Deep Q-Network) agent on a specific ticker.

    DQN is a value-based reinforcement learning algorithm ideal for discrete action spaces.
    In the Meta-Labeling architecture, this acts as the primary model predicting trade direction.

    Args:
        ticker (str): The specific stock to train on.
        total_timesteps (int): Number of steps to train. Defaults to 10000.
        **kwargs: Extensible hyperparameter overrides (e.g. learning rate, target update freq).

    Returns:
        stable_baselines3.DQN: The trained primary agent.
    """
    # Determine learning rate and target update interval from kwargs
    learning_rate = kwargs.get('dqn_lr', 1e-4)
    target_update_interval = kwargs.get('dqn_target_update', 1000)

    # DQN requires discrete actions
    is_discrete = True

    # Check if a specific file exists, if so load that file, else pass data_dir
    data_dir = "data/train/"
    # Ideally we should pass a specific dataframe, but for simplicity, we pass data_dir
    # However, memory mentions: "train_agent.py acts as the unified training script... defaults to 50,000 timesteps...".
    # And we know TradingEnv loads all *_data.csv in data_dir by default.
    # To train on a specific ticker, we should pass its dataframe.

    ticker_file = os.path.join(data_dir, f"{ticker}_data.csv")
    if os.path.exists(ticker_file):
        df = pd.read_csv(ticker_file)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        env = TradingEnv(df=df, is_discrete=is_discrete)
    else:
        # Fallback to loading all if specific ticker file not found (though unexpected)
        env = TradingEnv(is_discrete=is_discrete, data_dir=data_dir)

    model = DQN("MlpPolicy", env, verbose=0, learning_rate=learning_rate, target_update_interval=target_update_interval)
    model.learn(total_timesteps=total_timesteps)
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
    learning_rate = kwargs.get('ppo_lr', 3e-4)
    clip_range = kwargs.get('ppo_clip', 0.2)
    ent_coef = kwargs.get('ppo_ent', 0.01)

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
            env_kwargs={"df": df, "is_discrete": False},
            vec_env_cls=SubprocVecEnv
        )
    else:
        env = make_vec_env(
            TradingEnv,
            n_envs=max(1, multiprocessing.cpu_count() - 1),
            env_kwargs={"is_discrete": False, "data_dir": data_dir},
            vec_env_cls=SubprocVecEnv
        )

    model = PPO("MlpPolicy", env, verbose=0, learning_rate=learning_rate, clip_range=clip_range, ent_coef=ent_coef)
    model.learn(total_timesteps=total_timesteps)
    return model
