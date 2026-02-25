import argparse
import os
from trading_gym import TradingEnv
from stable_baselines3 import PPO, DQN

def train_model(model_type, total_timesteps, data_dir, save_path):
    # Determine if environment should be discrete based on model type
    is_discrete = (model_type == 'dqn')

    print(f"Initializing TradingEnv with is_discrete={is_discrete}, data_dir={data_dir}")
    env = TradingEnv(is_discrete=is_discrete, data_dir=data_dir)

    print(f"Initializing {model_type.upper()} model...")
    if model_type == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1)
    elif model_type == 'dqn':
        model = DQN("MlpPolicy", env, verbose=1, target_update_interval=500)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    if save_path is None:
        save_path = f"models/{model_type}_trading_bot"

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Saving model to {save_path}...")
    model.save(save_path)
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a trading agent (PPO or DQN).")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ppo", "dqn"],
        help="Model type to train: 'ppo' or 'dqn'"
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total training timesteps (default: 50000)"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/train/",
        help="Directory containing training data (default: 'data/train/')"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the trained model. Defaults to 'models/{model_type}_trading_bot'"
    )

    args = parser.parse_args()

    train_model(
        model_type=args.model,
        total_timesteps=args.timesteps,
        data_dir=args.data_dir,
        save_path=args.save_path
    )
