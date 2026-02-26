import argparse
from trading_gym import TradingEnv
from stable_baselines3 import PPO, DQN

def parse_args():
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
        default=50000,
        help="Total timesteps to train. Default: 50000"
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

def main():
    args = parse_args()

    # Determine save path if not provided
    if args.save_path is None:
        save_path = f"models/{args.model}_trading_bot"
    else:
        save_path = args.save_path

    print(f"Training {args.model.upper()} agent for {args.timesteps} timesteps...")
    print(f"Data directory: {args.data_dir}")
    print(f"Model will be saved to: {save_path}")

    # Initialize the environment
    # DQN requires discrete actions, PPO can handle both but usually continuous
    # Based on original scripts: PPO -> is_discrete=False, DQN -> is_discrete=True
    is_discrete = (args.model == "dqn")
    env = TradingEnv(is_discrete=is_discrete, data_dir=args.data_dir)

    # Initialize the model
    if args.model == "ppo":
        model = PPO("MlpPolicy", env, verbose=1)
    elif args.model == "dqn":
        # Original DQN script used target_update_interval=500
        model = DQN("MlpPolicy", env, verbose=1, target_update_interval=500)

    # Command the model to learn
    model.learn(total_timesteps=args.timesteps)

    # Save the trained model
    model.save(save_path)
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
