from trading_gym import TradingEnv
from stable_baselines3 import PPO

# Initialize the environment
env = TradingEnv(is_discrete=False, data_dir='data/train/')

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Command the model to learn for 50000 total timesteps
model.learn(total_timesteps=50000)

# Save the trained model
model.save("models/ppo_trading_bot")
