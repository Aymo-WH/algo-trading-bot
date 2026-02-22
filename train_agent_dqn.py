from trading_gym import TradingEnv
from stable_baselines3 import DQN

# Initialize the environment
env = TradingEnv()

# Initialize the DQN model
model = DQN("MlpPolicy", env, verbose=1, target_update_interval=500)

# Command the model to learn for 50000 total timesteps
model.learn(total_timesteps=50000)

# Save the trained model
model.save("models/dqn_trading_bot")
