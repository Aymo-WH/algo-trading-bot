from trading_gym import TradingEnv
from stable_baselines3 import DQN

# Initialize the environment
env = TradingEnv()

# Initialize the DQN model
model = DQN("MlpPolicy", env, verbose=1)

# Command the model to learn for 10000 total timesteps
model.learn(total_timesteps=10000)

# Save the trained model
model.save("dqn_trading_bot")
