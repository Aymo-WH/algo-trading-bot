import time
import numpy as np
import pandas as pd
from trading_gym import TradingEnv

def run_benchmark():
    # Create dummy data
    np.random.seed(42)
    n_steps = 10000
    df = pd.DataFrame({
        'Close': np.random.lognormal(0, 0.01, n_steps).cumprod() * 100,
        'Close_FFD': np.random.randn(n_steps),
        'Sentiment_Score': np.random.randn(n_steps),
        'PCA_1': np.random.randn(n_steps),
        'PCA_2': np.random.randn(n_steps),
        'PCA_3': np.random.randn(n_steps),
        'PCA_4': np.random.randn(n_steps),
        'PCA_5': np.random.randn(n_steps),
        'ATR': np.random.uniform(0.5, 2.0, n_steps),
        'Optimal_PT': np.random.uniform(1.5, 3.0, n_steps),
        'Optimal_SL': np.random.uniform(1.5, 3.0, n_steps),
    })

    env = TradingEnv(df=df, is_discrete=True)
    env.reset()

    # Run a bunch of steps
    start_time = time.time()
    for _ in range(5000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()

    end_time = time.time()
    print(f"Time taken for 5000 steps: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
