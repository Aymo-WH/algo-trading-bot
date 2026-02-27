import time
import gymnasium as gym
from trading_gym import TradingEnv

def benchmark_init(n_envs=10):
    start_time = time.time()
    envs = []
    for _ in range(n_envs):
        env = TradingEnv(data_dir='data/train')
        envs.append(env)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Time to initialize {n_envs} environments: {duration:.4f} seconds")
    print(f"Average time per environment: {duration/n_envs:.4f} seconds")

    # Cleanup
    for env in envs:
        env.close()

if __name__ == "__main__":
    benchmark_init(n_envs=20)
