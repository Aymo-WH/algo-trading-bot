import numpy as np
from scipy.stats import norm

class MetaAgent:
    def __init__(self, dqn_model, ppo_model, step_size=0.1):
        self.dqn = dqn_model
        self.ppo = ppo_model
        self.step_size = step_size

    def predict(self, obs, deterministic=True):
        # 1. Primary Model (DQN) - Signal Generation
        dqn_act, _ = self.dqn.predict(obs, deterministic=deterministic)

        # Map DQN discrete actions (0-4) to Direction (-1, 0, 1)
        mapping = {0: -1, 1: -0.5, 2: 0, 3: 0.5, 4: 1}
        direction = mapping[int(dqn_act)]

        # If DQN says hold, we veto the trade entirely
        if direction == 0:
            return np.array([0.0]), None

        # 2. Secondary Model (PPO) - Bet Sizing
        ppo_act, _ = self.ppo.predict(obs, deterministic=deterministic)

        # Map PPO continuous action [-1, 1] to a raw probability [0, 1]
        raw_p = (ppo_act[0] + 1.0) / 2.0
        raw_p = np.clip(raw_p, 0.01, 0.99) # Prevent infinity in z-score calculation

        # 3. Marcos López de Prado Math (z-score & Normal CDF)
        z = (raw_p - 0.5) / np.sqrt(raw_p * (1.0 - raw_p))
        m = 2.0 * norm.cdf(z) - 1.0

        # 4. Size Discretization (Thresholding to prevent jitter)
        m_discrete = np.round(m / self.step_size) * self.step_size

        # Final Action: Direction * Conviction
        # Use np.sign to extract purely the Long/Short direction from DQN,
        # then multiply by the PPO discrete size
        final_action = np.sign(direction) * m_discrete

        # Return format matches stable-baselines3 predict() signature
        return np.array([final_action]), None
