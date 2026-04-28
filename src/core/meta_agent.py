import numpy as np

class MetaAgent:
    def __init__(self, xgb_model, ppo_model, step_size=0.1):
        self.xgb_model = xgb_model
        self.ppo = ppo_model
        self.step_size = step_size
        self.action_space = ppo_model.action_space

    def predict(self, pca_features, volatility, drawdown, deterministic=True):
        # 1. Primary Model (XGBoost) predicts direction
        probs = self.xgb_model.predict_proba(pca_features.reshape(1, -1))[0]
        pred_class = np.argmax(probs)
        xgb_signal = float(pred_class) - 1.0
        xgb_prob = float(probs[pred_class])

        if xgb_signal == 0.0:
            return np.array([0.0]), None

        # 2. Build the 1D Meta State array for PPO
        obs = np.array([xgb_signal, xgb_prob, volatility, drawdown], dtype=np.float32)

        # 3. Secondary Model (PPO) predicts bet size
        ppo_act, _ = self.ppo.predict(obs, deterministic=deterministic)
        raw_size = np.clip(ppo_act[0], 0.0, 1.0)
        m_discrete = np.round(raw_size / self.step_size) * self.step_size

        # 4. Return actual market action (Direction * Size) for the live exchange
        return np.array([xgb_signal * m_discrete]), None
