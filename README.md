# 📈 Quantitative RL Trading Laboratory 

An institutional-grade algorithmic trading framework built on Reinforcement Learning (Stable-Baselines3) and custom Gymnasium environments. This repository implements advanced quantitative finance methodologies derived from Marcos López de Prado's *Advances in Financial Machine Learning*.

---

## 🧠 System Architecture

Unlike standard "retail" trading bots that rely on raw price action and fixed-time horizons, this laboratory utilizes a robust, statistically sound pipeline to generate purely algorithmic Alpha.

### Core Technologies
- **Environment:** Custom Vectorized `trading_gym.py` (OpenAI Gymnasium) optimized for $O(1)$ ring-buffer memory execution.
- **Brain 1 (Directional Signal):** Deep Q-Network (DQN) for discrete regime identification.
- **Brain 2 (Meta-Labeling / Risk Sizing):** Proximal Policy Optimization (PPO) for continuous probability-based bet sizing.
- **Data Engine:** Asynchronous Yahoo Finance ingestion with automated caching.

---

## 🔬 Advanced Quantitative Methodologies

This project actively implements the following institutional mechanisms:

- **Fractional Differentiation (FFD):** Transforming non-stationary price data into stationary series while preserving maximum historical memory (optimizing the $d^*$ coefficient via Augmented Dickey-Fuller testing).
- **The Triple-Barrier Method:** Path-dependent dynamic reward horizons scaling with historical volatility (ATR) to mimic real-world stop-loss and profit-taking mechanics.
- **Orthogonal Feature Processing:** Utilizing Principal Component Analysis (PCA) to eliminate the "Substitution Effect" caused by collinear technical indicators (MACD, RSI).
- **Meta-Labeling with Size Discretization:** Separating the directional signal (Long/Short) from the bet sizing. The continuous bet size is calculated via the Normal CDF of the classification $z$-score and discretized into a step-function to eliminate micro-rebalancing fees.
- **Purged Walk-Forward Cross Validation:** Eliminating Look-Ahead bias and serial correlation leakage by enforcing strict embargoes between training and testing folds.
- **Deflated Sharpe Ratio (DSR):** Logging the variance of all failed training iterations to calculate the Probability of Backtest Overfitting (PBO), ensuring out-of-sample performance is statistically significant.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

###2. Train the Agents

```Bash
python train_agent.py --data_dir ./data --save_path ./models
```

###3. Evaluate the Deflated Sharpe Ratio

```Bash
python evaluate_agents.py
```

⚠️ Disclaimer
This repository is an academic and mathematical research laboratory. It does not constitute financial advice. The models herein are experimental and intended for quantitative research only.


***
