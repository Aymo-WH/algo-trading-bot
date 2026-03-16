# 🤖 Quantitative AI Trading Architecture: The Meta-Labeling System
**Version:** 8.0 (Institutional Grade)
**Core Methodology:** Marcos López de Prado (*Advances in Financial Machine Learning*)

## 🏛️ Executive Summary
This repository implements an advanced algorithmic trading pipeline moving beyond standard retail technical analysis. The architecture utilizes mathematical stationarity, strict point-in-time orthogonalization, and a dual-agent Reinforcement Learning (RL) meta-labeling system to generate and size trades. It is secured against multiple testing selection bias via Combinatorially Symmetric Cross-Validation (CSCV).

## ⚙️ Core Components
* **Fractional Differentiation (FFD) & Point-in-Time PCA:** Compresses correlated indicators (RSI, MACD, Bollinger, ATR) into stationary, orthogonal vectors without Look-Ahead Bias.
* **Meta-Labeling:** A Deep Q-Network (DQN) generates directional signals, while a Proximal Policy Optimization (PPO) agent predicts the probability of success, sizing the bet using the standard Normal CDF ($m = 2Z[z] - 1$).
* **Institutional Execution:** The Gym environment implements the **ETF Trick** (deducting fees as negative dividends) and the **Triple-Barrier Method** (dynamic Take-Profit, Stop-Loss, and Time Limits).
* **Optuna Alpha Search & PBO Validator:** Hyperparameters are optimized via Randomized Search. The chronological out-of-sample paths of all tested configurations are saved into a $T \times N$ matrix to calculate the Probability of Backtest Overfitting (PBO).
* **Research & Optimization:** Laboratory tools (benchmark and optimization scripts) are located in the `research/` directory.
