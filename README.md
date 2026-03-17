# 🤖 Quantitative AI Trading Architecture: Information-Driven Meta-Labeling
**Version:** 9.0 (Cloud-Ready Institutional Grade)
**Core Methodology:** Marcos López de Prado (*Advances in Financial Machine Learning*)

## 🏛️ Executive Summary
This repository implements an advanced algorithmic trading pipeline moving beyond standard retail technical analysis. By utilizing Information-Driven Dollar Bars, Point-in-Time PCA, and a dual-agent Reinforcement Learning (RL) meta-labeling framework, the system captures non-linear market dynamics safely.

## ⚙️ Core Quantitative Infrastructure

### 1. Information-Driven Dollar Bars
Traditional time bars suffer from heteroscedasticity. We construct Dollar Bars by sampling the market only when a dynamic threshold of fiat currency ($M$) is exchanged. The pipeline dynamically compresses 1-hour intraday proxy data into Information-Driven events.

### 2. Fractional Differentiation (FFD)
Standard integer differencing eliminates system memory. We apply a Fixed-Width Window Fractional Differentiation (FFD) using an optimized C-level `np.convolve` routine to preserve maximum memory while passing the Augmented Dickey-Fuller (ADF) test for stationarity.

### 3. The Meta-Labeling Framework (DQN + PPO)
We deploy a dual-network architecture to separate directional conviction from bet sizing:
* **Primary Model (DQN):** Generates directional signals (Long, Short, Hold).
* **Secondary Model (PPO):** Acts as the meta-labeler, predicting the probability $p$ that the DQN's signal is correct. Bet size is scaled dynamically using the standard Normal CDF.

### 4. Optimal Trading Rule (OTR) via Stochastic Calculus
To prevent execution-level overfitting, Stop-Loss and Profit-Taking barriers are dynamically generated using a rolling 60-day Ornstein-Uhlenbeck (O-U) stochastic process. The pipeline simulates 10,000 synthetic futures to calibrate execution limits independent of the neural network.

### 5. CSCV & Probability of Backtest Overfitting (PBO)
The `optimize_agents.py` script uses Optuna to search for alpha, saving the out-of-sample path of every tested configuration into a $T \times N$ matrix. This matrix is evaluated using Combinatorially Symmetric Cross-Validation (CSCV) to output a strict PBO metric.

## 🚀 Cloud Execution
The optimization engine is built for headless GPU execution:
`python research/optimize_agents.py --trials 500 --timesteps 1000000`