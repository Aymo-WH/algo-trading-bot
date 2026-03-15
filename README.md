# 🤖 Quantitative AI Trading Architecture (Meta-Labeling)

This repository implements an institutional-grade algorithmic trading pipeline based on the frameworks of Marcos López de Prado. It moves beyond standard retail technical analysis by employing stationarity transformations, mathematical orthogonalization, and secondary meta-labeling.

## 🏛️ System Architecture

### 1. Data Ingestion & Transformation
* **Fractional Differentiation (FFD):** Raw price series are made stationary without sacrificing memory. We use an expanding fixed-width window, optimizing the $d^*$ coefficient via Augmented Dickey-Fuller tests.
* **Orthogonalization:** Correlated technical indicators (RSI, MACD, Bollinger Bands) are compressed using strictly point-in-time Principal Component Analysis (PCA) to prevent the "Substitution Effect" and Look-Ahead Bias.

### 2. The Meta-Agent (Dual Brain System)
* **Primary Model (Signal Generation):** A Deep Q-Network (DQN) analyzes the orthogonalized state to generate strict discrete signals (Long, Short, Veto). 
* **Secondary Model (Meta-Labeling):** A Proximal Policy Optimization (PPO) agent acts as a bet-sizer. It evaluates the probability of the DQN's success. The continuous output is transformed via a z-score and standard Normal CDF ($m = 2Z[z] - 1$) to generate a mathematical conviction level.
* **Size Discretization:** The continuous conviction is passed through a step-function to prevent micro-rebalancing jitter.

### 3. Institutional Execution & Validation
* **The ETF Trick:** The Gym environment tracks a theoretical $1 invested ($K_t$), cleanly separating mark-to-market performance from transaction costs (treated as a negative dividend) to prevent fictitious compounded returns.
* **Purged & Embargoed Cross-Validation:** The MDP transitions enforce strict 1% embargo boundaries and purge overlapping labels to prevent serial correlation leakage during Reinforcement Learning.

## 🚀 Future Roadmap
* Information-Driven Bars (Volume/Dollar) via high-frequency tick data.
* Combinatorially Symmetric Cross-Validation (CSCV) for Probability of Backtest Overfitting (PBO) reporting.
