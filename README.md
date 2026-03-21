# 🏎️ Information-Driven Algorithmic Trading Architecture (V9.0)

Traditional algorithmic trading relies on chronological time bars (like 1-hour or daily candles). But time is a terrible way to measure markets. Markets do not move because time passes; they move because capital changes hands.

This repository is an institutional-grade Reinforcement Learning trading engine that throws away the clock. It samples the market like a machine, learns the underlying physics of different asset classes, and executes mathematically sized trades.



### 🧠 Core Engineering Features

* **Information-Driven Dollar Bars:** We neutralize market noise and heteroscedasticity by forming bars only when a dynamic threshold of fiat currency is exchanged.
* **The Multi-Asset Pit Stop:** A highly modular configuration pipeline. Swap from high-volatility Crypto to slow-moving Global Macro ETFs instantly. The data factory automatically wipes old memory to prevent cross-asset pollution.
* **Meta-Labeling (Dual-Brain System):** We strictly separate "Direction" from "Conviction". A Deep Q-Network (DQN) decides whether to go Long or Short. A secondary Proximal Policy Optimization (PPO) agent acts as the risk manager, continuously sizing the bet based on statistical confidence.
* **Mathematical Memory:** Real-time extraction of Principal Component Analysis (PCA) and Fractional Differentiation scaling matrices, exported dynamically to strictly prevent Look-Ahead Bias during live inference.

graph TD
classDef factory fill:#f9f2f4,stroke:#b35d7f,stroke-width:2px;
classDef agent fill:#e6f3ff,stroke:#4a90e2,stroke-width:2px;
classDef eval fill:#e9f5e9,stroke:#5cb85c,stroke-width:2px;

%% Data Factory Phase
subgraph Phase 1: The Data Factory
    A[Raw Hourly API Data] --> B{Dollar Volume > Threshold?}
    B -- Yes --> C[Information-Driven Dollar Bar]
    B -- No --> A
    C --> D[Fractional Differentiation]
    D --> E[PCA Orthogonalization]
    E --> F[(Save Local Matrices)]
end
class A,B,C,D,E,F factory;

%% Alpha Search Phase
subgraph Phase 2: Alpha Search & Meta-Labeling
    E --> G[DQN: Directional Scout]
    E --> H[PPO: Risk Manager]
    G -- Long/Short/Hold --> H
    H -- Statistical Confidence % --> I[Dynamic Trade Size]
end
class G,H,I agent;

%% CSCV Phase
subgraph Phase 3: Out-of-Sample Evaluation
    I --> J[O-U Stochastic Barriers]
    J --> K[CSCV Matrix Generator]
    K -- Slices into 16 partitions --> L[12,870 Market Combinations]
    L --> M{Calculate PBO}
    M -- PBO < 5% --> N[Approve Model]
    M -- PBO > 5% --> O[Reject & Retrain]
end
class J,K,L,M,N,O eval;

### 🗺️ Development Roadmap

* [x] **Core Backend:** Homoscedastic Data Factory, Optuna Bayesian Search, and CSCV Validation.
* [x] **Terminal Command Center:** Headless parallel-ready optimization and continuous out-of-sample evaluation.
* [ ] **UI Dashboard:** A local Streamlit / web interface for visual telemetry and execution management (Currently in development).

### 🚀 Quick Start Guide

**1. Clone the Engine**
```bash
git clone [https://github.com/aymo-wh/algo-trading-bot.git](https://github.com/aymo-wh/algo-trading-bot.git)
cd algo-trading-bot
pip install -r requirements.txt
```
**2. Build the Data (The Fuel)
Specify your asset class (Crypto, Macro, or Tech Equities) using the dynamic config argument.
```bash
python data_factory.py --config config/config_crypto.json
```
**3. Train the Brains (The Alpha Search)
Run headless optimization to find the optimal trading rules. The engine will loop through the specified basket automatically.
```bash
python research/optimize_agents.py --config config/config_crypto.json --trials 50 --timesteps 50000
```
**4. Evaluate Performance
Run continuous out-of-sample testing across the entire basket to generate the Gladiator Leaderboard.
```bash
python evaluate_agents.py --config config/config_crypto.json
```
##⚠️ Disclaimer
Not Financial Advice. This repository is an open-source engineering laboratory built for educational and research purposes. Do not deploy this architecture with real capital without fundamentally understanding the underlying stochastic calculus, execution limits, and transaction fee risks.
