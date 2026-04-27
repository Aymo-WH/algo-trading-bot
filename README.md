# The Gordian Project (Version 9.0 Cloud-Ready)

This repository implements an algorithmic trading pipeline dubbed **"The Gordian Project"**, deeply rooted in Marcos López de Prado's advanced financial machine learning methodologies.

Traditional algorithmic trading relies on chronological time bars, but time is a poor metric for market activity. This engine discards the clock, instead sampling the market based on capital exchange, learning the underlying physics of different asset classes, and executing mathematically sized trades.

**Note:** UI development (Gradio/Streamlit) has been intentionally abandoned. This project strictly focuses on backend perfection, algorithmic integrity, and live terminal execution.

---

## 🧠 Core Engineering Features

*   **Information-Driven Dollar Bars:** Neutralizes market noise and heteroscedasticity by sampling bars only when a dynamic threshold of fiat currency is exchanged.
*   **Fractional Differentiation (FFD) & Point-in-Time PCA:** Achieves stationarity on price series while preserving memory, and extracts orthogonal features dynamically without look-ahead bias.
*   **Dual-Agent Meta-Labeling (Dual-Brain System):** Strictly separates "Direction" from "Conviction."
    *   A primary **Deep Q-Network (DQN)** model generates directional signals (-1, -0.5, 0, 0.5, 1).
    *   A secondary **Proximal Policy Optimization (PPO)** agent acts as the risk manager, continuously sizing the bet based on statistical confidence (converting PPO output to a z-score and applying a Normal CDF).
*   **Optimal Trading Rule (OTR) & Dynamic Rolling O-U Barriers:** Execution barrier optimization (Profit-Taking and Stop-Loss limits) is handled offline by estimating Ornstein-Uhlenbeck (O-U) parameters to prevent execution-level overfitting.
*   **Probability of Backtest Overfitting (PBO):** Calculated via Combinatorially Symmetric Cross-Validation (CSCV) to strictly validate out-of-sample performance and penalize overfitting.
*   **Multi-Asset Modular Configuration:** Dynamic configuration handling for multiple asset classes via isolated files (e.g., `config/config_crypto.json`, `config/config_macro.json`).

---

## 🏗️ Architecture V1.5 Upgrades

*   **Hybrid Vectorization:** Transitioned to a 1D state-tracking loop paired with NumPy reducing functions for highly performant Dollar Bar construction.
*   **Data Leakage Prevention:** Enforced a strict 60-Period Train/Test Embargo to eliminate predictive look-ahead bias and isolate the out-of-sample datasets.
*   **Optimized Execution Barriers:** Utilized 3D Tensor broadcasting for calculating the Profit-Taking and Stop-Loss grids, maximizing localized grid search speeds.
*   **Cloud Execution Safeguards:** Implemented memory optimization techniques (removing DataFrame copies) and absolute path resolution to ensure strict RunPod serverless stability.

---

## 📂 Architecture & Core Modules

The project follows a strict modular architecture, isolating core engine logic from laboratory tools.

*   **`data_factory.py`**: The data pipeline. Fetches a rolling 730-day window of intraday data, compresses it into Information-Driven Dollar Bars, applies point-in-time PCA and FFD, and exports fitted mathematical matrices (`scaler.pkl`, `pca.pkl`) to `models/matrices/`. It completely wipes old data directories to prevent cross-asset pollution.
*   **`core/trading_gym.py`**: Contains `TradingEnv`, a highly optimized OpenAI Gym environment. Utilizes an $O(1)$ ring buffer (`collections.deque`) for historical observations and enforces strict data validation to ensure ultra-fast `step()` and `reset()` execution.
*   **`core/meta_agent.py`**: Combines the primary DQN model and secondary PPO model using Meta-Labeling mathematics to generate final, sized trade actions.
*   **`core/optimize_barriers.py`**: Offline engine to evaluate optimal dynamic execution barriers by estimating O-U parameters and conducting a localized grid search to maximize the Sharpe Ratio.
*   **`core/pbo_validator.py`**: Computes the PBO via CSCV, employing safety measures (like epsilon injection) to dynamically prevent division-by-zero errors.

---

## 🚀 Quick Start Guide

### 1. Clone & Setup
```bash
git clone https://github.com/aymo-wh/the-gordian-project.git
cd the-gordian-project
pip install -r requirements.txt
```

### 2. Build the Data Factory (The Fuel)
Specify your asset class (e.g., Crypto, Macro ETFs) using the dynamic config argument. This process generates datasets and fitted PCA/Scaler matrices.
```bash
python data_factory.py --config config/config_phase1.json
```

### 3. Agent Training (`train_agent.py`)
Provides core utilities (`train_dqn`, `train_ppo`) to programmatically initialize `TradingEnv` for specific tickers and train reinforcement learning agents using custom hyperparameter configurations.

### 4. Headless Optimization (`research/optimize_agents.py`)
Run headless hyperparameter optimization using Optuna. The engine loops through the specified basket of active tickers, tracking out-of-sample returns to build a True PBO matrix.
```bash
python research/optimize_agents.py --config config/config_phase1.json --trials 50 --timesteps 50000
```

### 5. Out-of-Sample Evaluation & Telemetry (`evaluate_agents.py` & `telemetry.py`)
Evaluate models strictly on fixed chronological blocks to ensure validity. Analyze decoupled agent telemetry (Confusion Matrix, Recall, Precision, Log-Loss) and execution latency metrics.
```bash
python evaluate_agents.py --config config/config_phase1.json
python telemetry.py
```

### 6. Live Inference Engine (`live_inference.py`)
The core terminal execution engine. Loads Stable-Baselines3 agents alongside ticker-specific state matrices for live, real-time execution simulation.
```bash
python live_inference.py --config config/config_phase1.json
```

---

## ⚠️ Disclaimer
**Not Financial Advice.** This repository is an open-source engineering laboratory built strictly for educational and research purposes. Do not deploy this architecture with real capital without fundamentally understanding the underlying stochastic calculus, execution limits, and transaction fee risks.
