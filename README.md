# The Gordian Project: Dual-Engine Algorithmic Trading

This repository implements an institutional-grade algorithmic trading pipeline dubbed "The Gordian Project", deeply rooted in Marcos López de Prado's advanced financial machine learning methodologies.  The V2 architecture evolves the project from a research sandbox into a bifurcated production system: a high-compute Research Node (Cloud) and a low-latency Execution Node (Edge).  

---

## 🧠 Core Quantitative Features
The project maintains its foundational commitment to Information-Driven Finance:
- **Information-Driven Dollar Bars:** Discards chronological time to neutralize market noise and heteroscedasticity. Sampling occurs only when a dynamic threshold of fiat currency is exchanged, restoring statistical normality to price series.
- **Fractional Differentiation (FFD):** Achieves stationarity while preserving maximum memory. The engine iteratively solves for an optimal $d$-value to ensure the series passes ADF tests without destroying the predictive signals found in historical price levels.
- **Point-in-Time PCA:** Prevents collinearity by dynamically extracting orthogonal features from the microstructural feature set without introducing look-ahead bias.
- **Microstructural Dynamics:** Captures informed trading probability and toxic order flow using **VPIN** (Volume-Synchronized Probability of Informed Trading), **SADF** (Supremum Augmented Dickey-Fuller) for bubble detection, and **Amihud Illiquidity** metrics.
- **Dual-Agent Meta-Labeling:** Strictly separates "Direction" from "Conviction."
   - **Primary Brain (XGBoost):** Utilizes the Triple-Barrier Method to generate directional signals (LONG/SHORT/HOLD).
   - **Secondary Agent (PPO):** Acts as the risk manager, dynamically sizing the bet based on statistical confidence.  

---

## 🏗️ Architecture V2 Upgrades

*   **Bifurcated Data Engine:** Implements a dual-pathway ingestion system. Traditional equities are sourced via yfinance, while high-volatility digital assets are ingested via CCXT/Binance to ensure volume fidelity for Dollar Bar construction.
*   **Edge-Inference Protocol:** Optimizes the live execution loop for local hardware. The execution node fetches real-time 60-day windows and applies pre-fitted Scaler and PCA matrices to bypass environment-based "Shape Mismatches".
*   **Data Leakage Prevention:** Enforces a strict 60-Period Train/Test Embargo to eliminate predictive look-ahead bias and ensure the validity of out-of-sample performance.
*   **Automated Execution Loop:** Fully integrated with the Binance Testnet API for real-time market order execution with automated receipt ID tracking.

*   **Microstructural Features:** Added VPIN, SADF, and Amihud Illiquidity to better capture market microstructure dynamics and informed trading probability.
*   **Safe RL Reward Function:** Implemented Turnover, Variance, and Drawdown penalties to strictly penalize excessive trading, volatility, and portfolio drawdown.

*   **Hybrid Vectorization:** Transitioned to a 1D state-tracking loop paired with NumPy reducing functions for highly performant Dollar Bar construction.
*   **Data Leakage Prevention:** Enforced a strict 60-Period Train/Test Embargo to eliminate predictive look-ahead bias and isolate the out-of-sample datasets.
*   **Optimized Execution Barriers:** Utilized 3D Tensor broadcasting for calculating the Profit-Taking and Stop-Loss grids, maximizing localized grid search speeds.
*   **Cloud Execution Safeguards:** Implemented memory optimization techniques (removing DataFrame copies) and absolute path resolution to ensure strict RunPod serverless stability.

---

## 📂 Architecture & Core Modules

The project follows a strict modular architecture, isolating core engine logic from laboratory tools.

*   **`data_factory.py`**: The pipeline engine. Fetches a rolling 730-day window, constructs Dollar Bars, calibrates FFD, and exports fitted mathematical matrices to models/matrices/.
*   (`scaler.pkl`, `pca.pkl`) to `models/matrices/`. It completely wipes old data directories to prevent cross-asset pollution.
*   **`core/trading_gym.py`**: Contains `TradingEnv`, a highly optimized OpenAI Gym environment. Utilizes an $O(1)$ ring buffer (`collections.deque`) for historical observations and enforces strict data validation to ensure ultra-fast `step()` and `reset()` execution.
*   **`core/meta_agent.py`**: Combines the primary XGBoost model and secondary PPO model using Meta-Labeling mathematics to generate final, sized trade actions.
*   **`core/optimize_barriers.py`**: Offline engine to evaluate optimal dynamic execution barriers by estimating O-U parameters and conducting a localized grid search to maximize the Sharpe Ratio.
*   **`core/pbo_validator.py`**: Computes the PBO via CSCV, employing safety measures (like epsilon injection) to dynamically prevent division-by-zero errors.
*   **`src/live_inference.py`**: The core execution engine. A lean, production-ready script that loads pre-trained brains and asset-specific matrices for real-time market action.

---

## 🚀 Quick Start Guide

### 1. Clone & Setup
```bash
git clone https://github.com/aymo-wh/the-gordian-project.git
cd the-gordian-project
pip install -r requirements.txt
```

### 2. Local Configuration: 
Ensure your .env file contains your BINANCE_API_KEY, BINANCE_SECRET, and LIVE_TRADING="TRUE".

### 3. Build the Data Factory (The Fuel)
Specify your asset class (e.g., Crypto, Macro ETFs) using the dynamic config argument. This process generates datasets and fitted PCA/Scaler matrices.
```bash
python data_factory.py --config config/config_phase1.json
```

### 4. Agent Training (`train_agent.py`)
Provides core utilities (`train_xgb`, `train_ppo`) to programmatically initialize `TradingEnv` for specific tickers and train reinforcement learning agents using custom hyperparameter configurations.

### 5. Headless Optimization (`research/optimize_agents.py`)
Run headless hyperparameter optimization using Optuna. The engine loops through the specified basket of active tickers, tracking out-of-sample returns to build a True PBO matrix.
```bash
python research/optimize_agents.py --config config/config_phase1.json --trials 50 --timesteps 50000
```

### 6. Out-of-Sample Evaluation & Telemetry (`evaluate_agents.py` & `telemetry.py`)
Evaluate models strictly on fixed chronological blocks to ensure validity. Analyze decoupled agent telemetry (Confusion Matrix, Recall, Precision, Log-Loss) and execution latency metrics.
```bash
python evaluate_agents.py --config config/config_phase1.json
python telemetry.py
```

### 7. Live Inference Engine (`live_inference.py`)
The core terminal execution engine. Loads Stable-Baselines3 agents alongside ticker-specific state matrices for live, real-time execution simulation.
```bash
python live_inference.py --config config/config_phase1.json
```

---

## ⚠️ Disclaimer
**Not Financial Advice.** This repository is an open-source engineering laboratory built strictly for educational and research purposes. Do not deploy this architecture with real capital without fundamentally understanding the underlying stochastic calculus, execution limits, and transaction fee risks.
