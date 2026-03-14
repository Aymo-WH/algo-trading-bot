# algo-trading-bot
# 📈 Quantitative Trading Lab: Progress & Architecture Ledger

## 🧠 System Architecture (Current State)
* **Environment:** Custom OpenAI Gymnasium (`trading_gym.py`)
* **Data Ingestion:** Yahoo Finance API (10-Year Timeline: 2016-2026)
* **Core Algorithms:** Reinforcement Learning (Stable-Baselines3)
    * PPO (Continuous Action Space)
    * DQN (Discrete Action Space)
* **Execution Speed:** Vectorized environments (`make_vec_env`) for parallel CPU training.
* **Reward Mechanism:** Sortino-style Risk-Adjusted returns (penalizing downside volatility) with integrated broker fee accounting.

## 🛠️ Phase 1: The Infrastructure Firefight (Status: IN PROGRESS)
* **Issue:** Agent developed "cowardice" (refusing to trade) due to heavy downside penalties, and the testing suite collapsed due to `sys.modules` mocking conflicts.
* **Resolution:** Softened downside penalty to 1.5x, added a slight hold-cash penalty, and purged broken mock-tests to strictly maintain the Bankruptcy Physics test.

## 🔬 Phase 2: Fractional Differentiation (Status: INITIATED)
* **Theory:** Marcos López de Prado (Advances in Financial Machine Learning).
* **Objective:** Standard returns destroy asset memory. Raw prices are non-stationary. We are implementing the Fixed-Width Window Fracdiff (FFD) method to find the optimal differentiation fraction ($d^*$) where the Augmented Dickey-Fuller (ADF) test achieves stationarity ($p < 0.05$) while preserving maximum historical memory.

## 🧮 Phase 3: Orthogonalization via PCA (Status: PENDING)
* **Objective:** Eliminate the "Substitution Effect." Highly correlated technical indicators (RSI, MACD, Bollinger Bands) confuse the AI's feature importance mapping.
* **Action:** Pass the fractionally differentiated feature set through Principal Component Analysis (PCA) to compress the data into purely uncorrelated mathematical eigenvectors before feeding it to the neural network.

## 🤖 Phase 4: Meta-Labeling Architecture (Status: PENDING)
* **Objective:** Separate Signal Generation (Direction) from Bet Sizing (Conviction).
* **Action:** Restructure the AI setup into a dual-agent system:
    1. **Primary Model (DQN):** Decides strictly whether the market is going Long or Short.
    2. **Secondary Model (PPO):** Acts as the Meta-Labeler. It takes the DQN's prediction and decides *how much capital* to allocate based on the probability of the DQN being correct.

## 📊 Phase 5: Deflated Sharpe Ratio (Status: PENDING)
* **Objective:** Prevent Backtest Overfitting (Selection Bias under Multiple Testing).
* **Action:** Implement a hidden ledger to log the Sharpe Ratios of all failed training iterations. Calculate the Deflated Sharpe Ratio (DSR) to mathematically prove the final model's out-of-sample performance is statistically significant against the variance of our failed trials.
