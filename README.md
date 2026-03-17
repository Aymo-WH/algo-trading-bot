# 🤖 Quantitative AI Trading Architecture: Information-Driven Meta-Labeling
**Version:** 9.0 (Cloud-Ready Institutional Grade)
**Core Methodology:** Marcos López de Prado (*Advances in Financial Machine Learning*)

## 🏛️ Executive Summary
This repository implements an advanced algorithmic trading pipeline moving beyond standard retail technical analysis. By utilizing Information-Driven Dollar Bars, Point-in-Time PCA, and a dual-agent Reinforcement Learning (RL) meta-labeling framework, the system captures non-linear market dynamics safely.

## ⚙️ Core Quantitative Infrastructure

### 1. Information-Driven Dollar Bars
Traditional time bars (OHLCV) suffer from severe heteroscedasticity—oversampling low-activity periods and undersampling high-activity events. This pipeline constructs **Dollar Bars**, sampling the market only when a dynamic threshold of fiat currency ($M$) is exchanged. By compressing 1-hour intraday proxy data into these Information-Driven events, the neural networks analyze market microstructure as a function of actual capital flow, resulting in returns that approximate an IID Normal distribution.

### 2. Fractional Differentiation (FFD) & Stationarity
Standard integer differencing (e.g., day-over-day returns) achieves stationarity but destroys up to 100% of the dataset's statistical memory. We apply a **Fixed-Width Window Fractional Differentiation (FFD)**. The algorithm utilizes an optimized, C-level `np.convolve` routine to find the minimum differencing coefficient ($d^*$) required to pass the Augmented Dickey-Fuller (ADF) test ($p < 0.05$) while preserving maximum historical memory.

### 3. Point-in-Time Orthogonalization (PCA)
Highly collinear technical indicators (RSI, MACD, Bollinger Bands, ATR) cause the Substitution Effect in machine learning models. We compress these into strictly independent vectors via Principal Component Analysis (PCA). 
* **Anti-Leakage Protocol:** The `StandardScaler` and `PCA` transformations are fitted strictly on the Training set timeline. Out-of-sample data is transformed using this historical matrix, mathematically preventing Look-Ahead Bias.

### 4. The Meta-Labeling Framework (DQN + PPO)
We deploy a dual-network Reinforcement Learning architecture to separate directional conviction from bet sizing:
* **Primary Model (DQN - Deep Q-Network):** Acts as the primary signal generator, outputting discrete directional actions (Long, Short, Hold).
* **Secondary Model (PPO - Proximal Policy Optimization):** Acts as the meta-labeler, predicting the probability ($p$) that the DQN's signal is profitable in the current regime. 
* **Dynamic Sizing:** The PPO's continuous output is scaled into a conviction bet size ($m$) using the standard Normal Cumulative Distribution Function (CDF).

### 5. Optimal Trading Rule (OTR) via Stochastic Calculus
To prevent execution-level overfitting, Stop-Loss and Profit-Taking barriers are strictly separated from the neural network's hyperparameter search. We model the asset's price dynamics as a discrete **Ornstein-Uhlenbeck (O-U) process**. Using a rolling 60-day window, the pipeline estimates the speed of mean reversion ($\varphi$) and volatility ($\sigma$), generating 10,000 synthetic futures paths to calibrate optimal point-in-time execution limits.

### 6. CSCV & Probability of Backtest Overfitting (PBO)
The optimization engine uses Optuna (Randomized Search) to hunt for alpha. The out-of-sample chronological paths of every tested configuration are saved into a $T \times N$ matrix. This matrix is evaluated using **Combinatorially Symmetric Cross-Validation (CSCV)** to output a strict Probability of Backtest Overfitting (PBO) metric, mathematically proving whether the selected model is a true discovery or a statistical mirage.

---

## 🚀 Execution & Cloud Compute

The optimization engine is built for headless GPU execution on cloud infrastructure (RunPod, Lambda Labs, GCP).

### Installation
Ensure you install the GPU-enabled dependencies:
```bash
pip install -r requirements.txt
