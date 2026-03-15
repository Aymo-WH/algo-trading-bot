# 🤖 Quantitative AI Trading Architecture: The Meta-Labeling System
**Version:** 8.0 (Institutional Grade)
**Core Methodology:** Marcos López de Prado (*Advances in Financial Machine Learning*)

## 🏛️ Executive Summary
This repository implements an advanced algorithmic trading pipeline moving beyond standard retail technical analysis. The architecture utilizes mathematical stationarity, strict point-in-time orthogonalization, and a dual-agent Reinforcement Learning (RL) meta-labeling system to generate and size trades. It is secured against multiple testing selection bias via Combinatorially Symmetric Cross-Validation (CSCV).

---

## ⚙️ Phase 1: Data Pipeline & Feature Engineering
**Objective:** Provide the neural networks with memory-rich, stationary, and purely independent (orthogonal) mathematical vectors without Look-Ahead Bias.

* **Fractional Differentiation (FFD):** * Standard integer differencing makes data stationary but destroys 100% of the statistical memory.
    * We apply an expanding Fixed-Width Window FFD. The coefficient $d^*$ is optimized via Augmented Dickey-Fuller (ADF) tests to find the minimum differencing required to achieve stationarity ($p < 0.05$) while preserving maximum historical memory.
* **Point-in-Time PCA Orthogonalization:**
    * Highly collinear technical indicators (RSI, MACD, Bollinger Bands, ATR) cause the Substitution Effect.
    * We compress these into 5 strictly independent vectors via Principal Component Analysis (PCA). 
    * *Anti-Leakage Protocol:* The `StandardScaler` and `PCA` transformations are fitted **strictly on the Training timeline (pre-2023)**. Out-of-sample data is transformed using this historical matrix, preventing future volatility from leaking into the training state.

---

## 🧠 Phase 2: The Meta-Agent (Dual Brain System)
**Objective:** Separate the "Side" (Direction) from the "Size" (Conviction) using two independent neural networks.

1. **Primary Signal Generator (DQN - Deep Q-Network):**
    * **Action Space:** Discrete (Long, Short, Hold).
    * **Role:** Analyzes the orthogonalized state to determine market direction. Acts as a strict safety filter. If DQN outputs "Hold", the trade is vetoed entirely.
2. **Secondary Bet-Sizer (PPO - Proximal Policy Optimization):**
    * **Action Space:** Continuous $[-1, 1]$.
    * **Role:** Meta-Labeling. It predicts the probability of the DQN being correct based on the current market state.
3. **The Meta-Labeling Mathematics:**
    * The PPO continuous output is normalized to a raw probability $p \in [0.01, 0.99]$.
    * Calculate the $z$-score: $z = \frac{p - 0.5}{\sqrt{p(1-p)}}$
    * Calculate continuous conviction using the standard Normal CDF: $m = 2Z[z] - 1$
4. **Size Discretization:**
    * The conviction level ($m$) is passed through a 10% ($0.1$) step-function. This mathematically prevents micro-rebalancing jitter and excessive transaction friction.

---

## 🏦 Phase 3: The RL Execution Environment (Trading Gym)
**Objective:** Simulate live trading conditions accurately without generating fictitious returns.

* **The ETF Trick:** * The environment tracks the portfolio as the value of a theoretical $\$1$ invested ($K_t$).
    * Broker fees are explicitly separated ($c_t$) and deducted directly from the cash reserve as a "negative dividend". This prevents short strategies from generating fake profits due to rebalancing costs.
* **Purged & Embargoed Transitions:**
    * The environment enforces a strict **1% Embargo** boundary after Train sets to allow MACD/Volatility memory to decay.
    * Overlapping training labels are **Purged**.
    * The Gym's Markov Decision Process (MDP) is programmed to strictly terminate (`done = True`) if an agent hits these boundaries, ensuring no "learning" occurs across restricted gaps.

---

## 📊 Phase 4: Statistical Validation & CSCV
**Objective:** Mathematically prove the backtest is not overfit due to selection bias under multiple testing.

* **Combinatorially Symmetric Cross-Validation (CSCV):**
    * A $T \times N$ matrix of trial performances (T = timesteps, N = model hyperparameter combinations) is partitioned into $S$ sub-matrices.
    * The system generates $\binom{S}{S/2}$ combinations of Training and Testing sets.
    * The optimal in-sample strategy is ranked against the median out-of-sample alternatives.
* **Probability of Backtest Overfitting (PBO):**
    * A logit is calculated for each combination: $\lambda_c = \log\left[\frac{\bar{\omega}_c}{1 - \bar{\omega}_c}\right]$.
    * The final risk metric is the ratio of logits less than zero: $PBO = \int_{-\infty}^{0} f(\lambda) d\lambda$.
