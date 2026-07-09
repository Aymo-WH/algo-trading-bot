# The Gordian Project — Directional Alpha Research (Archived)

> **Status: PAUSED / research archive (June 2026).** The directional thesis did
> not clear out-of-sample validation. The engineering infrastructure is sound and
> reusable; the *alpha hypothesis* (microstructure/price features predicting
> short-horizon **direction**) is disproven on this universe. Work on
> volatility-based strategies continues in the successor project,
> **The Vol-Gordian Project**.

**Recent engineering note (3 Jul 2026):** train_agent.py previously read the host's CPU count inside the training container (e.g. 128 cores on a 4/16-vCPU pod), spawning ~127 PPO workers and ~1900 numeric-library threads. Fixed via os.sched_getaffinity and by pinning OMP/MKL/OpenBLAS/NUMEXPR/NUMBA thread pools to 1 per worker process. Infrastructure fix only; does not change research findings below.

This repository implements an institutional-style algorithmic-trading research
pipeline rooted in Marcos López de Prado's financial-machine-learning methods.
Its goal was a dual-agent meta-labeling system: an **XGBoost** classifier for
trade *direction* and a **PPO** agent for *bet sizing*, validated against
overfitting via PBO/CSCV.

It is published as an honest engineering record — including what worked, what
didn't, and why the directional approach was set aside.

---

## 📉 Findings & Status (the honest result)

The pipeline was run end-to-end on hourly dollar bars across 7 assets (sector
ETFs + crypto majors + TQQQ/VXX), with a purged train/test split.

- **The direction signal has no out-of-sample edge.** Across three independent
  experiments the out-of-sample 15-bar win rate never cleared ~51% (a coin flip):

  | Experiment | OOS 15-bar net EV | OOS win rate |
  |---|---|---|
  | 4 microstructural features, random split | −0.19% | 50.8% |
  | + purged temporal split + regularization | −1.12% | 49.2% |
  | + price/momentum/vol/FFD enrichment (11 features) | −0.94% | 42.7% |

- **In-sample edge was largely a leakage artifact.** As the leaky random
  `train_test_split` was replaced with a purged, embargoed temporal split, the
  *in-sample* win rate fell from 63.6% → 53.7%. There was far less real signal
  than the original metrics suggested.

- **The overfitting gate confirmed it.** PBO via CSCV returned **0.65** (FAIL):
  the in-sample-best configuration does not survive out-of-sample recombination.
  Run-to-run results swung from roughly −5% to +5% summed ROI on identical
  settings — the fingerprint of a noise-dominated signal.

- **Diagnosis.** The microstructural features (VPIN, Amihud, Kyle's λ, SADF) are
  documented predictors of **volatility and informed-flow regimes**, *not*
  direction. Short-horizon directional prediction is the hardest problem in
  quant and, on this universe, it was not learnable with these inputs.

**What this is not:** a failure of engineering. The data pipeline, validation
gate, and RL sizing harness all work correctly — they are precisely what let us
*prove* the signal wasn't there before any capital was risked.

---

## 🧠 Core Quantitative Infrastructure (sound & reusable)

- **Information-Driven Dollar Bars** — sample on dollar volume, not clock time, to
  neutralize heteroscedasticity and restore statistical normality.
- **Fractional Differentiation (FFD)** — stationarity with maximum memory; solves
  for the smallest `d` that passes the ADF test.
- **Point-in-Time PCA** — orthogonalizes the feature set with the scaler/PCA fit
  on the training split only (no look-ahead). Saved per asset for inference.
- **Triple-Barrier labels** — profit-take / stop-loss / vertical-time labeling for
  the directional classifier.
- **Dual-Agent Meta-Labeling** — XGBoost (direction) → PPO (bet size in `[0,1]`);
  live action = `xgb_signal × bet_size`.
- **PBO via CSCV** — the overfitting gate (`core/pbo_validator.py` +
  `validate_pbo.py`) that ultimately failed the directional system.

---

## 📂 Module Map

- **`src/data_factory.py`** — data pipeline. Fetches a rolling 730-day window
  (yfinance for equities, CCXT/Binance for crypto), builds Dollar Bars,
  microstructural + price features, FFD, point-in-time PCA, train/test split with
  a 60-bar embargo. Exports per-asset scaler/PCA matrices to `models/matrices/`.
- **`src/train_agent.py`** — `--model {xgb,ppo}`. XGBoost uses a purged per-ticker
  temporal split (train/validation/calibration) with regularization + early
  stopping; PPO trains the sizing agent in `TradingEnv`.
- **`src/core/trading_gym.py`** — `TradingEnv`: action/observation spaces, reward,
  and Triple-Barrier enforcement (can force-liquidate the position).
- **`src/core/meta_agent.py`** — inference wrapper combining XGBoost → PPO.
- **`src/core/optimize_barriers.py`** — Ornstein-Uhlenbeck barrier optimization
  (dynamic PT/SL multipliers).
- **`src/core/pbo_validator.py`** — PBO via CSCV.
- **`src/validate_pbo.py`** — runnable overfitting gate (threshold-grid proxy;
  swap in a seed/hyperparameter ensemble for the rigorous version).
- **`src/evaluate_agents.py`** — out-of-sample evaluation vs Buy-and-Hold and S&P 500.
- **`src/telemetry.py`** — per-agent diagnostics. **`src/live_inference.py`** —
  Binance-testnet execution skeleton (XGBoost only; **not** validated — see Caveats).

---

## 🚀 Reproducing the Research

```bash
git clone https://github.com/Aymo-WH/algo-trading-bot.git
cd algo-trading-bot
pip install -r requirement-training.txt

# 1. Build data (Dollar Bars, features, FFD, PCA matrices)
python src/data_factory.py --config config/config_phase1.json

# 2. Train the direction classifier, then the PPO sizer
python src/train_agent.py --model xgb
python src/train_agent.py --model ppo --timesteps 300000

# 3. Out-of-sample evaluation
python src/evaluate_agents.py --config config/config_phase1.json

# 4. Overfitting gate (PBO via CSCV) — the deployment gate
python src/validate_pbo.py --config config/config_phase1.json
```

The notebook `TheGordian.ipynb` (kept locally; not tracked) runs the same
pipeline cell-by-cell with explanations.

---

## ⚠️ Caveats & Known Limitations (for anyone reusing this)

- **Long-only in practice.** `act = xgb_signal × bet_size` only opens longs; a
  short signal can exit a long but never open a short. ~40% of directional signals
  produce no P&L.
- **"Safe RL" reward is scaffolded, not active.** The turnover/variance/CVaR
  penalty terms in `TradingEnv.step()` exist but their coefficients are **0** — the
  reward is `daily_return × 100` + a 50% bonus on positive returns. The docs/paper
  describe the intended risk-adjusted reward; it was never validated.
- **`live_inference.py` is unvalidated.** It uses XGBoost directly (PPO not wired
  in) and computes only the 4 microstructural features on raw time bars — a
  train/serve mismatch vs the dollar-bar pipeline. Do not treat it as a live path.
- **PBO is a v1 proxy.** `validate_pbo.py` uses a confidence-threshold grid on a
  single ticker. The rigorous seed/hyperparameter ensemble was not run.

---

## ➡️ Successor: The Vol-Gordian Project

Gordian is *hard by design* — directional alpha is the hardest problem in the
field. The natural next step uses the same engine for what the features actually
predict — **volatility** — via volatility-managed exposure (size a position
inversely to forecast vol; target risk-adjusted return rather than directional
alpha). That work lives in a separate repository, **The Vol-Gordian Project**.

---

## ⚠️ Disclaimer
**Not Financial Advice.** This repository is an open-source engineering laboratory
built strictly for educational and research purposes. The directional strategy
herein did **not** pass out-of-sample validation and must not be deployed with
real capital.
