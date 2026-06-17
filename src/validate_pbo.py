"""
Probability of Backtest Overfitting (PBO) gate via Combinatorially Symmetric
Cross-Validation (CSCV).

This wires the orphaned `core.pbo_validator.PBOValidator` into a runnable gate.

Method
------
CSCV needs a performance matrix M of shape (T timesteps, N trials), where each
COLUMN is one competing configuration's per-step return series over a COMMON
timeline. PBOValidator partitions T into S blocks, forms all C(S, S/2) IS/OOS
recombinations, finds the IS-best trial, and measures how often it ranks below
the OOS median. PBO is that fraction.

Trial definition (v1: threshold grid)
--------------------------------------
We hold the single trained XGB+PPO fixed and synthesise N competing strategies
cheaply by varying the XGBoost confidence threshold tau: a trade is only taken
when xgb_prob > tau, otherwise bet size is forced to 0. Each tau yields one OOS
return series -> one column of M.

This is a PROXY: it measures sensitivity to the confidence-threshold choice, not
full hyperparameter selection bias. The rigorous version (one trained PPO variant
per column, different seeds/hyperparams) drops into `build_trial_columns` later
for RunPod-scale compute. Labelled honestly in the output.

No new dependencies (numpy/pandas/matplotlib already used elsewhere).
"""

import argparse
import os
import sys
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trading_gym import TradingEnv
from core.utils import load_agent
from core.pbo_validator import PBOValidator

warnings.filterwarnings("ignore")

DATA_DIR = "data/test/"
MODELS_DIR = "models/"
INITIAL_CAPITAL = 10000.0


def run_threshold_trial(df, xgb_path, ppo_model, tau, config_path):
    """
    Runs the fixed XGB+PPO over the FULL test series, taking a trade only when
    xgb_prob (obs[1]) exceeds tau. Returns the per-step portfolio return series.
    """
    env = TradingEnv(df=df, is_discrete=False, xgb_model_path=xgb_path, config_path=config_path)
    obs, _ = env.reset(options={'start_step': 0})
    # Run the whole series as one episode (defeat the 90-bar episode cap; the
    # end-of-data / circuit-breaker conditions in step() still terminate it).
    env.episode_length = len(df) + 2

    returns = []
    prev_pv = INITIAL_CAPITAL
    done = False
    while not done:
        if obs[1] > tau:
            action, _ = ppo_model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
        else:
            action = np.array([0.0], dtype=np.float32)  # below confidence -> flat

        obs, _reward, terminated, truncated, info = env.step(action)
        pv = info.get('portfolio_value', prev_pv)
        returns.append((pv - prev_pv) / prev_pv if prev_pv > 0 else 0.0)
        prev_pv = pv
        done = terminated or truncated

    return np.asarray(returns, dtype=np.float64)


def build_trial_columns(df, xgb_path, ppo_model, n_trials, config_path):
    """
    Builds N return-series columns. v1 = confidence-threshold grid.

    To swap in the rigorous ensemble later: replace this body with a loop that
    loads/trains N PPO variants and runs each over `df` unfiltered.
    """
    # Predicted-class probability for a calibrated 3-class model sits ~0.33-0.66.
    taus = np.linspace(0.33, 0.60, n_trials)
    columns = []
    for i, tau in enumerate(taus):
        rets = run_threshold_trial(df, xgb_path, ppo_model, tau, config_path)
        columns.append(rets)
        nonzero = int(np.count_nonzero(rets))
        print(f"  trial {i+1:>2}/{n_trials}  tau={tau:.3f}  steps={len(rets)}  active_steps={nonzero}")
    return columns, taus


def pick_ticker(ticker):
    """Resolve the test CSV: explicit ticker, else the one with the most bars."""
    if ticker:
        path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
        if not os.path.exists(path):
            print(f"[ERROR] Test data for {ticker} not found at {path}.")
            return None, None
        return ticker, path

    candidates = glob.glob(os.path.join(DATA_DIR, "*_data.csv"))
    if not candidates:
        print(f"[ERROR] No test data found in {DATA_DIR}.")
        return None, None

    best_path = max(candidates, key=lambda p: len(pd.read_csv(p)))
    best_ticker = os.path.basename(best_path).replace("_data.csv", "")
    return best_ticker, best_path


def main(ticker=None, n_trials=28, partitions=16, config_path='config/config_phase1.json'):
    print("=" * 88)
    print("PROBABILITY OF BACKTEST OVERFITTING (PBO) via CSCV")
    print("=" * 88)
    print("Trial definition: confidence-threshold grid (v1 proxy). See module docstring.")

    xgb_path = os.path.join(MODELS_DIR, "xgb_trading_bot.pkl")
    ppo_path = os.path.join(MODELS_DIR, "ppo_trading_bot.zip")
    if not os.path.exists(xgb_path) or not os.path.exists(ppo_path):
        print("[ERROR] Required models (xgb_trading_bot.pkl, ppo_trading_bot.zip) not found in models/.")
        return

    ppo_model = load_agent(ppo_path)

    resolved_ticker, data_path = pick_ticker(ticker)
    if resolved_ticker is None:
        return

    df = pd.read_csv(data_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    print(f"Timeline: {resolved_ticker}  ({len(df)} OOS bars)  |  trials N={n_trials}  partitions S={partitions}")

    columns, taus = build_trial_columns(df, xgb_path, ppo_model, n_trials, config_path)

    # Common timeline: truncate every column to the shortest (rectangular matrix).
    T = min(len(c) for c in columns)
    if any(len(c) != T for c in columns):
        print(f"[WARN] Unequal trial lengths; truncating all to T={T} (a trial hit the circuit breaker early).")
    M = np.column_stack([c[:T] for c in columns])  # (T, N)

    if T < partitions * 2:
        print(f"[ERROR] Timeline too short (T={T}) for S={partitions} partitions. Use a longer-history ticker.")
        return

    validator = PBOValidator(M, num_partitions=partitions)
    pbo, logits = validator.calculate_pbo()

    print("-" * 88)
    print(f"PBO = {pbo:.4f}   ({pbo*100:.1f}% probability the in-sample-best config is overfit)")
    median_logit = float(np.median(logits))
    print(f"Median logit = {median_logit:+.4f}  (positive => IS-best tends to stay above OOS median)")
    if pbo < 0.10:
        verdict = "STRONG: low overfitting probability."
    elif pbo < 0.50:
        verdict = "ACCEPTABLE: overfitting probability below chance, but not negligible."
    else:
        verdict = "FAIL: in-sample edge does not survive OOS recombination. Do NOT deploy."
    print(f"Verdict: {verdict}")
    print("NOTE: v1 proxy (threshold grid). Confirm with the seed/hyperparameter ensemble on RunPod before any live path.")
    print("=" * 88)

    out_png = "pbo_logit_distribution.png"
    plt.figure(figsize=(10, 6))
    plt.hist(logits, bins=50, color='crimson', alpha=0.7)
    plt.axvline(0.0, color='black', linestyle='--', linewidth=1)
    plt.title(f'CSCV Logit Distribution — {resolved_ticker} (PBO={pbo:.3f})')
    plt.xlabel('Logit  (< 0 => IS-best below OOS median)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(out_png)
    plt.close()
    print(f"Saved logit distribution to {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PBO/CSCV overfitting gate")
    parser.add_argument('--config', type=str, default='config/config_phase1.json',
                        help="Config whose transaction fee the env should use.")
    parser.add_argument('--ticker', type=str, default=None,
                        help="Test ticker to use as the timeline. Default: the one with the most bars.")
    parser.add_argument('--trials', type=int, default=28, help="Number of threshold trials (matrix columns N).")
    parser.add_argument('--partitions', type=int, default=16, help="CSCV partitions S (default 16 -> 12,870 combos).")
    args = parser.parse_args()

    main(ticker=args.ticker, n_trials=args.trials, partitions=args.partitions, config_path=args.config)
