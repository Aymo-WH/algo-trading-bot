"""
Volatility-Managed Long Exposure -- rule-based baseline (no ML).

Directional prediction was disproven out-of-sample (see project notes). This
pivots to using the assets' forecastable VOLATILITY rather than their direction.

Hypothesis (Moreira & Muir, 2017): volatility is persistent and forecastable,
while high-vol periods do NOT pay proportionally higher returns. So scaling a
LONG position inversely to forecast vol improves risk-adjusted return (Sharpe)
and cuts drawdown versus static buy-and-hold.

Rule (deterministic, ONE risk parameter -> near-zero overfitting surface):
    w_t = clip( target_vol / forecast_vol_t , 0, vol_cap )
    forecast_vol_t = EWMA realised vol using returns up to t-1 (no look-ahead)
    target_vol     = median EWMA vol on the TRAIN split (no test leakage)
    strategy_return_t = w_t * r_t  -  |w_t - w_{t-1}| * fee

Capital protection: vol_cap = 1.0 (no leverage).

This is a BETA + RISK-MANAGEMENT product, not directional alpha: it captures the
asset's drift with a better Sharpe / shallower drawdown. Success is measured on
Sharpe and max drawdown vs buy-and-hold, NOT on raw ROI.

No new dependencies (numpy / pandas only).
"""

import argparse
import os
import json
import numpy as np
import pandas as pd

TRAIN_DIR = "data/train/"
TEST_DIR = "data/test/"


def ewma_vol(returns, halflife):
    """EWMA volatility (std of returns). Backward-looking by construction."""
    var = returns.pow(2).ewm(halflife=halflife, min_periods=halflife).mean()
    return np.sqrt(var)


def max_drawdown(equity):
    """Max peak-to-trough decline of an equity curve (returns a negative fraction)."""
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min())


def annualized_sharpe(returns, periods_per_year):
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(periods_per_year))


def periods_per_year(test_df):
    """Estimate bars/year from the Date span (dollar bars are not uniform in time)."""
    if 'Date' not in test_df.columns:
        return 2520.0  # ~10 bars/day * 252
    dates = pd.to_datetime(test_df['Date'])
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    return (len(test_df) / years) if years > 0 else 2520.0


def backtest_ticker(ticker, fee, halflife=20, vol_cap=1.0):
    train_path = os.path.join(TRAIN_DIR, f"{ticker}_data.csv")
    test_path = os.path.join(TEST_DIR, f"{ticker}_data.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return None

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if 'Close' not in train.columns or 'Close' not in test.columns or len(test) < halflife + 5:
        return None

    # Target vol from TRAIN only (no test leakage): typical vol -> avg exposure ~1.0
    target_vol = float(ewma_vol(train['Close'].pct_change(), halflife).median())

    r = test['Close'].pct_change().fillna(0.0)
    # Forecast for bar t uses returns up to t-1 (shift) -> strictly no look-ahead
    fvol = ewma_vol(r, halflife).shift(1)
    w = (target_vol / fvol).clip(lower=0.0, upper=vol_cap).fillna(0.0)

    strat_r = w * r
    turnover = w.diff().abs().fillna(w.abs())
    strat_r_net = strat_r - turnover * fee

    ppy = periods_per_year(test)
    eq = (1.0 + strat_r_net).cumprod()
    bh_eq = (1.0 + r).cumprod()

    return {
        "Ticker": ticker,
        "Strat ROI %": (eq.iloc[-1] - 1) * 100,
        "B&H ROI %": (bh_eq.iloc[-1] - 1) * 100,
        "Strat Sharpe": annualized_sharpe(strat_r_net, ppy),
        "B&H Sharpe": annualized_sharpe(r, ppy),
        "Strat MaxDD %": max_drawdown(eq.values) * 100,
        "B&H MaxDD %": max_drawdown(bh_eq.values) * 100,
        "Avg Expo": float(w.mean()),
        "strat_returns": strat_r_net,  # kept for downstream PBO; dropped from the printed table
    }


def main(tickers, fee, halflife=20, vol_cap=1.0):
    print("=" * 110)
    print("VOLATILITY-MANAGED LONG EXPOSURE  (rule-based baseline, no ML)")
    print(f"halflife={halflife}  vol_cap={vol_cap}  fee={fee}  |  success metric = Sharpe & MaxDD vs Buy & Hold")
    print("=" * 110)

    rows = []
    for tk in tickers:
        res = backtest_ticker(tk, fee, halflife, vol_cap)
        if res is None:
            print(f"  (skipped {tk}: data missing or too short)")
            continue
        rows.append(res)

    if not rows:
        print("No results -- check that data/test and data/train exist for the configured tickers.")
        return

    df = pd.DataFrame(rows)
    show = df.drop(columns=["strat_returns"]).copy()
    for c in show.columns:
        if c != "Ticker":
            show[c] = show[c].astype(float).round(2)
    print(show.to_string(index=False))
    print("-" * 110)

    wins = int((df["Strat Sharpe"] > df["B&H Sharpe"]).sum())
    dd_better = int((df["Strat MaxDD %"] > df["B&H MaxDD %"]).sum())  # less negative = shallower
    print(f"Sharpe improved vs B&H on {wins}/{len(df)} assets.")
    print(f"Max drawdown shallower vs B&H on {dd_better}/{len(df)} assets.")
    print(f"Mean Strat Sharpe: {df['Strat Sharpe'].mean():+.2f}   |   Mean B&H Sharpe: {df['B&H Sharpe'].mean():+.2f}")
    print("=" * 110)
    print("Read: vol targeting should LIFT Sharpe and SHALLOW drawdown; raw ROI may trail B&H in")
    print("bull runs (it is de-risked on average). If Sharpe doesn't beat B&H here, the premise fails.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rule-based volatility-managed long exposure backtest")
    parser.add_argument('--config', type=str, default='config/config_phase1.json')
    parser.add_argument('--halflife', type=int, default=20, help="EWMA halflife (bars) for the vol forecast.")
    parser.add_argument('--vol_cap', type=float, default=1.0, help="Max position weight (1.0 = no leverage).")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)
    tickers = cfg.get("tickers", [])
    fee = cfg.get("transaction_fee_percent", 0.0001)

    main(tickers, fee, halflife=args.halflife, vol_cap=args.vol_cap)
