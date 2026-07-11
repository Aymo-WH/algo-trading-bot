"""Vectorized panel backtester — the referee's only performance engine.

Causality contract: ``weights`` are TARGET portfolio weights decided at date t
using data available at/before t's close. They are applied from the NEXT bar:
the position earns ``asset_returns`` starting at t+1 (enforced by shift(1) —
never hand this function pre-shifted weights).

Costs: per-side proportional costs on turnover |Δw|, charged when trading.

Ledger discipline (§3.4): ``backtest_panel`` is the pure engine and does not
log; the logging entry points are ``run_and_log`` here and
``validation.run_battery.run_battery`` — every strategy configuration
evaluated on REAL data must flow through one of those, so nothing runs
uncounted. (Referee self-tests on synthetic panels use throwaway ledgers.)
"""
import numpy as np
import pandas as pd

from validation.ledger import log_trial
from validation.metrics import annualized_sharpe

TRADING_DAYS = 252


def backtest_panel(weights: pd.DataFrame, asset_returns: pd.DataFrame,
                   cost_bps_per_side: float = 5.0) -> pd.DataFrame:
    """Daily portfolio gross/net returns from target weights and asset returns."""
    idx = asset_returns.index
    w = weights.reindex(idx).ffill().reindex(columns=asset_returns.columns).fillna(0.0)
    held = w.shift(1).fillna(0.0)                      # causality: earn from t+1
    gross = (held * asset_returns.fillna(0.0)).sum(axis=1)
    turnover = (w - held).abs().sum(axis=1)            # one-way, both legs
    costs = turnover * cost_bps_per_side / 1e4
    net = gross - costs
    return pd.DataFrame({"gross": gross, "net": net,
                         "turnover": turnover, "costs": costs})


def performance_summary(portfolio: pd.DataFrame,
                        periods_per_year: int = TRADING_DAYS) -> dict:
    """Contract metrics (gross vs net side by side, per §2)."""
    out = {}
    for leg in ("gross", "net"):
        r = portfolio[leg]
        eq = (1 + r).cumprod()
        dd = (eq / eq.cummax() - 1.0).min()
        yearly = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
        out[leg] = {
            "ann_return": float((1 + r).prod() ** (periods_per_year / max(len(r), 1)) - 1),
            "ann_sharpe": float(annualized_sharpe(r.values, periods_per_year)),
            "max_drawdown": float(dd),
            "pct_years_positive": float((yearly > 0).mean()),
            "yearly_returns": {int(y): float(v) for y, v in yearly.items()},
        }
    out["ann_turnover"] = float(portfolio["turnover"].mean() * periods_per_year)
    out["total_cost_drag_ann"] = float(portfolio["costs"].mean() * periods_per_year)
    return out


def run_and_log(*, experiment_id: str, config: dict, weights: pd.DataFrame,
                asset_returns: pd.DataFrame, seed: int | None = None,
                cost_grid=(5.0, 10.0, 20.0), notes: str | None = None) -> dict:
    """Backtest at the pre-registered cost grid and LOG THE TRIAL. Main entry point."""
    results = {}
    for bps in cost_grid:
        port = backtest_panel(weights, asset_returns, cost_bps_per_side=bps)
        results[f"{bps:g}bps"] = performance_summary(port)
    metrics = {"net_sharpe_5bps": results.get("5bps", {}).get("net", {}).get("ann_sharpe"),
               "net_sharpe_10bps": results.get("10bps", {}).get("net", {}).get("ann_sharpe"),
               "full": results}
    log_trial(experiment_id=experiment_id, phase="trial", config=config,
              metrics={k: v for k, v in metrics.items() if k != "full"},
              seed=seed, notes=notes)
    return results
