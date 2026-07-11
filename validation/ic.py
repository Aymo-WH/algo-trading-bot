"""Cross-sectional information-coefficient testing (Grinold-Kahn; Newey-West).

Convention: ``signal`` and ``fwd_returns`` are DataFrames indexed by rebalance
date with one column per asset. ``signal`` at date t must be computable from
data available at/before t; ``fwd_returns`` at date t is the return over
(t, t+horizon]. Alignment is the caller's responsibility; these functions only
measure.
"""
import numpy as np
import pandas as pd

from validation.metrics import newey_west_tstat

MIN_NAMES = 10


def cross_sectional_ic(signal: pd.DataFrame, fwd_returns: pd.DataFrame,
                       min_names: int = MIN_NAMES) -> pd.Series:
    """Per-date Spearman rank IC across assets. NaN where < min_names overlap."""
    sig, fwd = signal.align(fwd_returns, join="inner")
    out = {}
    for dt in sig.index:
        s = sig.loc[dt]
        r = fwd.loc[dt]
        mask = s.notna() & r.notna()
        if mask.sum() < min_names:
            out[dt] = np.nan
            continue
        out[dt] = s[mask].rank().corr(r[mask].rank())
    return pd.Series(out, name="ic")


def ic_summary(ic: pd.Series, nw_lags: int = 2) -> dict:
    """Mean IC, Newey-West t, and per-calendar-year sign consistency."""
    ic = ic.dropna()
    if len(ic) < 10:
        return {"n_dates": len(ic), "mean_ic": np.nan, "t_nw": np.nan,
                "pct_years_positive": np.nan, "years": {}}
    yearly = ic.groupby(ic.index.year).mean()
    return {
        "n_dates": int(len(ic)),
        "mean_ic": float(ic.mean()),
        "std_ic": float(ic.std()),
        "t_nw": float(newey_west_tstat(ic.values, lags=nw_lags)),
        "pct_years_positive": float((yearly > 0).mean()),
        "years": {int(y): float(v) for y, v in yearly.items()},
    }


def passes_graduation(summary: dict, min_ic: float = 0.01, min_t: float = 2.0,
                      min_year_share: float = 0.60) -> bool:
    """The pre-registered Tier-1 graduation rule (specs/DESIGN-v2-2026-07-07.md)."""
    return (not np.isnan(summary["mean_ic"])
            and summary["mean_ic"] >= min_ic
            and summary["t_nw"] >= min_t
            and summary["pct_years_positive"] >= min_year_share)


def ic_decay(signal: pd.DataFrame, prices: pd.DataFrame,
             horizons_days: tuple = (5, 10, 15, 20, 30, 40), min_names: int = MIN_NAMES) -> dict:
    """Mean rank IC at increasing forward horizons (half-life diagnostic)."""
    out = {}
    for h in horizons_days:
        fwd = prices.shift(-h) / prices - 1.0
        fwd = fwd.reindex(signal.index)
        out[h] = float(cross_sectional_ic(signal, fwd, min_names).mean())
    return out
