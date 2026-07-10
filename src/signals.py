"""Tier-1 signal library (design.md §5; frozen construction in
specs/EXP-001-tier1-signal-ic.md). Every function is point-in-time: the
value at date t uses only prices/returns at or before t. Inputs/outputs are
DataFrames indexed by daily trading date, one column per asset; outputs are
NaN for names ineligible at that date (design's PIT eligibility mask).

Not a backtester — these are feature transforms only. Downstream callers
(the EXP-001 runner, later the Phase-3 combiner) restrict to rebalance dates
and apply the graduation rule / portfolio construction.
"""
import warnings

import numpy as np
import pandas as pd

SKIP_DAYS = 21          # ~1 calendar month, momentum's skip-month convention
MOM_LOOKBACKS = (252, 126, 63)    # 12m, 6m, 3m
TREND_LOOKBACKS = (252, 63)       # 12m, 3m
BETA_WINDOW = 252
MIN_SEASONAL_YEARS = 5


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None)


def equal_weight_market_return(ret: pd.DataFrame, elig: pd.DataFrame) -> pd.Series:
    """Cross-sectional mean return of the eligible set, per date (the shared
    market proxy for S3's beta and S4's relative return)."""
    return ret.where(elig).mean(axis=1)


def _cross_sectional_zscore(x: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    """Per-date z-score using only eligible names for the mean/std; NaN
    outside the eligible mask."""
    masked = x.where(elig)
    mu = masked.mean(axis=1)
    sd = masked.std(axis=1)
    return masked.sub(mu, axis=0).div(sd.where(sd > 0), axis=0)


def _blend(zs: list) -> pd.DataFrame:
    """Elementwise mean across same-shape frames, skipping NaN; NaN only
    where ALL inputs are NaN at that cell."""
    stacked = np.stack([z.to_numpy() for z in zs])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
        out = np.nanmean(stacked, axis=0)
    return pd.DataFrame(out, index=zs[0].index, columns=zs[0].columns)


def momentum_s1(close: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    """XS momentum: skip-21d trailing return at 12m/6m/3m, z-scored per date
    over eligible names, equal-blended."""
    zs = []
    for lb in MOM_LOOKBACKS:
        r = close.shift(SKIP_DAYS) / close.shift(SKIP_DAYS + lb) - 1.0
        zs.append(_cross_sectional_zscore(r, elig))
    return _blend(zs)


def trend_s2(close: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    """TS trend: trailing return at 12m/3m (no skip), z-scored per date over
    eligible names, equal-blended."""
    zs = []
    for lb in TREND_LOOKBACKS:
        r = close / close.shift(lb) - 1.0
        zs.append(_cross_sectional_zscore(r, elig))
    return _blend(zs)


def low_beta_s3(close: pd.DataFrame, elig: pd.DataFrame,
               market: pd.Series | None = None) -> pd.DataFrame:
    """BAB: -beta vs the trailing-252d eligible equal-weight market (higher
    S3 = lower beta = BAB-favored). No z-score: single component, and
    Spearman rank IC is invariant to any per-date affine rescale anyway."""
    ret = daily_returns(close)
    m = market if market is not None else equal_weight_market_return(ret, elig)
    cov = ret.rolling(BETA_WINDOW, min_periods=BETA_WINDOW).cov(m)
    var = m.rolling(BETA_WINDOW, min_periods=BETA_WINDOW).var()
    beta = cov.div(var.where(var > 0), axis=0)
    return (-beta).where(elig)


def seasonality_s4(close: pd.DataFrame, elig: pd.DataFrame,
                   market: pd.Series | None = None) -> pd.DataFrame:
    """Calendar seasonality: for each asset and calendar month, the expanding
    mean of PRIOR years' relative return in that month (>=5 prior years
    required); broadcast to every day of the current year's occurrence."""
    ret = daily_returns(close)
    m = market if market is not None else equal_weight_market_return(ret, elig)
    rel = ret.sub(m, axis=0)

    period = pd.PeriodIndex(rel.index, freq="M")
    monthly = rel.groupby(period).mean()          # M(a,k,y): one row per (y,k)

    prior_years_mean = pd.DataFrame(index=monthly.index, columns=monthly.columns,
                                    dtype=float)
    for k in range(1, 13):
        sel = monthly.index.month == k
        sub = monthly.loc[sel].sort_index()
        prior_years_mean.loc[sub.index] = (
            sub.expanding(min_periods=MIN_SEASONAL_YEARS).mean().shift(1))

    daily_period = pd.PeriodIndex(rel.index, freq="M")
    broadcast = prior_years_mean.reindex(daily_period)
    broadcast.index = rel.index
    return broadcast.where(elig)
