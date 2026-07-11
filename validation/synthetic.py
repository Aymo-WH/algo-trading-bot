"""Synthetic panels for referee calibration ONLY (mission §3.3d).

These generators exist so the referee can be proven honest: it must find
NOTHING in the null panel and RECOVER the planted signal. Synthetic data is
never used to report strategy performance.
"""
import numpy as np
import pandas as pd


def _index(n_days: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2005-01-03", periods=n_days)


def null_panel(n_days: int = 2500, n_assets: int = 50, seed: int = 0,
               cov: np.ndarray | None = None, daily_vol: float = 0.012,
               common_factor_share: float = 0.5) -> pd.DataFrame:
    """Zero-edge daily returns with realistic cross-sectional correlation.

    If cov is given (e.g. the empirical pre-holdout covariance) it is used
    directly; otherwise a one-factor structure reproduces the measured
    'PC1 ~ half the variance' property of the real ETF panel.
    """
    rng = np.random.default_rng(seed)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    if cov is not None:
        r = rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=n_days)
        return pd.DataFrame(r, index=_index(n_days), columns=cols[:cov.shape[0]])
    f = rng.normal(0, 1, n_days)
    betas = rng.uniform(0.5, 1.5, n_assets)
    idio = rng.normal(0, 1, (n_days, n_assets))
    s = np.sqrt(common_factor_share)
    r = daily_vol * (s * np.outer(f, betas) + np.sqrt(1 - common_factor_share) * idio)
    return pd.DataFrame(r, index=_index(n_days), columns=cols)


def planted_signal_panel(n_days: int = 2500, n_assets: int = 50, seed: int = 0,
                         ic: float = 0.05, horizon: int = 5,
                         daily_vol: float = 0.012, common_factor_share: float = 0.5):
    """Null panel + a signal that truly predicts the next ``horizon``-day
    CROSS-SECTIONAL relative return with rank IC ≈ ``ic``.

    Returns (returns_panel, signal_panel). signal.loc[t] is known at t and
    predicts demeaned returns over (t, t+horizon]. The alpha is injected into
    the idiosyncratic component only, spread over the horizon, so the market
    factor stays unpredictable (as in the real design).
    """
    rng = np.random.default_rng(seed)
    base = null_panel(n_days, n_assets, seed=seed + 1,
                      daily_vol=daily_vol, common_factor_share=common_factor_share)
    idio_vol = daily_vol * np.sqrt(1 - common_factor_share)
    sig = pd.DataFrame(rng.normal(0, 1, base.shape), index=base.index,
                       columns=base.columns)
    # per-day alpha so that horizon-sum has corr ~ ic with the signal
    lam = ic * idio_vol * np.sqrt(1.0 / horizon) / np.sqrt(max(1 - ic**2, 1e-9))
    alpha = pd.DataFrame(0.0, index=base.index, columns=base.columns)
    for k in range(1, horizon + 1):
        alpha += lam * sig.shift(k).fillna(0.0)
    alpha = alpha.sub(alpha.mean(axis=1), axis=0)      # market component untouched
    return base + alpha, sig
