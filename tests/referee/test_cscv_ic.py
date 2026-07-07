"""CSCV and IC machinery calibration proofs."""
import numpy as np
import pandas as pd
import pytest

from validation.cscv import cscv_pbo
from validation.ic import cross_sectional_ic, ic_summary, passes_graduation
from validation.synthetic import null_panel, planted_signal_panel

RNG = np.random.default_rng(7)


def test_cscv_pure_noise_pbo_near_half():
    M = RNG.normal(0, 0.01, size=(640, 40))
    out = cscv_pbo(M, S=16)
    assert out["n_combinations"] == 12870
    assert 0.35 <= out["pbo"] <= 0.65, out


def test_cscv_one_true_strategy_pbo_low():
    M = RNG.normal(0, 0.01, size=(640, 40))
    M[:, 7] += 0.004                       # one genuinely superior config
    out = cscv_pbo(M, S=16)
    assert out["pbo"] <= 0.10, out


def test_cscv_input_validation():
    with pytest.raises(ValueError):
        cscv_pbo(np.zeros((10, 5)), S=16)  # T < 2S
    with pytest.raises(ValueError):
        cscv_pbo(np.zeros((100, 1)), S=16)  # single column


def test_ic_recovers_planted_signal_magnitude():
    rets, sig = planted_signal_panel(n_days=2000, n_assets=50, seed=3, ic=0.05)
    prices = (1 + rets).cumprod()
    fwd = prices.shift(-5) / prices - 1.0
    weekly = sig.index[::5]
    summary = ic_summary(cross_sectional_ic(sig.loc[weekly], fwd.loc[weekly]))
    assert 0.02 <= summary["mean_ic"] <= 0.09, summary
    assert summary["t_nw"] > 3
    assert passes_graduation(summary)


def test_ic_finds_nothing_in_null_panel():
    rets = null_panel(n_days=2000, n_assets=50, seed=11)
    noise_sig = pd.DataFrame(RNG.normal(0, 1, rets.shape),
                             index=rets.index, columns=rets.columns)
    prices = (1 + rets).cumprod()
    fwd = prices.shift(-5) / prices - 1.0
    weekly = noise_sig.index[::5]
    summary = ic_summary(cross_sectional_ic(noise_sig.loc[weekly], fwd.loc[weekly]))
    assert abs(summary["mean_ic"]) < 0.02
    assert not passes_graduation(summary)


def test_ic_ignores_dates_with_too_few_names():
    sig = pd.DataFrame(np.nan, index=pd.bdate_range("2020-01-01", periods=20),
                       columns=[f"A{i}" for i in range(12)])
    sig.iloc[:, :5] = 1.0                  # only 5 names -> below MIN_NAMES
    fwd = sig * 0.0
    ic = cross_sectional_ic(sig, fwd)
    assert ic.isna().all()
