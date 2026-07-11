"""Sanity proofs for the Sharpe/PSR/DSR machinery against known behavior."""
import numpy as np
import pytest

from validation.metrics import (annualized_sharpe, dsr, dsr_from_ledger,
                                effective_trials, expected_max_sharpe,
                                newey_west_tstat, psr, sharpe)

RNG = np.random.default_rng(42)


def test_sharpe_known_value():
    r = np.array([0.01, -0.01] * 500)
    assert abs(sharpe(r)) < 1e-12  # zero mean => zero sharpe
    r2 = np.full(100, 0.01) + RNG.normal(0, 0.001, 100)
    assert sharpe(r2) > 5  # tiny noise around positive mean


def test_psr_at_reference_is_half():
    # SR exactly equal to the reference => P(true SR > ref) = 0.5
    assert psr(0.1, 0.1, 1000) == pytest.approx(0.5)


def test_psr_increases_with_sample_length():
    assert psr(0.1, 0.0, 2000) > psr(0.1, 0.0, 200) > 0.5


def test_psr_penalizes_fat_tails_and_negative_skew():
    base = psr(0.1, 0.0, 500, skew=0.0, kurt=3.0)
    assert psr(0.1, 0.0, 500, skew=-1.0, kurt=3.0) < base
    assert psr(0.1, 0.0, 500, skew=0.0, kurt=10.0) < base


def test_expected_max_sharpe_grows_with_trials():
    v = 0.02
    e10, e100, e1000 = (expected_max_sharpe(n, v) for n in (10, 100, 1000))
    assert 0 < e10 < e100 < e1000
    assert expected_max_sharpe(1, v) == 0.0  # single trial: no deflation


def test_effective_trials_bounds():
    assert effective_trials(100, 0.0) == 100    # independent
    assert effective_trials(100, 1.0) == 1.0    # perfectly correlated
    assert 1 < effective_trials(100, 0.5) < 100


def test_dsr_deflates_relative_to_psr():
    # same observed SR: DSR under many trials must be below PSR under one trial
    sr_hat, T = 0.15, 1000
    many = dsr(sr_hat, trial_sr_variance=0.01, n_trials=200, n_obs=T)
    one = dsr(sr_hat, trial_sr_variance=0.01, n_trials=1, n_obs=T)
    assert many < one


def test_dsr_selection_bias_calibration():
    """The core property: picking the best of N noise strategies must NOT pass DSR.

    Generate N pure-noise return series, select the best Sharpe, compute DSR with
    the full trial ledger — it should not be significant at 95%.
    """
    T, N = 750, 60
    trials = RNG.normal(0, 0.01, size=(N, T))
    sharpes = np.array([sharpe(t) for t in trials])
    best = trials[np.argmax(sharpes)]
    corr = np.corrcoef(trials)
    mean_corr = float(np.abs(corr[np.triu_indices(N, 1)]).mean())
    out = dsr_from_ledger(best, sharpes, mean_corr)
    assert out["dsr"] < 0.95, f"noise passed DSR: {out}"


def test_dsr_real_signal_passes():
    """And a genuinely strong signal with few trials must pass."""
    T = 2000
    good = RNG.normal(0.002, 0.01, T)  # per-period SR ~0.2, T large
    sharpes = np.array([sharpe(good)] + [sharpe(RNG.normal(0, 0.01, T)) for _ in range(4)])
    out = dsr_from_ledger(good, sharpes, 0.0)
    assert out["dsr"] > 0.95, f"real signal failed DSR: {out}"


def test_newey_west_matches_plain_t_on_iid():
    x = RNG.normal(0.5, 1.0, 5000)
    plain = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    nw = newey_west_tstat(x, lags=5)
    assert nw == pytest.approx(plain, rel=0.05)


def test_newey_west_shrinks_t_under_autocorrelation():
    # AR(1) with strong positive autocorrelation: NW t must be well below plain t
    n, phi = 5000, 0.8
    e = RNG.normal(0, 1, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    x += 0.1
    plain = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    nw = newey_west_tstat(x, lags=20)
    assert nw < 0.6 * plain


def test_annualized_sharpe_scaling():
    r = RNG.normal(0.001, 0.01, 10000)
    assert annualized_sharpe(r, 252) == pytest.approx(sharpe(r) * np.sqrt(252))
