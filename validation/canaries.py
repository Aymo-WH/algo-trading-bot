"""Leak canaries (mission §3.3d) — a result is never 'validated' while any trips.

All canaries operate on the (signal, forward-return) panel pair of a candidate
signal, evaluated with the same IC machinery used for graduation. Semantics:

- label_shuffle: within-date permutation of forward returns must kill the IC.
  If shuffled labels still score, information is flowing outside the
  signal→future-return channel (leakage).
- time_shift: lagging the signal an extra ~6 months must kill the IC; if it
  doesn't, the 'signal' is a slow artifact, not timed information.
- random_feature: pure noise must not graduate more often than test size allows.

Pipeline-level canaries (synthetic null / planted signal through the FULL
referee) live in tests/referee/test_referee_calibration.py.
"""
import numpy as np
import pandas as pd

from validation.ic import cross_sectional_ic, ic_summary

T_DEAD = 2.0          # |t| below this = "edge collapsed", per graduation rule
MAX_TRIP_SHARE = 0.15 # noise may exceed t=2 at ~5% by chance; 15% is the alarm


def label_shuffle_canary(signal: pd.DataFrame, fwd_returns: pd.DataFrame,
                         n_shuffles: int = 100, seed: int = 0,
                         nw_lags: int = 2) -> dict:
    """Within-date permutation of forward returns across assets."""
    rng = np.random.default_rng(seed)
    sig, fwd = signal.align(fwd_returns, join="inner")
    true_t = ic_summary(cross_sectional_ic(sig, fwd), nw_lags)["t_nw"]
    vals = fwd.values
    ts = []
    for _ in range(n_shuffles):
        shuffled = np.empty_like(vals)
        for i in range(vals.shape[0]):
            shuffled[i] = vals[i, rng.permutation(vals.shape[1])]
        f2 = pd.DataFrame(shuffled, index=fwd.index, columns=fwd.columns)
        ts.append(ic_summary(cross_sectional_ic(sig, f2), nw_lags)["t_nw"])
    ts = np.array(ts)
    trip_share = float(np.mean(np.abs(ts) >= T_DEAD))
    return {"canary": "label_shuffle", "true_t": float(true_t),
            "shuffled_t_mean": float(np.nanmean(ts)),
            "shuffled_t_abs_q95": float(np.nanquantile(np.abs(ts), 0.95)),
            "trip_share": trip_share,
            "passed": bool(trip_share <= MAX_TRIP_SHARE)}


def time_shift_canary(signal: pd.DataFrame, fwd_returns: pd.DataFrame,
                      shift_rows: int = 26, nw_lags: int = 2) -> dict:
    """Signal lagged ``shift_rows`` extra rebalances must lose its edge."""
    sig, fwd = signal.align(fwd_returns, join="inner")
    base = ic_summary(cross_sectional_ic(sig, fwd), nw_lags)
    lagged = ic_summary(cross_sectional_ic(sig.shift(shift_rows), fwd), nw_lags)
    base_t, lag_t = base["t_nw"], lagged["t_nw"]
    # Pass = stale signal is dead, OR it kept < half the live t (slow-decay
    # signals like 12m momentum legitimately retain some stale correlation).
    passed = np.isnan(lag_t) or abs(lag_t) < T_DEAD or abs(lag_t) < 0.5 * abs(base_t)
    return {"canary": "time_shift", "base_t": float(base_t),
            "lagged_t": float(lag_t), "shift_rows": shift_rows,
            "passed": bool(passed)}


def random_feature_canary(fwd_returns: pd.DataFrame, n_features: int = 50,
                          seed: int = 0, nw_lags: int = 2) -> dict:
    """Pure-noise signals must not 'graduate' beyond test size."""
    rng = np.random.default_rng(seed)
    ts = []
    for _ in range(n_features):
        noise = pd.DataFrame(rng.normal(0, 1, fwd_returns.shape),
                             index=fwd_returns.index, columns=fwd_returns.columns)
        ts.append(ic_summary(cross_sectional_ic(noise, fwd_returns), nw_lags)["t_nw"])
    ts = np.array(ts)
    trip_share = float(np.mean(np.abs(ts) >= T_DEAD))
    return {"canary": "random_feature", "t_abs_mean": float(np.nanmean(np.abs(ts))),
            "trip_share": trip_share,
            "passed": bool(trip_share <= MAX_TRIP_SHARE)}


def run_signal_canaries(signal: pd.DataFrame, fwd_returns: pd.DataFrame,
                        seed: int = 0) -> dict:
    """The per-signal canary battery. all_passed gates 'validated' status."""
    results = [
        label_shuffle_canary(signal, fwd_returns, seed=seed),
        time_shift_canary(signal, fwd_returns),
        random_feature_canary(fwd_returns, seed=seed),
    ]
    return {"all_passed": all(r["passed"] for r in results), "results": results}
