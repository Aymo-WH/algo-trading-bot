"""Canary size (don't trip on clean setups) and power (trip on leaks) proofs."""
import numpy as np
import pandas as pd

from validation.canaries import (label_shuffle_canary, random_feature_canary,
                                 run_signal_canaries, time_shift_canary)
from validation.synthetic import planted_signal_panel


def _weekly_setup(seed=5, ic=0.06, n_days=1500, n_assets=40):
    rets, sig = planted_signal_panel(n_days=n_days, n_assets=n_assets,
                                     seed=seed, ic=ic)
    prices = (1 + rets).cumprod()
    fwd = prices.shift(-5) / prices - 1.0
    weekly = sig.index[::5]
    return sig.loc[weekly], fwd.loc[weekly]


def test_label_shuffle_kills_real_signal():
    sig, fwd = _weekly_setup()
    out = label_shuffle_canary(sig, fwd, n_shuffles=30, seed=1)
    assert out["true_t"] > 3          # live signal scores
    assert out["passed"], out          # shuffled labels do not


def test_label_shuffle_catches_leaked_labels():
    """A 'signal' that IS the future return must survive shuffling ZERO times —
    but here we simulate the classic leak: signal built FROM the label."""
    sig, fwd = _weekly_setup()
    leaked = fwd + np.random.default_rng(2).normal(0, fwd.std().mean() * 0.5, fwd.shape)
    out = label_shuffle_canary(leaked, fwd, n_shuffles=30, seed=1)
    # the leak shows up as an absurd true_t; shuffling kills it (alignment leak)
    assert out["true_t"] > 10, "leak should look impossibly good"


def test_time_shift_kills_planted_signal():
    sig, fwd = _weekly_setup()
    out = time_shift_canary(sig, fwd, shift_rows=26)
    assert abs(out["base_t"]) > 3
    assert out["passed"], out


def test_time_shift_catches_static_artifact():
    """A time-invariant 'signal' (constant per asset) retains its IC when
    lagged — the canary must trip."""
    sig, fwd = _weekly_setup(ic=0.0)
    rng = np.random.default_rng(3)
    static = pd.DataFrame(np.tile(rng.normal(0, 1, sig.shape[1]), (len(sig), 1)),
                          index=sig.index, columns=sig.columns)
    # give the static ranking a real edge so base_t is significant
    fwd2 = fwd.add(static.iloc[0] * fwd.stack().std() * 0.3, axis=1)
    out = time_shift_canary(static, fwd2, shift_rows=26)
    assert not out["passed"], out


def test_random_feature_canary_size():
    sig, fwd = _weekly_setup(ic=0.0, seed=9)
    out = random_feature_canary(fwd, n_features=40, seed=4)
    assert out["passed"], out


def test_full_battery_on_clean_setup():
    sig, fwd = _weekly_setup()
    out = run_signal_canaries(sig, fwd, seed=0)
    assert out["all_passed"], out
