"""Unit tests for src/decomposition.py (EXP-003 construction). Expected values
are computed by hand for small synthetic panels (not by re-calling the
module's own helpers), matching tests/fast/test_signals.py's convention.
Point-in-time is asserted literally: truncating the future must not change
values already computed for the past.
"""
import numpy as np
import pandas as pd

from src.decomposition import MIN_PRIOR_OBS, expanding_static_timing


def _bdays(start, n):
    return pd.bdate_range(start, periods=n)


def test_default_min_prior_obs_matches_frozen_spec():
    assert MIN_PRIOR_OBS == 52


def test_basic_expanding_mean_no_gaps():
    idx = _bdays("2020-01-01", 6)
    sig = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=idx)
    static, timing = expanding_static_timing(sig, min_prior_obs=2)

    # rows 0-1: fewer than 2 strictly-prior observations -> undefined
    assert static["A"].iloc[0:2].isna().all()
    assert timing["A"].iloc[0:2].isna().all()

    # row 2 (value=3.0): prior obs {1.0, 2.0} -> static=1.5, timing=1.5
    assert np.isclose(static["A"].iloc[2], 1.5)
    assert np.isclose(timing["A"].iloc[2], 3.0 - 1.5)
    # row 3 (value=4.0): prior obs {1.0, 2.0, 3.0} -> static=2.0
    assert np.isclose(static["A"].iloc[3], 2.0)
    assert np.isclose(timing["A"].iloc[3], 4.0 - 2.0)
    # row 5 (value=6.0): prior obs {1.0..5.0} -> static=3.0
    assert np.isclose(static["A"].iloc[5], 3.0)
    assert np.isclose(timing["A"].iloc[5], 6.0 - 3.0)


def test_nan_gaps_excluded_from_count_and_sum_not_treated_as_zero():
    idx = _bdays("2020-01-01", 6)
    sig = pd.DataFrame({"B": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0]}, index=idx)
    static, timing = expanding_static_timing(sig, min_prior_obs=2)

    # row2 (v=3.0): only ONE strictly-prior non-NaN obs (row0=1.0; row1 is
    # NaN and must NOT count) -> still undefined at min_prior_obs=2
    assert np.isnan(static["B"].iloc[2])
    # row3 (v=4.0): prior non-NaN = {1.0, 3.0} -> count=2 -> static=2.0
    assert np.isclose(static["B"].iloc[3], 2.0)
    assert np.isclose(timing["B"].iloc[3], 4.0 - 2.0)
    # row4 (v=5.0): prior non-NaN = {1.0, 3.0, 4.0} -> static=8/3
    assert np.isclose(static["B"].iloc[4], 8.0 / 3.0)
    assert np.isclose(timing["B"].iloc[4], 5.0 - 8.0 / 3.0)
    # row5 (v=6.0): prior non-NaN = {1.0, 3.0, 4.0, 5.0} -> static=3.25
    assert np.isclose(static["B"].iloc[5], 3.25)


def test_static_can_be_defined_when_current_signal_is_nan():
    """Static depends only on PRIOR values, so it can be defined even at a
    date where the signal itself is currently missing (e.g. a temporary
    eligibility gap) -- and timing must then be NaN (signal - defined
    static, with signal NaN), not silently treated as zero deviation."""
    idx = _bdays("2020-01-01", 5)
    sig = pd.DataFrame({"C": [1.0, 2.0, 3.0, np.nan, 5.0]}, index=idx)
    static, timing = expanding_static_timing(sig, min_prior_obs=2)

    # row3 (v=NaN): prior non-NaN = {1.0, 2.0, 3.0} -> static=2.0, DEFINED
    assert np.isclose(static["C"].iloc[3], 2.0)
    assert np.isnan(timing["C"].iloc[3])
    # row4 (v=5.0): prior non-NaN = {1.0, 2.0, 3.0} (row3's NaN doesn't
    # count) -> static unchanged at 2.0
    assert np.isclose(static["C"].iloc[4], 2.0)
    assert np.isclose(timing["C"].iloc[4], 5.0 - 2.0)


def test_boundary_exact_min_prior_obs():
    """At exactly min_prior_obs strictly-prior observations, static becomes
    defined -- not one observation earlier, not one later."""
    idx = _bdays("2020-01-01", 5)
    sig = pd.DataFrame({"A": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=idx)
    static, _ = expanding_static_timing(sig, min_prior_obs=3)
    assert static["A"].iloc[:3].isna().all()          # 0,1,2 prior obs -> NaN
    assert not np.isnan(static["A"].iloc[3])           # 3 prior obs -> defined
    assert np.isclose(static["A"].iloc[3], 20.0)       # mean(10,20,30)


def test_columns_are_independent():
    idx = _bdays("2020-01-01", 6)
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b = [1.0, np.nan, 3.0, 4.0, 5.0, 6.0]
    both = pd.DataFrame({"A": a, "B": b}, index=idx)
    static_both, timing_both = expanding_static_timing(both, min_prior_obs=2)
    static_a, timing_a = expanding_static_timing(both[["A"]], min_prior_obs=2)
    static_b, timing_b = expanding_static_timing(both[["B"]], min_prior_obs=2)
    pd.testing.assert_series_equal(static_both["A"], static_a["A"])
    pd.testing.assert_series_equal(timing_both["A"], timing_a["A"])
    pd.testing.assert_series_equal(static_both["B"], static_b["B"])
    pd.testing.assert_series_equal(timing_both["B"], timing_b["B"])


def test_pit_truncation_invariant():
    n = 120
    idx = _bdays("2015-01-01", n)
    rng = np.random.default_rng(7)
    sig = pd.DataFrame(rng.normal(0, 1, (n, 3)), index=idx, columns=list("ABC"))
    # punch a few eligibility-style gaps in one column
    sig.iloc[10:15, 1] = np.nan

    static_full, timing_full = expanding_static_timing(sig, min_prior_obs=20)
    cut = n - 30
    static_part, timing_part = expanding_static_timing(sig.iloc[:cut], min_prior_obs=20)
    pd.testing.assert_frame_equal(static_full.iloc[:cut], static_part)
    pd.testing.assert_frame_equal(timing_full.iloc[:cut], timing_part)


def test_timing_plus_static_recovers_signal_where_defined():
    idx = _bdays("2020-01-01", 8)
    rng = np.random.default_rng(9)
    sig = pd.DataFrame(rng.normal(0, 1, (8, 2)), index=idx, columns=["A", "B"])
    static, timing = expanding_static_timing(sig, min_prior_obs=3)
    reconstructed = static + timing
    mask = static.notna()
    pd.testing.assert_frame_equal(reconstructed.where(mask), sig.where(mask))
