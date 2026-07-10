"""Unit tests for src/signals.py (EXP-001 constructions). Expected values are
computed independently in each test (not by re-calling the module's private
helpers) so these are real checks, not tautologies. Point-in-time is asserted
literally: truncating the future must not change values already computed for
the past (same style as tests/data/test_panel_integrity.py).
"""
import numpy as np
import pandas as pd
import pytest

from src.signals import (BETA_WINDOW, MIN_SEASONAL_YEARS, MOM_LOOKBACKS,
                         SKIP_DAYS, TREND_LOOKBACKS, _blend,
                         _cross_sectional_zscore, equal_weight_market_return,
                         low_beta_s3, momentum_s1, seasonality_s4, trend_s2)


def _bdays(start, n):
    return pd.bdate_range(start, periods=n)


# --------------------------- helpers ---------------------------

def test_cross_sectional_zscore_uses_eligible_only():
    idx = _bdays("2020-01-01", 3)
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 4.0, 6.0],
                      "C": [100.0, 100.0, 100.0]}, index=idx)
    elig = pd.DataFrame({"A": True, "B": True, "C": False}, index=idx)
    z = _cross_sectional_zscore(x, elig)
    assert z["C"].isna().all()                      # ineligible -> NaN always
    row0 = x.loc[idx[0], ["A", "B"]]
    mu, sd = row0.mean(), row0.std()
    assert np.isclose(z.loc[idx[0], "A"], (1.0 - mu) / sd)
    assert np.isclose(z.loc[idx[0], "B"], (2.0 - mu) / sd)


def test_blend_nan_only_when_all_inputs_nan():
    idx = _bdays("2020-01-01", 1)
    a = pd.DataFrame({"X": [1.0], "Y": [np.nan]}, index=idx)
    b = pd.DataFrame({"X": [3.0], "Y": [np.nan]}, index=idx)
    c = pd.DataFrame({"X": [np.nan], "Y": [np.nan]}, index=idx)
    out = _blend([a, b, c])
    assert np.isclose(out.loc[idx[0], "X"], 2.0)     # mean(1,3), skipping the NaN
    assert np.isnan(out.loc[idx[0], "Y"])             # all 3 NaN -> NaN


# --------------------------- S1 momentum ---------------------------

def test_momentum_s1_skip_and_lookback_formula():
    n = max(MOM_LOOKBACKS) + SKIP_DAYS + 5
    idx = _bdays("2015-01-01", n)
    rng = np.random.default_rng(0)
    # independent per-asset daily returns -> deterministic once seeded
    rets = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=idx, columns=list("ABC"))
    close = (1 + rets).cumprod()
    elig = pd.DataFrame(True, index=idx, columns=list("ABC"))

    out = momentum_s1(close, elig)
    t = idx[-1]
    # independently recompute via raw numpy/pandas, not the module's helpers
    manual_r = {}
    for lb in MOM_LOOKBACKS:
        p_skip = close.iloc[-1 - SKIP_DAYS]
        p_skip_lb = close.iloc[-1 - SKIP_DAYS - lb]
        manual_r[lb] = p_skip / p_skip_lb - 1.0
    zs = []
    for lb in MOM_LOOKBACKS:
        r = manual_r[lb]
        zs.append((r - r.mean()) / r.std())
    expected = pd.concat(zs, axis=1).mean(axis=1)
    pd.testing.assert_series_equal(out.loc[t].sort_index(), expected.sort_index(),
                                   check_names=False, atol=1e-10)


def test_momentum_s1_pit_truncation_invariant():
    n = max(MOM_LOOKBACKS) + SKIP_DAYS + 40
    idx = _bdays("2015-01-01", n)
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.01, (n, 4)), index=idx, columns=list("ABCD"))
    close = (1 + rets).cumprod()
    elig = pd.DataFrame(True, index=idx, columns=list("ABCD"))
    full = momentum_s1(close, elig)
    cut = n - 20
    part = momentum_s1(close.iloc[:cut], elig.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], part)


def test_momentum_s1_all_lookbacks_undefined_is_nan_not_all_eligible():
    n = 10
    idx = _bdays("2020-01-01", n)
    rets = pd.DataFrame(0.001, index=idx, columns=["A"])
    close = (1 + rets).cumprod()
    elig = pd.DataFrame(True, index=idx, columns=["A"])
    out = momentum_s1(close, elig)               # far too short for any lookback
    assert out["A"].isna().all()


# --------------------------- S2 trend (no skip) ---------------------------

def test_trend_s2_has_no_skip_month_unlike_s1():
    """A sharp reversal in the skip window must move S2 (no skip) but leave
    S1 (skip=21) unaffected -- proves the skip is actually wired in."""
    n = max(MOM_LOOKBACKS + TREND_LOOKBACKS) + SKIP_DAYS + 5
    idx = _bdays("2015-01-01", n)
    rets = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    rets.loc[:, "B"] = 0.0005                      # steady comparator
    rets.iloc[-1 - SKIP_DAYS // 2, 0] = 0.20        # sharp jump INSIDE the skip window
    close = (1 + rets).cumprod()
    elig = pd.DataFrame(True, index=idx, columns=["A", "B"])

    s1 = momentum_s1(close, elig).loc[idx[-1]]
    s2 = trend_s2(close, elig).loc[idx[-1]]
    close_no_jump = close.copy()
    # rebuild without the jump to isolate its effect
    rets_flat = rets.copy()
    rets_flat.iloc[-1 - SKIP_DAYS // 2, 0] = 0.0
    close_flat = (1 + rets_flat).cumprod()
    s1_flat = momentum_s1(close_flat, elig).loc[idx[-1]]
    s2_flat = trend_s2(close_flat, elig).loc[idx[-1]]

    assert np.isclose(s1["A"], s1_flat["A"], atol=1e-9), "S1 should ignore the skip-window jump"
    assert not np.isclose(s2["A"], s2_flat["A"], atol=1e-9), "S2 should reflect the jump (no skip)"


def test_trend_s2_pit_truncation_invariant():
    n = max(TREND_LOOKBACKS) + 40
    idx = _bdays("2015-01-01", n)
    rng = np.random.default_rng(2)
    rets = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=idx, columns=list("ABC"))
    close = (1 + rets).cumprod()
    elig = pd.DataFrame(True, index=idx, columns=list("ABC"))
    full = trend_s2(close, elig)
    cut = n - 15
    part = trend_s2(close.iloc[:cut], elig.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], part)


# --------------------------- S3 low-beta ---------------------------

def test_low_beta_s3_recovers_known_beta_exactly():
    n = BETA_WINDOW + 10
    idx = _bdays("2015-01-01", n)
    rng = np.random.default_rng(3)
    base = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=idx, columns=list("XYZ"))
    elig_base = pd.DataFrame(True, index=idx, columns=list("XYZ"))
    m = equal_weight_market_return(base.pct_change(fill_method=None) * 0 + base, elig_base)
    # m above double-counts pct_change; recompute market cleanly from prices:
    close_base = (1 + base).cumprod()
    ret_base = close_base.pct_change(fill_method=None)
    m = equal_weight_market_return(ret_base, elig_base)

    ret_full = ret_base.copy()
    ret_full["A_beta2"] = 2.0 * m                 # exact beta = 2 by construction
    ret_full["B_beta_half"] = 0.5 * m             # exact beta = 0.5
    ret_full = ret_full.fillna(0.0)               # day-0 NaN -> 0 return for cumprod
    close = (1 + ret_full).cumprod()
    elig = pd.DataFrame(True, index=idx, columns=close.columns)

    # Pass the intended 3-asset market explicitly: letting low_beta_s3
    # recompute it over ALL 5 eligible columns would fold A/B themselves
    # into the market they were built against (self-referential, wrong).
    out = low_beta_s3(close, elig, market=m)
    assert out["A_beta2"].iloc[:BETA_WINDOW].isna().all()
    assert out["A_beta2"].iloc[BETA_WINDOW:].notna().all()
    tail_a = out["A_beta2"].iloc[BETA_WINDOW:]
    tail_b = out["B_beta_half"].iloc[BETA_WINDOW:]
    assert np.allclose(tail_a, -2.0, atol=1e-6), tail_a.tail()
    assert np.allclose(tail_b, -0.5, atol=1e-6), tail_b.tail()


def test_low_beta_s3_pit_truncation_invariant():
    n = BETA_WINDOW + 30
    idx = _bdays("2015-01-01", n)
    rng = np.random.default_rng(4)
    rets = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=idx, columns=list("ABC"))
    close = (1 + rets).cumprod()
    elig = pd.DataFrame(True, index=idx, columns=list("ABC"))
    full = low_beta_s3(close, elig)
    cut = n - 10
    part = low_beta_s3(close.iloc[:cut], elig.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], part)


# --------------------------- S4 seasonality ---------------------------

def _monthly_pattern_panel(n_years=8):
    """One asset, business daily. Every January day in year-index y (0-based)
    gets a distinct constant relative-return contribution `y`; other months
    are exactly 0. Market is passed explicitly as a flat zero series (not
    recomputed from these columns) so rel == raw return with no
    self-contamination from A's own huge synthetic moves -- with only 1-2
    columns, equal_weight_market_return would average A into its own
    baseline at 50% weight instead of the ~1.4% it gets at real (70-name)
    scale; that is a test-isolation concern, not something seasonality_s4
    needs to guard against itself (S3's beta test hit the identical issue)."""
    full_idx = pd.bdate_range("2010-01-01", f"{2010 + n_years - 1}-12-31")
    ret_a = pd.Series(0.0, index=full_idx)
    for y_offset in range(n_years):
        year = 2010 + y_offset
        jan_mask = (full_idx.year == year) & (full_idx.month == 1)
        ret_a.loc[jan_mask] = float(y_offset)
    rets = pd.DataFrame({"A": ret_a}, index=full_idx)
    close = (1 + rets).cumprod()
    elig = pd.DataFrame(True, index=full_idx, columns=["A"])
    market = pd.Series(0.0, index=full_idx)
    return close, elig, full_idx, market


def test_seasonality_s4_requires_five_strictly_prior_years():
    close, elig, idx, market = _monthly_pattern_panel(n_years=8)
    out = seasonality_s4(close, elig, market=market)
    for y_offset in range(8):
        year = 2010 + y_offset
        jan_days = idx[(idx.year == year) & (idx.month == 1)]
        val = out.loc[jan_days, "A"]
        if y_offset < MIN_SEASONAL_YEARS:            # < 5 strictly-prior years
            assert val.isna().all(), f"year {year} should be NaN (only {y_offset} priors)"
        else:
            expected = np.mean(range(y_offset))       # mean of 0..y_offset-1
            assert np.allclose(val, expected, atol=1e-10), (year, val.iloc[0], expected)


def test_seasonality_s4_current_year_never_included():
    """Year index 5's assigned Jan value must be mean(0,1,2,3,4)=2.0, NOT
    shifted to include its own y=5 contribution (would give a different mean)."""
    close, elig, idx, market = _monthly_pattern_panel(n_years=8)
    out = seasonality_s4(close, elig, market=market)
    jan5 = idx[(idx.year == 2015) & (idx.month == 1)]   # y_offset=5
    assert np.allclose(out.loc[jan5, "A"], 2.0, atol=1e-10)


def test_seasonality_s4_other_months_unaffected():
    close, elig, idx, market = _monthly_pattern_panel(n_years=8)
    out = seasonality_s4(close, elig, market=market)
    feb = idx[(idx.year == 2017) & (idx.month == 2)]
    assert np.allclose(out.loc[feb, "A"], 0.0, atol=1e-10)


def test_seasonality_s4_pit_truncation_invariant():
    close, elig, idx, market = _monthly_pattern_panel(n_years=8)
    full = seasonality_s4(close, elig, market=market)
    cut = len(idx) - 100
    part = seasonality_s4(close.iloc[:cut], elig.iloc[:cut], market=market.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], part)


# --------------------------- eligibility masking (all signals) ---------------------------

def test_all_signals_nan_outside_eligibility():
    n = BETA_WINDOW + 20
    idx = _bdays("2015-01-01", n)
    rng = np.random.default_rng(5)
    rets = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=idx, columns=list("ABC"))
    close = (1 + rets).cumprod()
    elig = pd.DataFrame(True, index=idx, columns=list("ABC"))
    elig["C"] = False                              # C never eligible
    for fn in (momentum_s1, trend_s2, low_beta_s3, seasonality_s4):
        out = fn(close, elig)
        assert out["C"].isna().all(), fn.__name__
