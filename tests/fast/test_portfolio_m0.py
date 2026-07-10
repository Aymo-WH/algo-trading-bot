"""Unit tests for src/portfolio_m0.py (EXP-002 constructions). Expected values
are computed independently in each test (not by re-calling the module's own
helpers), mirroring tests/fast/test_signals.py's convention. Point-in-time is
asserted literally: truncating the future must not change weights already
computed for the past.
"""
import numpy as np
import pandas as pd
import pytest

from src.portfolio_m0 import (CATEGORY_CAP_FRAC, GROSS, POSITION_CAP_FRAC,
                              _freeze_small_changes, apply_no_trade_band,
                              build_capped_weights, compute_signal_and_weights,
                              neutralize_against_beta)


def _bdays(start, n):
    return pd.bdate_range(start, periods=n)


# --------------------- neutralize_against_beta ---------------------

def test_neutralize_against_beta_exact_identity():
    idx = _bdays("2020-01-01", 2)
    cols = list("ABCDE")
    rng = np.random.default_rng(1)
    composite = pd.DataFrame(rng.normal(0, 1, (2, 5)), index=idx, columns=cols)
    beta = pd.DataFrame(rng.normal(1, 0.5, (2, 5)), index=idx, columns=cols)
    elig = pd.DataFrame(True, index=idx, columns=cols)

    resid = neutralize_against_beta(composite, beta, elig)

    for t in idx:
        r, b = resid.loc[t].to_numpy(), beta.loc[t].to_numpy()
        assert np.isclose(r.sum(), 0.0, atol=1e-10)            # dollar-neutral
        assert np.isclose(np.dot(r, b), 0.0, atol=1e-10)       # beta-neutral

        # independently recompute via numpy.polyfit (degree-1 OLS), not the
        # module's own centered-covariance shortcut
        c = composite.loc[t].to_numpy()
        slope, intercept = np.polyfit(b, c, 1)
        expected_resid = c - (intercept + slope * b)
        np.testing.assert_allclose(r, expected_resid, atol=1e-10)


def test_neutralize_against_beta_respects_eligibility_mask():
    idx = _bdays("2020-01-01", 1)
    cols = list("ABC")
    composite = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [999.0]}, index=idx)
    beta = pd.DataFrame({"A": [0.5], "B": [1.5], "C": [999.0]}, index=idx)
    elig = pd.DataFrame({"A": True, "B": True, "C": False}, index=idx)
    resid = neutralize_against_beta(composite, beta, elig)
    assert np.isnan(resid.loc[idx[0], "C"])
    # A/B-only result must match a 2-name-only computation (C never leaks in)
    only_ab = neutralize_against_beta(composite[["A", "B"]], beta[["A", "B"]],
                                      elig[["A", "B"]])
    assert np.isclose(resid.loc[idx[0], "A"], only_ab.loc[idx[0], "A"])
    assert np.isclose(resid.loc[idx[0], "B"], only_ab.loc[idx[0], "B"])


# --------------------- build_capped_weights ---------------------

def test_capped_weights_hit_gross_target_when_uncapped():
    # NOTE: with only ~4 names, a 10%-of-gross position cap makes the 200%
    # gross target mathematically unreachable (4 * 0.2 = 0.8 max) -- these
    # cap tests need a realistic-size universe (>= 10 names), matching the
    # real 70-ETF universe where caps bind on individual names, not the book.
    idx = _bdays("2020-01-01", 1)
    cols = [f"N{i}" for i in range(20)]
    # alternating +-1.0 / +-0.9, comfortably under the 10% position cap once
    # proportionally scaled to gross -- no cap should bind
    vals = [(1.0 if i % 2 == 0 else -1.0) * (1.0 if i < 10 else 0.9)
           for i in range(20)]
    raw = pd.DataFrame([vals], index=idx, columns=cols)
    categories = pd.Series({c: f"cat{i}" for i, c in enumerate(cols)})
    w = build_capped_weights(raw, categories)
    row = w.loc[idx[0]]
    assert np.isclose(row.abs().sum(), GROSS, atol=1e-8)
    assert np.isclose(row.sum(), 0.0, atol=1e-8)              # proportional scaling preserves dollar-neutrality
    assert (row.abs() < POSITION_CAP_FRAC * GROSS).all()      # confirms no cap actually bound
    # relative ratios preserved (proportional scaling only, no cap triggered)
    assert np.isclose(row["N0"] / row["N10"], 1.0 / 0.9, atol=1e-6)


def test_position_cap_enforced():
    idx = _bdays("2020-01-01", 1)
    # one dominant name -> must be clipped to POSITION_CAP_FRAC * GROSS; 20
    # filler names (unique categories, modest magnitude) supply enough spare
    # capacity that the 200% gross target is still reachable after clipping.
    fillers = [f"F{i}" for i in range(20)]
    cols = ["A"] + fillers
    raw = pd.DataFrame([[100.0] + [(1.0 if i % 2 == 0 else -1.0) for i in range(20)]],
                      index=idx, columns=cols)
    categories = pd.Series({"A": "catA", **{f: f"cat{f}" for f in fillers}})
    w = build_capped_weights(raw, categories)
    row = w.loc[idx[0]]
    pos_cap = POSITION_CAP_FRAC * GROSS
    assert row["A"] <= pos_cap + 1e-6
    assert np.isclose(row["A"], pos_cap, atol=1e-6)            # confirms the cap actually bound
    assert np.isclose(row.abs().sum(), GROSS, atol=1e-6)       # remainder redistributed to hit gross


def test_category_cap_enforced():
    idx = _bdays("2020-01-01", 1)
    # 5 names in the same category, each individually well under the 10%
    # position cap post-scaling, but their category total exceeds the 40%
    # category cap; 15 filler names (unique categories) supply spare capacity
    # so the 200% gross target is still reachable after category scale-down.
    same = [f"S{i}" for i in range(5)]
    fillers = [f"F{i}" for i in range(15)]
    cols = same + fillers
    raw = pd.DataFrame([[2.5] * 5 + [(1.0 if i % 2 == 0 else -1.0) for i in range(15)]],
                      index=idx, columns=cols)
    categories = pd.Series({**{s: "same" for s in same},
                            **{f: f"cat{f}" for f in fillers}})
    w = build_capped_weights(raw, categories)
    row = w.loc[idx[0]]
    pos_cap = POSITION_CAP_FRAC * GROSS
    cat_cap = CATEGORY_CAP_FRAC * GROSS
    assert (row[same].abs() < pos_cap - 1e-6).all()            # position cap never the binding constraint here
    cat_gross = row[same].abs().sum()
    assert np.isclose(cat_gross, cat_cap, atol=1e-6)           # confirms the category cap actually bound
    assert np.isclose(row.abs().sum(), GROSS, atol=1e-6)


# --------------------- no-trade band ---------------------

def test_freeze_small_changes_skips_small_keeps_large():
    prev = pd.Series({"A": 0.10, "B": 0.05})
    target = pd.Series({"A": 0.1001, "B": 0.10})   # A: +0.0001 (<band); B: +0.05 (>=band)
    out = _freeze_small_changes(target, prev, band=0.005)
    assert np.isclose(out["A"], 0.10)      # small change skipped -> stays at prev
    assert np.isclose(out["B"], 0.10)      # large change executed -> moves to target


def test_apply_no_trade_band_restores_gross_after_freezing():
    # A GROSS-scaled 2-date portfolio where date2's true target barely moves
    # for A (a no-trade-band case) but trades meaningfully for B/C -- checks
    # that freezing some names while trading others doesn't leave the
    # aggregate off the gross target (the bug this function's rescale fixes).
    idx = _bdays("2020-01-01", 2)
    w = pd.DataFrame({
        "A": [1.0, 1.0003],     # +0.0003 (< band) -> would be frozen at 1.0
        "B": [-0.6, -0.5],      # +0.1 (>= band) -> trades
        "C": [-0.4, -0.5003],   # -0.1003 (>= band) -> trades
    }, index=idx)
    assert np.isclose(w.abs().sum(axis=1).iloc[0], GROSS)   # date1 already at target
    out = apply_no_trade_band(w, band=0.005)
    assert np.isclose(out.abs().sum(axis=1).iloc[1], GROSS, atol=1e-9)
    # A was frozen (its raw target was within the band) -- confirm it moved
    # by the SAME uniform rescale factor applied to the traded names, not an
    # independent value
    raw_gross_before_rescale = 1.0 + 0.5 + 0.5003     # A kept at prev + B/C at target
    implied_scale = GROSS / raw_gross_before_rescale
    assert np.isclose(out.loc[idx[1], "A"], 1.0 * implied_scale, atol=1e-9)


# --------------------- point-in-time (full construction) ---------------------

def test_full_construction_is_point_in_time():
    n = 400
    idx = _bdays("2015-01-01", n)
    cols = list("ABCDEFGH")
    rng = np.random.default_rng(2)
    rets = pd.DataFrame(rng.normal(0, 0.01, (n, len(cols))), index=idx, columns=cols)
    close = (1 + rets).cumprod()
    elig = pd.DataFrame(True, index=idx, columns=cols)
    categories = pd.Series({c: "cat1" if i % 2 == 0 else "cat2"
                            for i, c in enumerate(cols)})
    window = idx[::5]           # every 5th day as a stand-in rebalance calendar

    cutoff = idx[300]
    full_w, full_s = compute_signal_and_weights(close, elig, categories, window)
    trunc_w, trunc_s = compute_signal_and_weights(
        close.loc[:cutoff], elig.loc[:cutoff], categories,
        window[window <= cutoff])

    common = window[window <= cutoff]
    common = common[common.isin(trunc_w.index) & common.isin(full_w.index)]
    assert len(common) > 10
    pd.testing.assert_frame_equal(full_w.loc[common], trunc_w.loc[common],
                                  check_exact=False, atol=1e-10)
    pd.testing.assert_frame_equal(full_s.loc[common], trunc_s.loc[common],
                                  check_exact=False, atol=1e-10)


def test_gross_and_neutrality_hold_across_a_full_run():
    n = 400
    idx = _bdays("2015-01-01", n)
    cols = list("ABCDEFGH")
    rng = np.random.default_rng(3)
    rets = pd.DataFrame(rng.normal(0, 0.01, (n, len(cols))), index=idx, columns=cols)
    close = (1 + rets).cumprod()
    elig = pd.DataFrame(True, index=idx, columns=cols)
    categories = pd.Series({c: "cat1" if i % 2 == 0 else "cat2"
                            for i, c in enumerate(cols)})
    window = idx[252::5]

    weights, signal = compute_signal_and_weights(close, elig, categories, window)
    nonzero_w = weights.dropna(how="all")
    nonzero_s = signal.dropna(how="all")
    assert len(nonzero_w) > 5
    gross = nonzero_w.abs().sum(axis=1)
    assert (gross <= GROSS + 1e-6).all()
    # Exact dollar-neutrality is guaranteed for the PRE-CAP signal (the
    # redemean step in compute_signal_and_weights) -- assert it there, where
    # the guarantee actually holds.
    sig_net = nonzero_s.sum(axis=1)
    assert (sig_net.abs() <= 1e-6).all()
    # Post-cap weights are only approximately neutral (spec §6): capping can
    # break exact zero-sum when it binds asymmetrically between the long and
    # short sides. Bound the residual rather than claim exactness.
    w_net = nonzero_w.sum(axis=1)
    assert (w_net.abs() <= CATEGORY_CAP_FRAC * GROSS + 1e-6).all()
