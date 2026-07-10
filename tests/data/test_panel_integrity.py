"""Phase 1 data-integrity suite (mission §5 Phase 1: "clean, leakage-audited
panel data + a data-integrity test suite").

Part A — unit tests of the point-in-time machinery on synthetic frames
(always run). Part B — integrity checks of the frozen artifacts under
data/panel/ (skipped until the panel is cut; green artifact tests are the
Phase-1 gate).
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.panel_factory import (HOLDOUT_END, HOLDOUT_START, MANIFEST,
                               MIN_ELIGIBLE, MIN_HISTORY_OBS, PANEL_DIR,
                               TRAINVAL_END, UNIVERSE, panel_start,
                               pit_eligibility, rebalance_dates,
                               trading_calendar)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------- Part A: unit tests ---------------------------

def _bdays(start, n):
    return pd.bdate_range(start, periods=n)


def test_pit_eligibility_first_true_is_nth_observation():
    idx = _bdays("2020-01-01", 20)
    close = pd.DataFrame({"A": 1.0, "B": np.nan}, index=idx)
    close.loc[idx[5]:, "B"] = 2.0            # B's first obs is idx[5]
    close.loc[idx[8], "B"] = np.nan          # a gap must not count
    m = pit_eligibility(close, min_obs=5)
    assert m["A"].idxmax() == idx[4]          # 5th obs of A (inclusive count)
    # B: obs at 5,6,7,(gap),9,10 -> 5th obs lands at idx[10]
    assert m["B"].idxmax() == idx[10]
    assert not m["B"].loc[:idx[9]].any()


def test_pit_eligibility_monotone_no_exits():
    rng = np.random.default_rng(0)
    idx = _bdays("2015-01-01", 300)
    close = pd.DataFrame(rng.uniform(1, 2, (300, 3)), index=idx,
                         columns=list("ABC"))
    close.iloc[rng.choice(300, 40, replace=False), 1] = np.nan  # random gaps
    m = pit_eligibility(close, min_obs=50)
    assert (m.astype(int).diff().fillna(0) >= 0).all().all()


def test_pit_eligibility_uses_only_past_data():
    """Truncating the future must not change the past — the no-look-ahead
    property, asserted literally."""
    rng = np.random.default_rng(1)
    idx = _bdays("2015-01-01", 400)
    close = pd.DataFrame(rng.uniform(1, 2, (400, 4)), index=idx,
                         columns=list("ABCD"))
    close.iloc[:150, 2] = np.nan
    full = pit_eligibility(close, min_obs=100)
    for cut in (100, 250, 399):
        part = pit_eligibility(close.iloc[:cut], min_obs=100)
        pd.testing.assert_frame_equal(full.iloc[:cut], part)


def test_rebalance_dates_wednesday_or_next_open_day():
    idx = pd.bdate_range("2024-01-01", "2024-01-31")
    holiday_wed = pd.Timestamp("2024-01-17")
    cal = pd.DatetimeIndex([d for d in idx if d != holiday_wed])
    rb = rebalance_dates(cal)
    assert pd.Timestamp("2024-01-03") in rb          # normal Wednesday
    assert holiday_wed not in rb
    assert pd.Timestamp("2024-01-18") in rb          # Thursday fallback
    assert all(d.dayofweek >= 2 for d in rb)
    # one rebalance per ISO week
    iso = rb.isocalendar()
    assert not pd.MultiIndex.from_arrays([iso.year, iso.week]).duplicated().any()


def test_rebalance_dates_skips_week_with_no_wed_fri_session():
    cal = pd.DatetimeIndex(["2024-01-08", "2024-01-09",          # Mon, Tue only
                            "2024-01-15", "2024-01-17"])
    rb = rebalance_dates(cal)
    assert list(rb) == [pd.Timestamp("2024-01-17")]


def test_trading_calendar_drops_anchor_gaps_and_rejects_weekends():
    idx = _bdays("2020-01-01", 10)
    close = pd.DataFrame({"SPY": 1.0, "X": 1.0}, index=idx)
    close.loc[idx[3], "SPY"] = np.nan
    cal = trading_calendar(close)
    assert idx[3] not in cal and len(cal) == 9
    bad = close.copy()
    bad.index = idx.map(lambda d: d + pd.Timedelta(days=5 - d.dayofweek)
                        if d.dayofweek == 4 else d)  # push Fridays to weekend
    with pytest.raises(ValueError):
        trading_calendar(bad.dropna())


def test_panel_start_first_rebalance_with_enough_names():
    idx = _bdays("2020-01-01", 60)
    elig = pd.DataFrame(False, index=idx, columns=list("ABC"))
    elig.loc[idx[10]:, ["A", "B"]] = True
    elig.loc[idx[30]:, "C"] = True
    rb = rebalance_dates(pd.DatetimeIndex(idx))
    start = panel_start(elig, rb, min_eligible=3)
    assert start >= idx[30] and start in rb
    assert elig.loc[start].sum() >= 3
    with pytest.raises(ValueError):
        panel_start(elig, rb, min_eligible=4)


# ----------------------- Part B: frozen-artifact tests ---------------------

artifact = pytest.mark.skipif(not os.path.exists(MANIFEST),
                              reason="panel not cut yet (no MANIFEST.json)")


@pytest.fixture(scope="module")
def panel():
    if not os.path.exists(MANIFEST):
        pytest.skip("panel not cut yet")
    m = json.load(open(MANIFEST))
    read = lambda n: pd.read_csv(os.path.join(PANEL_DIR, n), index_col=0,
                                 parse_dates=True).sort_index()
    return {"manifest": m, "close": read("trainval_close.csv"),
            "volume": read("trainval_volume.csv"),
            "elig": read("eligibility_trainval.csv").astype(bool)}


@artifact
def test_quarantine_boundary_nothing_after_trainval_end(panel):
    tv_end = pd.Timestamp(TRAINVAL_END)
    for k in ("close", "volume", "elig"):
        assert panel[k].index.max() <= tv_end, f"{k} leaks past the boundary"
    assert panel["close"].index.max() == pd.Timestamp("2021-12-31")


@artifact
def test_quarantine_no_cleartext_holdout_dates_anywhere_in_data_dir(panel):
    """No clear-text file under data/ may contain post-boundary dates.
    (The lockbox is the ONLY place 2022+ lives; its dir is excluded — and
    never read here.)"""
    tv_end = pd.Timestamp(TRAINVAL_END)
    for path in glob.glob(os.path.join(REPO, "data", "**", "*.csv"),
                          recursive=True):
        if os.sep + "lockbox" + os.sep in path:
            continue
        df = pd.read_csv(path, index_col=0)
        try:
            idx = pd.to_datetime(df.index)
        except (ValueError, TypeError):
            continue                         # not a date-indexed file
        assert idx.max() <= tv_end, f"{path} contains post-boundary dates"


@artifact
def test_stale_recon_price_files_removed():
    for name in ("prices_close.csv", "prices_volume.csv"):
        assert not os.path.exists(os.path.join(REPO, "data", "recon", name)), \
            f"stale Phase-R recon file {name} still on disk (holds 2022+ data)"


@artifact
def test_lockbox_exists_without_reading_it(panel):
    enc = panel["manifest"]["holdout"]["encrypted_at"]
    assert os.path.exists(enc)
    assert panel["manifest"]["holdout"]["plaintext_sha256"]
    assert panel["manifest"]["holdout"]["columns_match_trainval"] is True


@artifact
def test_universe_columns_exact(panel):
    want = sorted(UNIVERSE)
    assert len(want) == 70
    for k in ("close", "volume", "elig"):
        assert list(panel[k].columns) == want, f"{k} columns != frozen universe"


@artifact
def test_calendar_integrity(panel):
    idx = panel["close"].index
    assert idx.is_monotonic_increasing and idx.is_unique
    assert (idx.dayofweek < 5).all(), "weekend rows present"
    assert panel["close"]["SPY"].notna().all(), "calendar anchor has gaps"
    assert panel["close"].notna().any(axis=1).all(), "all-NaN rows present"
    for k in ("volume", "elig"):
        assert panel[k].index.equals(idx), f"{k} index misaligned"


@artifact
def test_eligibility_recomputable_point_in_time(panel):
    recomputed = pit_eligibility(panel["close"], min_obs=MIN_HISTORY_OBS)
    pd.testing.assert_frame_equal(panel["elig"], recomputed,
                                  check_dtype=False)
    assert (panel["elig"].astype(int).diff().fillna(0) >= 0).all().all()


@artifact
def test_panel_start_rule_holds(panel):
    m = panel["manifest"]
    start = pd.Timestamp(m["panel_start"])
    rb = rebalance_dates(panel["close"].index)
    assert start in rb
    assert panel["elig"].loc[start].sum() >= MIN_ELIGIBLE
    earlier = [d for d in rb if d < start]
    if earlier:
        assert panel["elig"].loc[earlier].sum(axis=1).max() < MIN_ELIGIBLE, \
            "an earlier rebalance already had enough names — start is wrong"


@artifact
def test_prices_positive_volume_nonnegative(panel):
    c, v = panel["close"], panel["volume"]
    assert (c.stack().dropna() > 0).all(), "non-positive prices"
    assert (v.stack().dropna() >= 0).all(), "negative volume"


@artifact
def test_no_split_sized_return_outliers(panel):
    """A mishandled split shows up as a +/-50-100% one-day 'return'. Adjusted
    mega-ETF prices must never move 60% in a session."""
    rets = panel["close"].pct_change(fill_method=None)
    worst = rets.stack().abs().sort_values(ascending=False).head(5)
    assert worst.iloc[0] < 0.60, f"split-sized outliers: {worst.to_dict()}"


@artifact
def test_manifest_hashes_match_disk(panel):
    from src.panel_factory import sha256_file
    for name, meta in panel["manifest"]["artifacts"].items():
        assert sha256_file(os.path.join(PANEL_DIR, name)) == meta["sha256"], \
            f"{name} was modified after the cut"


@artifact
def test_boundaries_agree_with_frozen_referee(panel):
    import validation.final_eval as fe
    assert HOLDOUT_START == fe.HOLDOUT_START
    assert HOLDOUT_END == fe.HOLDOUT_END
    assert panel["manifest"]["holdout"]["start"] == fe.HOLDOUT_START
    assert pd.Timestamp(TRAINVAL_END) < pd.Timestamp(fe.HOLDOUT_START)


@artifact
def test_build_logged_but_excluded_from_dsr_trial_count(panel):
    from validation.ledger import read_ledger, trial_count
    rows = [r for r in read_ledger() if r.get("phase") == "data_build"]
    assert len(rows) == 1 and rows[0]["id"] == "PHASE1-PANEL-BUILD"
    result_rows = [r for r in read_ledger() if r.get("phase") in ("trial", "result")]
    assert trial_count() == len(result_rows), "trial_count() must reflect only trial/result rows"
    assert not any(r.get("id") == "PHASE1-PANEL-BUILD" for r in result_rows), \
        "data build must not inflate the DSR trial count"
