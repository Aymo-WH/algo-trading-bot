"""EXP-004 runner (specs/EXP-004-tier2-carry-ic-screen-v3.md): S5_carry, a
10-name bond-sleeve carry signal combining a FRED treasury-curve leg
(unchanged from v2) and a yfinance-derived trailing-yield-spread proxy for
the credit leg (v3's correction after v2's OAS source proved structurally
infeasible pre-holdout, D36).

Train/validation ONLY for the confirmatory test -- holdout-period FRED and
yfinance rows (2022-01-01 onward) are pulled (for Sec3.3b reproducibility)
but immediately encrypted into a NEW lockbox (data/lockbox_carry/, D38) and
never used for grading. Descriptive/diagnostic legs (leg attribution,
static/timing decomposition, canaries, correlation vs S1/S2) follow the
frozen spec; none of them gate pass/fail.

Requires FRED_API_KEY in the environment (source /workspace/activate.sh).

Run: /workspace/venv/bin/python research/exp004_tier2_carry/run_exp004.py
Output: research/exp004_tier2_carry/results.json (committed evidence).
"""
import io
import json
import os
import sys
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

from src.decomposition import MIN_PRIOR_OBS, expanding_static_timing
from src.panel_factory import PANEL_DIR, TRAINVAL_END, rebalance_dates
from src.signals import momentum_s1, trend_s2
from validation.canaries import run_signal_canaries
from validation.ic import cross_sectional_ic, ic_summary, passes_graduation
from validation.ledger import config_hash, log_trial, read_ledger
from validation.lockbox import build_lockbox

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = "specs/EXP-004-tier2-carry-ic-screen-v3.md"
PANEL_START = "1999-12-22"        # same project-wide boundary EXP-001/003 used
HOLDOUT_START = "2022-01-01"      # must match src.panel_factory.HOLDOUT_START
BILL_SERIES = "DGS3MO"

TREASURY_TENOR = {"SHY": "DGS2", "IEF": "DGS7", "TLT": "DGS20",
                  "AGG": "DGS5", "BND": "DGS5", "TIP": "DGS7"}
CREDIT_TENOR = {"LQD": "DGS7", "HYG": "DGS3", "JNK": "DGS3", "EMB": "DGS7"}
TREASURY_NAMES = list(TREASURY_TENOR)
CREDIT_NAMES = list(CREDIT_TENOR)
ALL_NAMES = TREASURY_NAMES + CREDIT_NAMES        # 10-name sleeve

MIN_TTM_EXDATES = 8               # pinned data-sufficiency floor (spec v3)
TTM_WINDOW_DAYS = 366
LAG_BDAYS = 1                     # publication-lag buffer (spec v3/v2)
MAX_STALENESS_BDAYS = 5


def fred_asof(fred_series: pd.Series, asof_index: pd.DatetimeIndex,
              lag_bdays: int = LAG_BDAYS,
              max_staleness_bdays: int = MAX_STALENESS_BDAYS) -> pd.Series:
    """Value at or before `t - lag_bdays business days`; NaN if the most
    recent value is more than `max_staleness_bdays` business days stale."""
    fred_series = fred_series.sort_index()
    full_bidx = pd.bdate_range(fred_series.index.min(), asof_index.max())
    filled = fred_series.reindex(full_bidx).ffill(limit=max_staleness_bdays)
    lagged = filled.shift(lag_bdays)
    return lagged.reindex(asof_index)


def ttm_yield_and_coverage(dividends: pd.Series, raw_close: pd.Series,
                           asof_index: pd.DatetimeIndex) -> tuple:
    """TTM distribution yield at each asof date, plus the count of distinct
    ex-dividend dates found in that date's trailing window (diagnostic)."""
    dividends = dividends.copy()
    dividends.index = pd.DatetimeIndex(dividends.index).tz_localize(None).normalize()
    div_daily = dividends.groupby(level=0).sum()

    day_idx = pd.date_range(div_daily.index.min(), asof_index.max(), freq="D")
    div_full = div_daily.reindex(day_idx, fill_value=0.0)
    rolling_sum = div_full.rolling(f"{TTM_WINDOW_DAYS}D").sum()
    rolling_count = (div_full > 0).astype(float).rolling(f"{TTM_WINDOW_DAYS}D").sum()

    cutoff = asof_index - pd.tseries.offsets.BDay(LAG_BDAYS)
    ttm_sum = rolling_sum.reindex(cutoff)
    ttm_sum.index = asof_index
    n_exdates = rolling_count.reindex(cutoff)
    n_exdates.index = asof_index

    close_at_t = raw_close.reindex(asof_index).ffill(limit=3)
    ttm_yield = 100.0 * ttm_sum / close_at_t   # percentage points, matching FRED's
                                                # native CMT convention (e.g. 4.40
                                                # meaning 4.40%) -- BUG FOUND AND
                                                # FIXED post-first-run: the raw
                                                # decimal fraction (~0.05) was being
                                                # subtracted directly from a FRED
                                                # percentage-point value (~2-7),
                                                # which made credit_carry collapse
                                                # to approximately -matched_treasury_
                                                # cmt(t) -- see journal.md for the
                                                # diagnostic that caught this.
    ttm_yield = ttm_yield.where(n_exdates >= MIN_TTM_EXDATES)
    return ttm_yield, n_exdates


def pull_fred(fred: Fred, series_ids: list) -> dict:
    return {sid: fred.get_series(sid) for sid in series_ids}


def pull_credit_raw(tickers: list) -> dict:
    """Raw (split-adjusted, NOT dividend-adjusted) close + ex-dividend
    distributions, full history, one ticker at a time (Ticker.dividends is
    not available via the bulk yf.download path)."""
    out = {}
    for t in tickers:
        tk = yf.Ticker(t)
        hist = tk.history(period="max", auto_adjust=False)
        close = hist["Close"].copy()
        close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
        div = tk.dividends.copy()
        out[t] = {"close": close, "dividends": div}
    return out


def _build_lockbox_payload(s5_holdout, credit_raw):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("s5_carry_holdout.csv", s5_holdout.to_csv())
        for name in CREDIT_NAMES:
            z.writestr(f"{name}_raw_close_full.csv", credit_raw[name]["close"].to_csv())
            z.writestr(f"{name}_dividends_full.csv", credit_raw[name]["dividends"].to_csv())
    return buf.getvalue()


def main(lockbox_only: bool = False):
    fred_key = os.environ.get("FRED_API_KEY")
    assert fred_key, "FRED_API_KEY not set -- source /workspace/activate.sh first"
    fred = Fred(api_key=fred_key)

    close_panel = pd.read_csv(os.path.join(PANEL_DIR, "trainval_close.csv"),
                              index_col=0, parse_dates=True).sort_index()
    elig = pd.read_csv(os.path.join(PANEL_DIR, "eligibility_trainval.csv"),
                       index_col=0, parse_dates=True).sort_index().astype(bool)
    assert close_panel.index.max() == pd.Timestamp(TRAINVAL_END), "wrong trainval panel"

    rebal_all = rebalance_dates(close_panel.index)   # train/val calendar only
    # Full history of FRED series (train/val + holdout, Sec3.3b reproducibility);
    # we need the calendar to extend through the holdout too for the credit
    # leg's yfinance pull and the lockbox, so build a synthetic holdout
    # weekly grid off the same Wednesday-first-of-ISO-week rule.
    holdout_grid = pd.bdate_range(HOLDOUT_START, "2026-06-30")
    iso = holdout_grid.isocalendar()
    key = pd.MultiIndex.from_arrays([iso.year.to_numpy(), iso.week.to_numpy()])
    wed_or_later = holdout_grid[holdout_grid.dayofweek >= 2]
    iso2 = wed_or_later.isocalendar()
    key2 = pd.MultiIndex.from_arrays([iso2.year.to_numpy(), iso2.week.to_numpy()])
    holdout_rebal = pd.DatetimeIndex(sorted(
        pd.Series(wed_or_later, index=key2).groupby(level=[0, 1]).min().to_numpy()))
    rebal_full = rebal_all.append(holdout_rebal).unique().sort_values()

    print(f"Pulling FRED series (train/val + holdout, full history) ...")
    fred_ids = sorted(set(TREASURY_TENOR.values()) | set(CREDIT_TENOR.values()) | {BILL_SERIES})
    fred_raw = pull_fred(fred, fred_ids)

    bill = fred_asof(fred_raw[BILL_SERIES], rebal_full)
    treasury_carry = pd.DataFrame({
        name: fred_asof(fred_raw[TREASURY_TENOR[name]], rebal_full) - bill
        for name in TREASURY_NAMES
    })
    matched_treasury_for_credit = pd.DataFrame({
        name: fred_asof(fred_raw[CREDIT_TENOR[name]], rebal_full)
        for name in CREDIT_NAMES
    })

    print(f"Pulling yfinance raw close + dividends for {CREDIT_NAMES} ...")
    credit_raw = pull_credit_raw(CREDIT_NAMES)
    ttm_yields, exdate_coverage = {}, {}
    for name in CREDIT_NAMES:
        ttm, ncov = ttm_yield_and_coverage(credit_raw[name]["dividends"],
                                           credit_raw[name]["close"], rebal_full)
        ttm_yields[name] = ttm
        exdate_coverage[name] = ncov
    credit_carry = pd.DataFrame({name: ttm_yields[name] - matched_treasury_for_credit[name]
                                 for name in CREDIT_NAMES})

    s5_full = pd.concat([treasury_carry, credit_carry], axis=1)[ALL_NAMES]

    # --- split train/val (grading) vs holdout (lockbox only, never graded) ---
    s5_trainval = s5_full.loc[s5_full.index <= TRAINVAL_END]
    s5_holdout = s5_full.loc[s5_full.index >= HOLDOUT_START]

    if lockbox_only:
        # Housekeeping rebuild only (e.g. after the operator clears a stale
        # lockbox+token pair) -- reuses the exact same construction as the
        # full run, but skips the already-audited confirmatory/diagnostic
        # battery to avoid re-spending ~10 minutes on canaries that don't
        # need re-running. Does NOT touch results.json or the ledger.
        payload = _build_lockbox_payload(s5_holdout, credit_raw)
        import hashlib
        payload_sha = hashlib.sha256(payload).hexdigest()
        lockbox_dir = os.path.join(REPO, "data", "lockbox_carry")
        token_path = "/workspace/OPERATOR_TOKEN_CARRY.txt"
        enc_path = build_lockbox(payload, lockbox_dir=lockbox_dir, token_path=token_path)
        print(f"lockbox_carry REBUILT: {len(payload)} plaintext bytes -> {enc_path} "
              f"(sha256={payload_sha}); operator token written to {token_path}")
        return 0

    window = rebal_all[(rebal_all >= PANEL_START) & (rebal_all <= TRAINVAL_END)]
    s5_window = s5_trainval.reindex(window)
    elig_window = elig.reindex(window)[ALL_NAMES].fillna(False)
    s5_window = s5_window.where(elig_window)

    fwd5 = close_panel.shift(-5) / close_panel - 1.0

    # --- confirmatory: combined 10-name sleeve ---
    ic_raw = cross_sectional_ic(s5_window, fwd5, min_names=10)
    summary_raw = ic_summary(ic_raw, nw_lags=2)
    passed = passes_graduation(summary_raw)
    print(json.dumps({"signal": "S5_carry", **{k: v for k, v in summary_raw.items() if k != "years"},
                      "passes_graduation": passed}, default=str))

    # --- descriptive: leg attribution ---
    ic_treasury_leg = cross_sectional_ic(s5_window[TREASURY_NAMES], fwd5, min_names=6)
    ic_credit_leg = cross_sectional_ic(s5_window[CREDIT_NAMES], fwd5, min_names=4)
    leg_attribution = {
        "treasury_leg": {k: v for k, v in ic_summary(ic_treasury_leg).items() if k != "years"},
        "credit_leg": {k: v for k, v in ic_summary(ic_credit_leg).items() if k != "years"},
    }

    # --- descriptive: static/timing decomposition (src/decomposition.py, unchanged) ---
    static, timing = expanding_static_timing(s5_window, min_prior_obs=MIN_PRIOR_OBS)
    ic_static = cross_sectional_ic(static, fwd5, min_names=10)
    summary_static = {k: v for k, v in ic_summary(ic_static).items() if k != "years"}
    ic_timing = cross_sectional_ic(timing, fwd5, min_names=10)
    summary_timing = ic_summary(ic_timing)
    timing_passed = passes_graduation(summary_timing)
    summary_timing_out = {**{k: v for k, v in summary_timing.items() if k != "years"},
                          "passes_graduation": bool(timing_passed)}

    # --- diagnostic: canaries on raw + timing (D30/D33, non-blocking) ---
    canary_raw = run_signal_canaries(s5_window, fwd5)
    canary_timing = run_signal_canaries(timing, fwd5)

    # --- descriptive: correlation vs S1/S2 static/timing (D31 req 4) ---
    raw_close_full = close_panel  # full 70-name panel, for S1/S2 recompute
    s1_raw = momentum_s1(raw_close_full, elig)
    s2_raw = trend_s2(raw_close_full, elig)
    s1_static, s1_timing = expanding_static_timing(s1_raw.reindex(window), min_prior_obs=MIN_PRIOR_OBS)
    s2_static, s2_timing = expanding_static_timing(s2_raw.reindex(window), min_prior_obs=MIN_PRIOR_OBS)

    def stacked(df):
        return df.reindex(window)[ALL_NAMES].stack()

    corr_frame = pd.concat({
        "S5_raw": stacked(s5_window), "S5_static": stacked(static), "S5_timing": stacked(timing),
        "S1_raw": s1_raw.reindex(window)[ALL_NAMES].stack(),
        "S1_static": s1_static[ALL_NAMES].stack(), "S1_timing": s1_timing[ALL_NAMES].stack(),
        "S2_raw": s2_raw.reindex(window)[ALL_NAMES].stack(),
        "S2_static": s2_static[ALL_NAMES].stack(), "S2_timing": s2_timing[ALL_NAMES].stack(),
    }, axis=1)
    corr = corr_frame.corr().round(4)

    # --- descriptive: data-sanity diagnostics (v3 spec) ---
    data_sanity = {}
    for name in CREDIT_NAMES:
        yld = ttm_yields[name].reindex(window)
        cov = exdate_coverage[name].reindex(window)
        data_sanity[name] = {
            "ttm_yield_min": float(np.nanmin(yld)) if yld.notna().any() else None,
            "ttm_yield_max": float(np.nanmax(yld)) if yld.notna().any() else None,
            "any_negative": bool((yld.dropna() < 0).any()),
            "exdate_coverage_min": float(np.nanmin(cov)) if cov.notna().any() else None,
            "exdate_coverage_max": float(np.nanmax(cov)) if cov.notna().any() else None,
        }

    out = {
        "spec": SPEC,
        "trial_id": "EXP-004-S5_carry",
        "data_window": [str(window.min().date()), str(window.max().date())],
        "n_rebalance_dates": int(len(window)),
        "confirmatory": {**summary_raw, "passes_graduation": bool(passed)},
        "leg_attribution_descriptive": leg_attribution,
        "static_descriptive": summary_static,
        "timing_descriptive_not_graded": summary_timing_out,
        "canaries_diagnostic": {"raw": canary_raw, "timing": canary_timing},
        "correlation_vs_s1_s2_descriptive": corr.to_dict(),
        "credit_leg_data_sanity_descriptive": data_sanity,
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # --- lockbox: holdout-period raw yfinance inputs + derived carry (D38) ---
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("s5_carry_holdout.csv", s5_holdout.to_csv())
        for name in CREDIT_NAMES:
            z.writestr(f"{name}_raw_close_full.csv", credit_raw[name]["close"].to_csv())
            z.writestr(f"{name}_dividends_full.csv", credit_raw[name]["dividends"].to_csv())
    payload = buf.getvalue()
    import hashlib
    payload_sha = hashlib.sha256(payload).hexdigest()
    lockbox_dir = os.path.join(REPO, "data", "lockbox_carry")
    token_path = "/workspace/OPERATOR_TOKEN_CARRY.txt"
    try:
        enc_path = build_lockbox(payload, lockbox_dir=lockbox_dir, token_path=token_path)
        print(f"lockbox_carry: {len(payload)} plaintext bytes -> {enc_path} "
              f"(sha256={payload_sha[:16]}...); operator token written to {token_path}")
    except FileExistsError:
        print(f"lockbox_carry ALREADY EXISTS (from a prior run) -- refusing to "
              f"overwrite, same one-shot semantics as the price-panel lockbox "
              f"(D17/D21 precedent). This run's holdout payload sha256="
              f"{payload_sha} was NOT written. Operator must delete the stale "
              f"lockbox_dir/token pair themselves before a corrected lockbox can "
              f"be built -- not something this script or the quarantine guard "
              f"will do automatically.")
    del payload, buf

    # --- log ledger row. Dedup is by (id, config_hash), NOT id alone: a code
    # bug (credit-leg units mismatch, found and fixed post-first-run -- see
    # journal.md) produced a real, already-logged EXP-004-S5_carry row from
    # buggy data. That row is NOT deleted (append-only ledger, "log every
    # trial including discards") but this corrected run must log as its OWN
    # trial, not silently no-op as "just a reproducibility re-run" -- hence
    # CONSTRUCTION_REV in the config, which changes config_hash across the
    # bugfix boundary on purpose. ---
    CONSTRUCTION_REV = "credit_units_fixed_2026-07-11"
    trial_config = {"spec": SPEC, "treasury_tenor": TREASURY_TENOR,
                    "credit_tenor": CREDIT_TENOR, "min_ttm_exdates": MIN_TTM_EXDATES,
                    "data_window": out["data_window"], "construction_rev": CONSTRUCTION_REV}
    this_hash = config_hash(trial_config)
    already = {(r["id"], r.get("config_hash")) for r in read_ledger()
              if r.get("phase") == "result" and r.get("id") == "EXP-004-S5_carry"}
    if ("EXP-004-S5_carry", this_hash) not in already:
        log_trial(
            experiment_id="EXP-004-S5_carry", phase="result",
            config=trial_config,
            metrics={k: v for k, v in summary_raw.items() if k != "years"},
            hypothesis="S5_carry clears mean_ic>=0.01 AND t_nw>=2.0 AND "
                      "pct_years_positive>=0.60 within the 10-name bond sleeve "
                      "(min_names=10), per specs/EXP-004-tier2-carry-ic-screen-v3.md",
            notes=f"EXP-004 real trial (v3 construction: FRED treasury leg + "
                  f"yfinance credit-leg TTM-yield proxy). passes_graduation="
                  f"{passed}. Full detail in "
                  f"{os.path.relpath(out_path, REPO)}. lockbox_carry sha256="
                  f"{payload_sha}",
        )
        print("LOGGED EXP-004-S5_carry (phase=result)")
    else:
        print("SKIP logging EXP-004-S5_carry: already present in the real ledger "
              "(re-run for verification only, not a new trial)")

    print(f"\nS5_carry passes_graduation={passed}")
    print(f"Leg attribution: {json.dumps(leg_attribution, default=str)}")
    return 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lockbox-only", action="store_true",
                   help="rebuild the lockbox only (e.g. after the operator clears "
                        "a stale enc/token pair); skips the confirmatory/diagnostic "
                        "battery and does not touch results.json or the ledger.")
    a = p.parse_args()
    sys.exit(main(lockbox_only=a.lockbox_only))
