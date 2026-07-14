"""S5_carry construction (specs/EXP-004-tier2-carry-ic-screen-v3.md), extracted
VERBATIM from research/exp004_tier2_carry/run_exp004.py for reuse by Phase 3
(specs/EXP-005-phase3-m0-3signal-combiner-v2.md).

This is a fresh, independently-verified COPY, not a refactor-in-place --
research/exp004_tier2_carry/run_exp004.py is left untouched (frozen
experiment code). S5_carry's construction is CLOSED per decisions.md D39
(the graduating result is real; reopening the construction requires new
operator authorization this module does not seek) -- every constant and
function body below must match the original exactly. Required verification
before this module is trusted for any Phase 3 run:
research/exp005_phase3_m0_3sig/verify_carry_s5.py reproduces EXP-004's
confirmatory numbers bit-identically against
research/exp004_tier2_carry/results.json (independently re-run and
confirmed by both the reviewer and leak-hunter audits of EXP-005).

Requires FRED_API_KEY in the environment (source /workspace/activate.sh).
"""
import pandas as pd
import yfinance as yf
from fredapi import Fred

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
                                                # native CMT convention
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


def build_s5_carry(fred_api_key: str, rebal_full: pd.DatetimeIndex) -> tuple:
    """Full construction of S5_carry over `rebal_full` (any rebalance-date
    index spanning train/val and/or holdout). Returns (s5_full, credit_raw,
    ttm_yields, exdate_coverage) -- the same intermediate objects
    run_exp004.py computes, so downstream code (correlation diagnostics,
    lockbox payloads) can be built identically without re-deriving anything.
    """
    fred = Fred(api_key=fred_api_key)
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
    return s5_full, credit_raw, ttm_yields, exdate_coverage
