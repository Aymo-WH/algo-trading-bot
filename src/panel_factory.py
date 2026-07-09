"""Phase 1 — Gordian v2 data & universe layer (mission §5 Phase 1; design §2–§3).

Builds the frozen daily panel for the v2 universe and cuts the quarantine split:

- train/validation (dates <= 2021-12-31) written IN THE CLEAR to data/panel/
  (committed to git: clean-checkout reproducibility, mission §3.3b);
- holdout close panel (2022-01-01 .. 2026-06-30) encrypted into a FRESH lockbox
  via validation.lockbox.build_lockbox — the plaintext is never written to disk,
  only its SHA-256 lands in the manifest for tamper-evidence at final eval;
- MANIFEST.json (committed) freezes the universe, the entry rule, the split
  boundaries, and the SHA-256 of every artifact.

Universe rule (frozen, decision D9 + design §2, measured in
research/data_recon/universe_summary.csv): status OK and median 3y daily dollar
volume >= $10M. Applying the rule yields 70 names (the design doc's "69" was an
arithmetic slip — see decisions.md D16). Membership is fixed by this rule alone;
no return-based inclusion or removal, ever.

Point-in-time entry (design §2): a ticker becomes ELIGIBLE (tradable) at its
first date with >= 252 trading days of observed history; the tradable panel
starts at the first weekly (Wednesday) rebalance with >= 20 eligible names.
Prices BEFORE eligibility stay in the panel — history is information,
eligibility is tradability.

One-shot: refuses to run if MANIFEST.json or the lockbox already exists
(re-cutting the dataset is an operator-approved action, same semantics as
validation.lockbox.build_lockbox).

Run:  /workspace/venv/bin/python src/panel_factory.py
"""
import argparse
import datetime
import hashlib
import io
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PANEL_DIR = os.path.join(REPO, "data", "panel")
MANIFEST = os.path.join(PANEL_DIR, "MANIFEST.json")

TRAINVAL_END = "2021-12-31"   # design §3 / D8 — quarantine boundary
HOLDOUT_START = "2022-01-01"  # must match validation.final_eval.HOLDOUT_START
HOLDOUT_END = "2026-06-30"    # must match validation.final_eval.HOLDOUT_END

MIN_HISTORY_OBS = 252         # PIT entry rule (design §2)
MIN_ELIGIBLE = 20             # panel starts at first rebalance with >= 20 names
REBALANCE_WEEKDAY = 2         # Wednesday (design §4)
CALENDAR_TICKER = "SPY"       # NYSE-session proxy: SPY trades every NYSE session

LIQUIDITY_RULE = ("status == 'OK' and median_dollar_vol_3y_musd >= 10.0 in "
                  "research/data_recon/universe_summary.csv (recon of 78 "
                  "candidates, 2026-07-07)")

# The 70-name v2 universe: the liquidity rule applied to the committed recon
# evidence. Categories carried from research/data_recon/recon_universe.py
# (used later for category caps, design §7).
UNIVERSE = {
    "AGG": "bond", "BND": "bond", "EMB": "bond", "HYG": "bond", "IEF": "bond",
    "JNK": "bond", "LQD": "bond", "SHY": "bond", "TIP": "bond", "TLT": "bond",
    "DIA": "broad", "IWM": "broad", "MDY": "broad", "QQQ": "broad", "SPY": "broad",
    "DBC": "commodity", "GLD": "commodity", "SLV": "commodity",
    "UNG": "commodity", "USO": "commodity",
    "EEM": "country", "EFA": "country", "EPI": "country", "EWA": "country",
    "EWC": "country", "EWG": "country", "EWH": "country", "EWJ": "country",
    "EWL": "country", "EWP": "country", "EWQ": "country", "EWT": "country",
    "EWU": "country", "EWW": "country", "EWY": "country", "EWZ": "country",
    "FXI": "country", "ILF": "country", "VGK": "country",
    "UUP": "fx",
    "IYR": "reit", "RWR": "reit", "VNQ": "reit",
    "GDX": "sector", "IBB": "sector", "ITB": "sector", "IYT": "sector",
    "KRE": "sector", "SMH": "sector", "XBI": "sector", "XHB": "sector",
    "XLB": "sector", "XLE": "sector", "XLF": "sector", "XLI": "sector",
    "XLK": "sector", "XLP": "sector", "XLU": "sector", "XLV": "sector",
    "XLY": "sector", "XME": "sector", "XOP": "sector", "XRT": "sector",
    "IWD": "style", "IWF": "style", "IWN": "style", "IWO": "style",
    "RSP": "style", "VTV": "style", "VUG": "style",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def trading_calendar(close: pd.DataFrame,
                     anchor: str = CALENDAR_TICKER) -> pd.DatetimeIndex:
    """NYSE-session calendar = dates where the anchor (SPY) traded."""
    if anchor not in close.columns:
        raise ValueError(f"calendar anchor {anchor} missing from panel")
    cal = close.index[close[anchor].notna()]
    if (cal.dayofweek >= 5).any():
        raise ValueError("calendar contains weekend dates — bad source data")
    return cal


def pit_eligibility(close: pd.DataFrame,
                    min_obs: int = MIN_HISTORY_OBS) -> pd.DataFrame:
    """True from each ticker's `min_obs`-th observed date onward (inclusive).

    Uses only data at or before each date (a cumulative count) — no look-ahead
    by construction. Monotone non-decreasing: no exits (design §2).
    """
    return close.notna().cumsum() >= min_obs


def rebalance_dates(calendar: pd.DatetimeIndex,
                    weekday: int = REBALANCE_WEEKDAY) -> pd.DatetimeIndex:
    """Weekly rebalance dates: the first trading day >= `weekday` (Wed) within
    each ISO week; a week with no Wed-Fri session is skipped."""
    days = calendar[calendar.dayofweek >= weekday]
    if len(days) == 0:
        return pd.DatetimeIndex([])
    iso = days.isocalendar()
    key = pd.MultiIndex.from_arrays([iso.year.to_numpy(), iso.week.to_numpy()])
    first = pd.Series(days, index=key).groupby(level=[0, 1]).min()
    return pd.DatetimeIndex(sorted(first.to_numpy()))


def panel_start(elig: pd.DataFrame, rebal: pd.DatetimeIndex,
                min_eligible: int = MIN_ELIGIBLE) -> pd.Timestamp:
    counts = elig.loc[elig.index.intersection(rebal)].sum(axis=1)
    ok = counts[counts >= min_eligible]
    if ok.empty:
        raise ValueError(f"no rebalance date reaches {min_eligible} eligible names")
    return ok.index[0]


def download_panel(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily auto-adjusted close + volume, max history (same call shape as the
    Phase-R recon, research/data_recon/recon_universe.py)."""
    import yfinance as yf
    df = yf.download(tickers, period="max", interval="1d", auto_adjust=True,
                     progress=False, threads=8, group_by="column")
    close = df["Close"].sort_index().dropna(how="all")
    volume = df["Volume"].reindex(close.index)
    close.index = pd.DatetimeIndex(close.index.date)   # drop tz/time component
    volume.index = close.index
    return close[sorted(tickers)], volume[sorted(tickers)]


def build(end: str = HOLDOUT_END, out_dir: str = PANEL_DIR) -> dict:
    manifest_path = os.path.join(out_dir, "MANIFEST.json")
    if os.path.exists(manifest_path):
        raise FileExistsError(
            f"{manifest_path} exists — the panel is already cut. Re-cutting is "
            "an operator-approved action: delete the panel AND the lockbox "
            "first (validation.lockbox has the same one-shot semantics).")

    tickers = sorted(UNIVERSE)
    print(f"Downloading {len(tickers)} tickers, period=max, daily, adjusted ...")
    close, volume = download_panel(tickers)

    missing = [t for t in tickers if close[t].dropna().empty]
    if missing:
        raise RuntimeError(f"no data for {missing} — universe/data mismatch, "
                           "refusing to cut a short panel")

    cal = trading_calendar(close)
    close, volume = close.loc[cal], volume.loc[cal]
    close, volume = close.loc[close.index <= end], volume.loc[volume.index <= end]

    elig = pit_eligibility(close)
    rebal = rebalance_dates(close.index)
    start = panel_start(elig, rebal)

    tv_end = pd.Timestamp(TRAINVAL_END)
    h_start = pd.Timestamp(HOLDOUT_START)
    tv_close = close.loc[close.index <= tv_end]
    tv_volume = volume.loc[volume.index <= tv_end]
    tv_elig = elig.loc[elig.index <= tv_end]
    holdout_close = close.loc[close.index >= h_start]

    # --- refuse-before-write sanity (mirrors final_eval's own checks) ---
    assert tv_close.index.max() < h_start, "train/val crosses the boundary"
    assert list(holdout_close.columns) == list(tv_close.columns), \
        "holdout/trainval column mismatch"
    assert abs((holdout_close.index.min() - h_start).days) <= 7, \
        f"holdout starts {holdout_close.index.min().date()} — wrong slice"
    assert abs((holdout_close.index.max() - pd.Timestamp(end)).days) <= 7, \
        f"holdout ends {holdout_close.index.max().date()} — truncated slice"
    assert start < tv_end, "panel start fell inside the holdout — rule broken"

    # --- lockbox FIRST (it is the one-shot guard: raises if a box exists) ---
    holdout_bytes = holdout_close.to_csv().encode()
    from validation.lockbox import build_lockbox
    enc_path = build_lockbox(holdout_bytes)
    holdout_sha = sha256_bytes(holdout_bytes)
    del holdout_bytes, holdout_close  # plaintext holdout never touches disk

    os.makedirs(out_dir, exist_ok=True)
    artifacts = {}
    for name, frame in (("trainval_close.csv", tv_close),
                        ("trainval_volume.csv", tv_volume),
                        ("eligibility_trainval.csv", tv_elig)):
        path = os.path.join(out_dir, name)
        frame.to_csv(path)
        artifacts[name] = {"sha256": sha256_file(path),
                           "rows": len(frame), "cols": frame.shape[1],
                           "first": str(frame.index.min().date()),
                           "last": str(frame.index.max().date())}

    import yfinance
    manifest = {
        "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "builder": "src/panel_factory.py",
        "yfinance": yfinance.__version__, "pandas": pd.__version__,
        "universe_rule": LIQUIDITY_RULE,
        "universe": UNIVERSE, "n_tickers": len(tickers),
        "calendar": f"trading days = dates where {CALENDAR_TICKER} traded",
        "pit_entry_rule": f"eligible from the {MIN_HISTORY_OBS}th observed "
                          f"trading day (inclusive); no exits",
        "rebalance_rule": "first trading day >= Wednesday of each ISO week",
        "panel_start_rule": f"first rebalance with >= {MIN_ELIGIBLE} eligible",
        "panel_start": str(start.date()),
        "trainval_end": TRAINVAL_END,
        "holdout": {"start": HOLDOUT_START, "end": end,
                    "encrypted_at": enc_path,
                    "plaintext_sha256": holdout_sha,
                    "rows": int((close.index >= h_start).sum()),
                    "cols": len(tickers),
                    "columns_match_trainval": True,
                    "contents": "close prices only (Tier-1 is price-based; "
                                "holdout volume deliberately not retained)"},
        "artifacts": artifacts,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    from validation.ledger import log_trial
    log_trial(experiment_id="PHASE1-PANEL-BUILD", phase="data_build",
              config={"universe_rule": LIQUIDITY_RULE, "n_tickers": len(tickers),
                      "trainval_end": TRAINVAL_END,
                      "holdout": [HOLDOUT_START, end],
                      "pit_min_obs": MIN_HISTORY_OBS,
                      "min_eligible": MIN_ELIGIBLE},
              metrics={"panel_start": str(start.date()),
                       "trainval_rows": len(tv_close),
                       "holdout_plaintext_sha256": holdout_sha,
                       **{k: v["sha256"] for k, v in artifacts.items()}},
              notes="Phase 1 data cut; NOT a strategy trial (phase='data_build' "
                    "is excluded from trial_count/DSR by construction). Zero "
                    "strategy trials have been run as of this build.")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--end", default=HOLDOUT_END)
    a = p.parse_args()
    m = build(end=a.end)
    print(json.dumps({k: m[k] for k in
                      ("panel_start", "trainval_end", "n_tickers")}, indent=2))
    print(json.dumps(m["artifacts"], indent=2))
    print(f"holdout: rows={m['holdout']['rows']} sha256="
          f"{m['holdout']['plaintext_sha256'][:16]}... -> encrypted; "
          f"operator token written (see validation/lockbox.py TOKEN_PATH — "
          f"operator: retrieve and secure it now)")


if __name__ == "__main__":
    main()
