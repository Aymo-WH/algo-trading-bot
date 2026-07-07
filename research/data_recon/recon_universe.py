"""Phase R data reconnaissance: candidate ETF universe audit.

Downloads max-history daily adjusted OHLCV for the candidate universe, then reports
per ticker: inception, span, gap stats, and liquidity (median dollar volume).
Also measures effective breadth (participation ratio of return-correlation eigenvalues)
on PRE-HOLDOUT data only (through BREADTH_CUTOFF).

Quarantine note: this script computes DATA statistics only — no strategy returns,
no P&L, no signal evaluation. Allowed under mission §3.1.

Run:  /workspace/venv/bin/python research/data_recon/recon_universe.py
Outputs:
  data/recon/prices_close.csv, data/recon/prices_volume.csv  (raw panel, gitignored)
  research/data_recon/universe_summary.csv                    (committed evidence)
  research/data_recon/breadth_report.txt                      (committed evidence)
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yfinance as yf

# Candidate universe: liquid US-listed ETFs with (expected) long history, spanning
# sectors, broad equity, styles, countries, bonds, commodities, REITs, FX.
# RSX (Russia) is included deliberately as a survivorship probe: it was halted/delisted
# in 2022; whether yfinance still serves its history tells us how visible dead ETFs are.
UNIVERSE = {
    # US sectors (SPDR + industry)
    "XLB": "sector", "XLE": "sector", "XLF": "sector", "XLI": "sector",
    "XLK": "sector", "XLP": "sector", "XLU": "sector", "XLV": "sector",
    "XLY": "sector", "XBI": "sector", "XOP": "sector", "XHB": "sector",
    "XRT": "sector", "XME": "sector", "SMH": "sector", "IBB": "sector",
    "KRE": "sector", "GDX": "sector", "IYT": "sector", "ITB": "sector",
    # Broad US equity
    "SPY": "broad", "QQQ": "broad", "IWM": "broad", "DIA": "broad", "MDY": "broad",
    # Styles
    "IWD": "style", "IWF": "style", "IWN": "style", "IWO": "style",
    "VTV": "style", "VUG": "style", "RSP": "style",
    # Countries / regions
    "EWA": "country", "EWC": "country", "EWG": "country", "EWH": "country",
    "EWJ": "country", "EWL": "country", "EWM": "country", "EWP": "country",
    "EWQ": "country", "EWS": "country", "EWT": "country", "EWU": "country",
    "EWW": "country", "EWY": "country", "EWZ": "country", "FXI": "country",
    "EEM": "country", "EFA": "country", "ILF": "country", "EZA": "country",
    "TUR": "country", "EPI": "country", "RSX": "country_dead_probe", "VGK": "country",
    # Bonds / credit
    "TLT": "bond", "IEF": "bond", "SHY": "bond", "LQD": "bond", "HYG": "bond",
    "JNK": "bond", "AGG": "bond", "TIP": "bond", "EMB": "bond", "BND": "bond",
    # Commodities
    "GLD": "commodity", "SLV": "commodity", "USO": "commodity", "UNG": "commodity",
    "DBC": "commodity", "DBA": "commodity",
    # Real estate
    "IYR": "reit", "VNQ": "reit", "RWR": "reit",
    # FX
    "UUP": "fx", "FXE": "fx", "FXY": "fx",
}

BREADTH_CUTOFF = "2021-12-31"  # pre-holdout only; conservative (D2 in decisions.md)
LIQ_WINDOW_YEARS = 3

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(REPO, "data", "recon")
OUT_DIR = os.path.join(REPO, "research", "data_recon")
os.makedirs(RAW_DIR, exist_ok=True)


def main():
    tickers = sorted(UNIVERSE)
    print(f"Downloading {len(tickers)} tickers, period=max, daily, auto-adjusted ...")
    df = yf.download(tickers, period="max", interval="1d", auto_adjust=True,
                     progress=False, threads=8, group_by="column")
    close = df["Close"].sort_index()
    volume = df["Volume"].sort_index()
    # Restrict to dates where at least one ticker traded (drops all-NaN rows)
    close = close.dropna(how="all")
    volume = volume.reindex(close.index)
    close.to_csv(os.path.join(RAW_DIR, "prices_close.csv"))
    volume.to_csv(os.path.join(RAW_DIR, "prices_volume.csv"))

    rows = []
    last_date = close.index.max()
    liq_start = last_date - pd.DateOffset(years=LIQ_WINDOW_YEARS)
    for t in tickers:
        s = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
        if s.empty:
            rows.append({"ticker": t, "category": UNIVERSE[t], "status": "NO_DATA"})
            continue
        v = volume[t].reindex(s.index)
        dollar_vol = (s * v)
        recent = dollar_vol.loc[dollar_vol.index >= liq_start]
        bdays = pd.bdate_range(s.index.min(), s.index.max())
        rows.append({
            "ticker": t,
            "category": UNIVERSE[t],
            "status": "OK",
            "first_date": s.index.min().date(),
            "last_date": s.index.max().date(),
            "n_obs": len(s),
            "years": round((s.index.max() - s.index.min()).days / 365.25, 1),
            "pct_bdays_missing": round(100 * (1 - len(s) / max(len(bdays), 1)), 2),
            "median_dollar_vol_3y_musd": round(float(recent.median()) / 1e6, 1) if len(recent) else np.nan,
            "still_trading": bool(s.index.max() >= last_date - pd.Timedelta(days=7)),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "universe_summary.csv"), index=False)
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(summary.to_string(index=False))

    # ---- Effective breadth (pre-holdout data only) ----
    pre = close.loc[close.index <= BREADTH_CUTOFF]
    rets = pre.pct_change(fill_method=None)
    lines = [f"Effective-breadth report (returns through {BREADTH_CUTOFF} ONLY)", ""]
    for start in ["2007-01-01", "2012-01-01", "2017-01-01"]:
        window = rets.loc[start:]
        # keep tickers with >=95% coverage in the window
        keep = window.columns[window.notna().mean() >= 0.95]
        w = window[keep].dropna(how="any")
        if len(w) < 250 or len(keep) < 10:
            lines.append(f"{start}..cutoff: insufficient data ({len(keep)} tickers, {len(w)} days)")
            continue
        corr = w.corr()
        eig = np.linalg.eigvalsh(corr.values)[::-1]
        pr = float(eig.sum() ** 2 / (eig ** 2).sum())
        n90 = int(np.searchsorted(np.cumsum(eig) / eig.sum(), 0.90) + 1)
        lines.append(
            f"{start}..{BREADTH_CUTOFF}: N={len(keep)} tickers, {len(w)} common days | "
            f"participation ratio (effective N) = {pr:.1f} | "
            f"PCs for 90% var = {n90} | top-1 PC share = {eig[0] / eig.sum():.1%}"
        )
    report = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "breadth_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)


if __name__ == "__main__":
    sys.exit(main())
