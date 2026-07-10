"""EXP-001 runner (specs/EXP-001-tier1-signal-ic.md): compute the real
train/validation IC for the four Tier-1 signals and grade them against the
pre-registered graduation rule. Train/validation ONLY -- data/panel never
touches the holdout/lockbox.

Logs exactly one real ledger row per signal (phase="result") -- guarded so a
re-run for reproducibility verification (mission §3.3b) does not silently
double-count trials against the 250 budget.

Run: /workspace/venv/bin/python research/exp001_signal_ic/run_exp001.py
Output: research/exp001_signal_ic/results.json (committed evidence).
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

from src.panel_factory import PANEL_DIR, TRAINVAL_END, rebalance_dates
from src.signals import (equal_weight_market_return, low_beta_s3,
                         momentum_s1, seasonality_s4, trend_s2)
from validation.ic import cross_sectional_ic, ic_summary, passes_graduation
from validation.ledger import log_trial, read_ledger

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_START = "1999-12-22"    # data/panel/MANIFEST.json panel_start (D19)

SIGNAL_FNS = {
    "S1_momentum": momentum_s1,
    "S2_ts_trend": trend_s2,
    "S3_low_beta": low_beta_s3,
    "S4_seasonality": seasonality_s4,
}


def already_logged() -> set:
    return {r["id"] for r in read_ledger()
            if r.get("phase") == "result" and str(r.get("id", "")).startswith("EXP-001-")}


def main():
    close = pd.read_csv(os.path.join(PANEL_DIR, "trainval_close.csv"),
                        index_col=0, parse_dates=True).sort_index()
    elig = pd.read_csv(os.path.join(PANEL_DIR, "eligibility_trainval.csv"),
                       index_col=0, parse_dates=True).sort_index().astype(bool)
    assert close.index.max() == pd.Timestamp(TRAINVAL_END), "wrong trainval panel"

    ret = close.pct_change(fill_method=None)
    market = equal_weight_market_return(ret, elig)
    fwd5 = close.shift(-5) / close - 1.0

    rebal = rebalance_dates(close.index)
    window = rebal[(rebal >= PANEL_START) & (rebal <= TRAINVAL_END)]

    raw = {
        "S1_momentum": momentum_s1(close, elig),
        "S2_ts_trend": trend_s2(close, elig),
        "S3_low_beta": low_beta_s3(close, elig, market=market),
        "S4_seasonality": seasonality_s4(close, elig, market=market),
    }
    weekly = {name: df.reindex(window) for name, df in raw.items()}

    results = {}
    for name, sig in weekly.items():
        ic = cross_sectional_ic(sig, fwd5, min_names=10)
        summary = ic_summary(ic, nw_lags=2)
        passed = passes_graduation(summary)
        results[name] = {**summary, "passes_graduation": bool(passed)}
        print(json.dumps({"signal": name, **{k: v for k, v in summary.items()
                                             if k != "years"},
                          "passes_graduation": passed}, default=str))

    # Descriptive only (design §6) -- not part of the pass/fail rule.
    stacked = pd.concat({name: df.reindex(window).stack() for name, df in raw.items()},
                        axis=1)
    corr = stacked.corr().round(4)
    print("\nInter-signal correlation (stacked date x asset observations):")
    print(corr.to_string())

    n_pass = sum(r["passes_graduation"] for r in results.values())
    verdict = {
        "hypothesis": "tier1_null" if n_pass == 0 else "partial_or_full_survival",
        "n_signals_passed": n_pass, "n_signals_total": len(results),
    }
    out = {
        "spec": "specs/EXP-001-tier1-signal-ic.md",
        "data_window": [str(window.min().date()), str(window.max().date())],
        "n_rebalance_dates": int(len(window)),
        "results": results,
        "inter_signal_correlation": corr.to_dict(),
        "verdict": verdict,
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    done = already_logged()
    for name, r in results.items():
        exp_id = f"EXP-001-{name}"
        if exp_id in done:
            print(f"SKIP logging {exp_id}: already present in the real ledger "
                  f"(re-run for verification only, not a new trial)")
            continue
        log_trial(
            experiment_id=exp_id, phase="result",
            config={"spec": "specs/EXP-001-tier1-signal-ic.md", "signal": name,
                    "data_window": out["data_window"]},
            metrics={k: v for k, v in r.items() if k != "years"},
            hypothesis=f"{name} clears mean_ic>=0.01 AND t_nw>=2.0 AND "
                       f"pct_years_positive>=0.60 (design.md S5)",
            notes=f"EXP-001 real trial. passes_graduation={r['passes_graduation']}. "
                  f"years detail in {os.path.relpath(os.path.join(HERE, 'results.json'), REPO)}",
        )
        print(f"LOGGED {exp_id} (phase=result)")

    print(f"\nVerdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
