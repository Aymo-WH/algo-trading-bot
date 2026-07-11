"""EXP-003 runner (specs/EXP-003-s1-s2-timing-decomposition.md): split S1/S2
into a static (expanding per-asset mean) component and a timing (residual)
component, grade the TIMING component against the unchanged Tier-1
graduation rule. Train/validation ONLY -- data/panel never touches the
holdout/lockbox. Static components and canaries are reported descriptively/
diagnostically (D30) and do NOT gate pass/fail -- see the frozen spec.

Logs exactly one real ledger row per timing component (phase="result") --
guarded so a re-run for reproducibility verification (mission Sec3.3b) does
not silently double-count trials against the 250 budget.

Run: /workspace/venv/bin/python research/exp003_timing_decomposition/run_exp003.py
Output: research/exp003_timing_decomposition/results.json (committed evidence).
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

from src.decomposition import MIN_PRIOR_OBS, expanding_static_timing
from src.panel_factory import PANEL_DIR, TRAINVAL_END, rebalance_dates
from src.signals import momentum_s1, trend_s2
from validation.canaries import run_signal_canaries
from validation.ic import cross_sectional_ic, ic_summary, passes_graduation
from validation.ledger import log_trial, read_ledger

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_START = "1999-12-22"    # same boundary EXP-001 used (data/panel/MANIFEST.json)

RAW_SIGNAL_FNS = {
    "S1_momentum": momentum_s1,
    "S2_ts_trend": trend_s2,
}


def already_logged() -> set:
    return {r["id"] for r in read_ledger()
            if r.get("phase") == "result" and str(r.get("id", "")).startswith("EXP-003-")}


def main():
    close = pd.read_csv(os.path.join(PANEL_DIR, "trainval_close.csv"),
                        index_col=0, parse_dates=True).sort_index()
    elig = pd.read_csv(os.path.join(PANEL_DIR, "eligibility_trainval.csv"),
                       index_col=0, parse_dates=True).sort_index().astype(bool)
    assert close.index.max() == pd.Timestamp(TRAINVAL_END), "wrong trainval panel"

    fwd5 = close.shift(-5) / close - 1.0
    rebal = rebalance_dates(close.index)
    window = rebal[(rebal >= PANEL_START) & (rebal <= TRAINVAL_END)]

    results = {}
    static_reports = {}
    canary_reports = {}
    for raw_name, fn in RAW_SIGNAL_FNS.items():
        raw = fn(close, elig)
        weekly = raw.reindex(window)
        static, timing = expanding_static_timing(weekly, min_prior_obs=MIN_PRIOR_OBS)

        # --- confirmatory: timing component vs the unchanged graduation rule ---
        timing_name = f"{raw_name.split('_')[0]}_timing"   # S1_timing / S2_timing
        ic_timing = cross_sectional_ic(timing, fwd5, min_names=10)
        summary_timing = ic_summary(ic_timing, nw_lags=2)
        passed = passes_graduation(summary_timing)
        results[timing_name] = {**summary_timing, "passes_graduation": bool(passed)}
        print(json.dumps({"signal": timing_name,
                          **{k: v for k, v in summary_timing.items() if k != "years"},
                          "passes_graduation": passed}, default=str))

        # --- descriptive only: static component's own IC (design.md-style, not a trial) ---
        ic_static = cross_sectional_ic(static, fwd5, min_names=10)
        summary_static = ic_summary(ic_static, nw_lags=2)
        static_reports[f"{raw_name.split('_')[0]}_static"] = \
            {k: v for k, v in summary_static.items() if k != "years"}

        # --- diagnostic only (D30): canaries on the timing component, non-blocking ---
        canary_reports[timing_name] = run_signal_canaries(timing, fwd5)

    # Descriptive only (mirrors EXP-001's inter-signal-correlation treatment).
    n_pass = sum(r["passes_graduation"] for r in results.values())
    verdict = {
        "hypothesis": "both_timing_components_null" if n_pass == 0 else "timed_information_present",
        "n_timing_passed": n_pass, "n_timing_total": len(results),
    }
    out = {
        "spec": "specs/EXP-003-s1-s2-timing-decomposition.md",
        "min_prior_obs": MIN_PRIOR_OBS,
        "data_window": [str(window.min().date()), str(window.max().date())],
        "n_rebalance_dates": int(len(window)),
        "results_timing_confirmatory": results,
        "results_static_descriptive": static_reports,
        "canaries_diagnostic": canary_reports,
        "verdict": verdict,
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    done = already_logged()
    for name, r in results.items():
        exp_id = f"EXP-003-{name}"
        if exp_id in done:
            print(f"SKIP logging {exp_id}: already present in the real ledger "
                  f"(re-run for verification only, not a new trial)")
            continue
        log_trial(
            experiment_id=exp_id, phase="result",
            config={"spec": "specs/EXP-003-s1-s2-timing-decomposition.md",
                    "signal": name, "min_prior_obs": MIN_PRIOR_OBS,
                    "data_window": out["data_window"]},
            metrics={k: v for k, v in r.items() if k != "years"},
            hypothesis=f"{name} clears mean_ic>=0.01 AND t_nw>=2.0 AND "
                       f"pct_years_positive>=0.60 (same rule as EXP-001/design.md S5, "
                       f"applied to the timing residual per specs/EXP-003)",
            notes=f"EXP-003 real trial. passes_graduation={r['passes_graduation']}. "
                  f"Canaries diagnostic-only (D30), static component descriptive-only "
                  f"(not a trial). Full detail in "
                  f"{os.path.relpath(os.path.join(HERE, 'results.json'), REPO)}",
        )
        print(f"LOGGED {exp_id} (phase=result)")

    print(f"\nVerdict: {verdict}")
    print(f"\nStatic components (descriptive, not graded): "
          f"{json.dumps(static_reports, default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
