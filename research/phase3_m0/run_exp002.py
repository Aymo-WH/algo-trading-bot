"""EXP-002 runner (specs/EXP-002-phase3-m0-combiner.md): invoke the referee's
run_battery against the M0 provider on the live-tradable train/validation
window (1999-12-21 buffer day .. 2021-12-31). Train/validation ONLY.

v2: re-run after a fresh-context reviewer (REQUEST-CHANGES) found real bugs
in the v1 implementation (NaN-mask misalignment in neutralize_against_beta;
an unsanctioned no-trade-band rescale that broke position/category caps).
Construction intent is unchanged from the frozen spec; only implementation
bugs were fixed and the no-trade band was descoped to Phase 4 (decisions.md
D26). v1's ledger row stays -- it was a real execution on real data and is
not deleted, per the "log every trial" discipline.

Run: /workspace/venv/bin/python research/phase3_m0/run_exp002.py
Output: research/phase3_m0/battery_out/battery_result.json (referee-written),
research/phase3_m0/battery_out/diagnostics.json (this script; the realized
max position, max category exposure, net dollar exposure, and ex-ante
portfolio beta the spec promised as diagnostics).
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

from validation.run_battery import run_battery
from src.portfolio_m0 import (CATEGORY_CAP_FRAC, GROSS, POSITION_CAP_FRAC,
                              compute_signal_and_weights, equal_weight_market_return,
                              load_canonical_trainval, low_beta_s3, m0_provider,
                              rebalance_dates)

HERE = os.path.dirname(os.path.abspath(__file__))


def compute_diagnostics() -> dict:
    """The spec's promised diagnostics (max position, max category exposure,
    net dollar exposure, ex-ante portfolio beta) -- computed directly, not
    through run_battery (which has no channel for them)."""
    close, elig, categories, manifest = load_canonical_trainval()
    rebal = rebalance_dates(close.index)
    window = rebal[(rebal >= manifest["panel_start"]) & (rebal <= manifest["trainval_end"])]
    weights, signal = compute_signal_and_weights(close, elig, categories, window)

    ret = close.pct_change(fill_method=None)
    market = equal_weight_market_return(ret, elig)
    beta = -low_beta_s3(close, elig, market=market).reindex(window)

    gross = weights.abs().sum(axis=1)
    net = weights.sum(axis=1)
    pos_cap = POSITION_CAP_FRAC * GROSS
    cat_cap = CATEGORY_CAP_FRAC * GROSS

    max_cat_exposure = 0.0
    n_cat_breaches = 0
    for _, row in weights.iterrows():
        r = row.dropna()
        cats = categories.reindex(r.index)
        for c in cats.dropna().unique():
            idx = cats[cats == c].index
            g = float(r[idx].abs().sum())
            max_cat_exposure = max(max_cat_exposure, g)
            if g > cat_cap + 1e-6:
                n_cat_breaches += 1

    ex_ante_beta = (weights * beta.reindex(columns=weights.columns)).sum(axis=1)

    return {
        "gross": {"min": float(gross.min()), "max": float(gross.max()), "target": GROSS},
        "net_dollar_exposure": {"max_abs": float(net.abs().max()),
                               "mean_abs": float(net.abs().mean())},
        "position_cap": {"cap": pos_cap, "max_abs_position": float(weights.abs().max().max()),
                         "n_breaches_gt_1e-6": int((weights.abs() > pos_cap + 1e-6).sum().sum())},
        "category_cap": {"cap": cat_cap, "max_category_exposure": max_cat_exposure,
                         "n_breaches_gt_1e-6": n_cat_breaches},
        "ex_ante_portfolio_beta": {"max_abs": float(ex_ante_beta.abs().max()),
                                  "mean_abs": float(ex_ante_beta.abs().mean())},
        "n_rebalance_dates": int(len(weights)),
    }


def live_window_prices() -> pd.DataFrame:
    """Train/validation close panel sliced to the live-tradable window (one
    buffer day before panel_start, so the first pct_change is well-defined,
    through trainval_end) -- so performance is measured over the period the
    strategy could actually trade, not diluted by the pre-panel_start years
    where eligibility never reaches MIN_ELIGIBLE. A deterministic slice of
    the already-committed data/panel/trainval_close.csv -- computed here
    in-memory rather than persisted as a second multi-MB copy on disk."""
    close, elig, categories, manifest = load_canonical_trainval()
    panel_start = pd.Timestamp(manifest["panel_start"])
    pos = close.index.searchsorted(panel_start)
    buffer_start = close.index[pos - 1]
    return close.loc[buffer_start:manifest["trainval_end"]]


def main():
    prices = live_window_prices()
    config = {"spec": "specs/EXP-002-phase3-m0-combiner.md", "version": "v2",
             "gross": 2.0, "position_cap_frac": 0.10, "category_cap_frac": 0.40,
             "clip": 2.5, "no_trade_band": "deferred to Phase 4, see decisions.md D26"}
    result = run_battery(
        provider_spec="src.portfolio_m0:m0_provider",
        config=config, prices=prices,
        out_dir=os.path.join(HERE, "battery_out"),
        experiment_id="PHASE3-M0-v2",
    )
    diagnostics = compute_diagnostics()
    with open(os.path.join(HERE, "battery_out", "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    print("gates:", result["gates"])
    print("all_gates_passed:", result["all_gates_passed"])
    print("cpcv:", result["cpcv"])
    print("dsr:", result["dsr"])
    print("pbo:", result["pbo"])
    print("canaries all_passed:", result["canaries"]["all_passed"])
    print("diagnostics:", json.dumps(diagnostics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
