"""The referee battery: walk-forward -> cost grid -> CPCV -> PBO -> DSR -> canaries.

Usage:
    /workspace/venv/bin/python -m validation.run_battery \
        --provider <module:function> --config <json path> --out research/runs/<name>/

The provider contract (strategy side, lives OUTSIDE validation/):
    provider(prices: pd.DataFrame, config: dict) -> {"weights": pd.DataFrame,
                                                     "signal": pd.DataFrame}
    - prices: daily adjusted close panel (train/validation dates ONLY — this
      module never sees the lockbox).
    - weights.loc[t] / signal.loc[t] must be computable from data <= t; the
      backtester additionally shifts weights by one bar, and the leak-hunter
      audits providers separately.

Gates applied are the frozen contract (specs/DESIGN-v2-2026-07-07.md). Every
invocation logs trials to the ledger.
"""
import argparse
import importlib
import json
import os

import numpy as np
import pandas as pd

from validation.backtest import backtest_panel, performance_summary
from validation.canaries import run_signal_canaries
from validation.cscv import cscv_pbo
from validation.ledger import log_trial, read_ledger
from validation.metrics import dsr_from_ledger, mean_pairwise_correlation, sharpe
from validation.splits import cpcv_splits

GATES = {  # frozen contract — do not edit (guard-enforced once referee freezes)
    "min_net_sharpe": 0.5,
    "max_drawdown": -0.20,
    "min_years_positive": 0.5,
    "max_pbo": 0.5,
    "min_dsr": 0.95,
    "cost_survival_bps": 10.0,
}


def _load_provider(spec: str):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def load_trial_matrix(returns_store: str):
    """T×N matrix of persisted per-trial daily net returns (inner-joined).

    Shared input for the CSCV/PBO sweep AND the effective-N (ρ̄) correction
    in DSR (specs/REFEREE-SELFTEST-2026-07-08-v2.md, amendment 1). Returns
    None when fewer than 2 trials have been stored.
    """
    if not os.path.isdir(returns_store):
        return None
    files = sorted(os.listdir(returns_store))
    if len(files) < 2:
        return None
    cols = [pd.read_csv(os.path.join(returns_store, f), index_col=0).iloc[:, 0]
            for f in files]
    return pd.concat(cols, axis=1, join="inner").values


def run_battery(provider_spec: str, config: dict, prices: pd.DataFrame,
                out_dir: str, experiment_id: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    provider = _load_provider(provider_spec)
    out = provider(prices, config)
    weights, signal = out["weights"], out.get("signal")
    rets = prices.pct_change(fill_method=None)

    # 1. Full-period walk-forward performance at the cost grid
    perf = {}
    for bps in (5.0, 10.0, 20.0):
        port = backtest_panel(weights, rets, cost_bps_per_side=bps)
        perf[f"{bps:g}bps"] = performance_summary(port)
    port5 = backtest_panel(weights, rets, cost_bps_per_side=5.0)

    # 2. CPCV-split Sharpe distribution of the DELIVERED net return series
    #    (5 bps). Honest semantics: this measures sub-period stability of the
    #    weights the provider handed us; it does NOT refit the model per split.
    #    PIT/refit discipline is the provider's responsibility (walk-forward
    #    refits inside the provider), audited separately by the leak-hunter.
    #    For M0 (static signal blend, nothing fitted) the two coincide.
    cpcv_sharpes = []
    net = port5["net"].values
    for _, test_idx in cpcv_splits(len(net), n_groups=8, k_test=2, purge=5, embargo=10):
        cpcv_sharpes.append(sharpe(net[test_idx]) * np.sqrt(252))
    cpcv_stats = {"median": float(np.nanmedian(cpcv_sharpes)),
                  "q05": float(np.nanquantile(cpcv_sharpes, 0.05)),
                  "q95": float(np.nanquantile(cpcv_sharpes, 0.95)),
                  "n_splits": len(cpcv_sharpes)}

    # 3. Log this trial FIRST — DSR below uses the whole ledger including it.
    log_trial(experiment_id=experiment_id, phase="trial", config=config,
              metrics={"net_sharpe_5bps": perf["5bps"]["net"]["ann_sharpe"],
                       "net_sharpe_10bps": perf["10bps"]["net"]["ann_sharpe"],
                       "cpcv": cpcv_stats},
              notes=f"battery run -> {out_dir}")

    # 4. Persist this trial's daily net returns. The store is the trial
    #    RETURN matrix feeding BOTH the CSCV/PBO sweep and the effective-N
    #    (ρ̄) correction in DSR.
    returns_store = os.path.join(os.path.dirname(out_dir.rstrip("/")), "_trial_returns")
    os.makedirs(returns_store, exist_ok=True)
    port5["net"].to_csv(os.path.join(returns_store, f"{experiment_id}.csv"))
    M = load_trial_matrix(returns_store)
    n_stored = len(os.listdir(returns_store))

    # 5. DSR over the whole ledger, deflated by the EFFECTIVE trial count
    #    (N̂ = ρ̄ + (1−ρ̄)·M — BLdP; amendment 1 of the v2 self-test spec).
    ledger_rows = [r for r in read_ledger() if r.get("phase") in ("trial", "result")
                   and (r.get("metrics") or {}).get("net_sharpe_5bps") is not None]
    trial_sharpes = np.array([r["metrics"]["net_sharpe_5bps"] / np.sqrt(252)
                              for r in ledger_rows])
    dsr_out = dsr_from_ledger(port5["net"].values, trial_sharpes,
                              trial_corr_mean=mean_pairwise_correlation(M))

    # 6. CSCV/PBO on the same matrix (meaningless below ~10 columns).
    if M is not None and n_stored >= 10 and M.shape[0] >= 32:
        pbo_out = cscv_pbo(M, S=16)
    else:
        pbo_out = {"pbo": None,
                   "note": f"needs >=10 stored trials with T>=32, have {n_stored}"}

    # 7. Canaries on the signal panel
    canary_out = (run_signal_canaries(signal, prices.shift(-5) / prices - 1.0)
                  if signal is not None else {"all_passed": None})

    # 8. Gates
    g = perf["5bps"]["net"]
    gates = {
        "net_sharpe": g["ann_sharpe"] >= GATES["min_net_sharpe"],
        "max_drawdown": g["max_drawdown"] >= GATES["max_drawdown"],
        "years_positive": g["pct_years_positive"] > GATES["min_years_positive"],
        "survives_10bps": perf["10bps"]["net"]["ann_sharpe"] >= GATES["min_net_sharpe"],
        "pbo": (pbo_out["pbo"] is not None and pbo_out["pbo"] < GATES["max_pbo"]),
        "dsr": dsr_out["dsr"] > GATES["min_dsr"],
        "canaries": bool(canary_out["all_passed"]),
    }
    result = {"experiment_id": experiment_id, "performance": perf,
              "cpcv": cpcv_stats, "pbo": pbo_out, "dsr": dsr_out,
              "canaries": canary_out, "gates": gates,
              "all_gates_passed": all(gates.values()),
              "audit_trigger_sharpe_gt_1.2": g["ann_sharpe"] > 1.2}
    with open(os.path.join(out_dir, "battery_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--prices", required=True, help="CSV of train/val close panel")
    p.add_argument("--out", required=True)
    p.add_argument("--id", required=True)
    a = p.parse_args()
    config = json.load(open(a.config))
    prices = pd.read_csv(a.prices, index_col=0, parse_dates=True)
    res = run_battery(a.provider, config, prices, a.out, a.id)
    print(json.dumps({k: res[k] for k in ("gates", "all_gates_passed", "cpcv")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
