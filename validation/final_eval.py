"""ONE-SHOT final holdout evaluation (mission §3.3h/§3.3i, Phase 6).

This is the ONLY code path allowed to see holdout-period strategy
performance, and it may run ONCE, ever. It is operator-triggered via the
/promote skill: the operator retrieves the token themselves (Claude is
hook-blocked from it) and runs:

    /workspace/venv/bin/python validation/final_eval.py \
        --provider <module:function> --config <json> \
        --prices <train_val_close_panel.csv> --operator-token <TOKEN>

Enforcement layers (scope stated honestly):
- cryptographic: the token IS the decryption key (validation/lockbox.py) —
  no token, no holdout data. Token custody is the operator's; the /promote
  procedure deletes the token file after the run, so a post-run replay has
  no key material to reuse.
- one-shot: refuses if a prior run's result bundle or an access-log RUN line
  exists — checked at BOTH the caller-supplied paths and the canonical repo
  paths, so custom out_dir/access_log arguments cannot dodge the markers of
  a completed real run ("You do not get a second holdout evaluation", §2);
- process: refuses unless the referee is frozen (validation/.frozen) and the
  trial ledger is non-empty (DSR at final eval needs the true trial count);
- audit: ATTEMPT and RUN lines land in research/lockbox_access.log (the RUN
  line write is mandatory — an unwritable audit log aborts); the run is
  logged to the ledger with phase="final".
This makes an accidental second evaluation impossible and a deliberate one
require visible, auditable steps (deleting markers / exfiltrating the token
before the operator destroys it) — the §3.3h design goal.

The verdict — PASS or FAIL — is final either way. A FAIL is reported with
the same fidelity as a PASS; there is no "fix and retry" on the holdout.
"""
import argparse
import hashlib
import io
import json
import os

import numpy as np
import pandas as pd

from validation.backtest import backtest_panel, performance_summary
from validation.canaries import run_signal_canaries
from validation.ledger import log_trial, read_ledger, trial_count
from validation.lockbox import ACCESS_LOG, LOCKBOX_DIR, _audit, open_lockbox
from validation.metrics import dsr_from_ledger, mean_pairwise_correlation
from validation.run_battery import GATES, _load_provider, load_trial_matrix

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "research", "final_eval")
RUNS_STORE = os.path.join(REPO, "research", "runs", "_trial_returns")
RESULT_NAME = "FINAL_EVAL_RESULT.json"

# Frozen quarantine boundary — specs/DESIGN-v2-2026-07-07.md (decision D8).
HOLDOUT_START = "2022-01-01"
HOLDOUT_END = "2026-06-30"


def _refuse(msg: str) -> None:
    raise RuntimeError(f"FINAL EVAL REFUSED: {msg}")


def _check_preconditions(out_dir: str, access_log: str,
                         require_frozen: bool) -> None:
    # One-shot markers: checked at the supplied paths AND the canonical repo
    # paths, so custom kwargs cannot dodge a completed real run's markers.
    for d in {out_dir, OUT_DIR}:
        result_path = os.path.join(d, RESULT_NAME)
        if os.path.exists(result_path):
            _refuse(f"{result_path} exists — the one-shot evaluation has "
                    "already run. There is no second holdout evaluation (§2).")
    for log in {access_log, ACCESS_LOG}:
        if os.path.exists(log):
            with open(log) as f:
                if any("FINAL_EVAL RUN" in line for line in f):
                    _refuse(f"{log} records a prior FINAL_EVAL RUN — this "
                            "runs once, ever.")
    if require_frozen and not os.path.exists(
            os.path.join(REPO, "validation", ".frozen")):
        _refuse("validation/.frozen missing — the referee must be frozen "
                "before the final evaluation (Phase 0 gate).")
    if trial_count() == 0:
        _refuse("experiment ledger is empty — a candidate cannot reach Phase 6 "
                "with zero logged trials, and DSR needs the true count.")


def _ledger_trial_sharpes() -> np.ndarray:
    rows = [r for r in read_ledger() if r.get("phase") in ("trial", "result")
            and (r.get("metrics") or {}).get("net_sharpe_5bps") is not None]
    return np.array([r["metrics"]["net_sharpe_5bps"] / np.sqrt(252)
                     for r in rows])


def run_final_eval(provider_spec: str, config: dict,
                   trainval_prices: pd.DataFrame, operator_token: str, *,
                   out_dir: str = OUT_DIR, lockbox_dir: str = LOCKBOX_DIR,
                   access_log: str = ACCESS_LOG, require_frozen: bool = True,
                   holdout_start: str = HOLDOUT_START,
                   holdout_end: str = HOLDOUT_END,
                   returns_store: str = RUNS_STORE) -> dict:
    _check_preconditions(out_dir, access_log, require_frozen)
    _audit(f"FINAL_EVAL ATTEMPT: provider={provider_spec}", access_log)

    holdout_bytes = open_lockbox(operator_token, lockbox_dir=lockbox_dir,
                                 access_log=access_log)
    holdout = pd.read_csv(io.BytesIO(holdout_bytes), index_col=0,
                          parse_dates=True).sort_index()

    # Data sanity — refuse on any boundary violation rather than mis-measure.
    hs = pd.Timestamp(holdout_start)
    if trainval_prices.index.max() >= holdout.index.min():
        _refuse("train/val panel overlaps the holdout slice — boundary "
                "violation, investigate before re-attempting.")
    if trainval_prices.index.max() >= hs:
        _refuse(f"train/val panel extends past the frozen holdout start {hs.date()}.")
    if abs((holdout.index.min() - hs).days) > 7:
        _refuse(f"decrypted holdout starts {holdout.index.min().date()}, expected "
                f"~{hs.date()} — wrong slice in the lockbox.")
    if abs((holdout.index.max() - pd.Timestamp(holdout_end)).days) > 7:
        _refuse(f"decrypted holdout ends {holdout.index.max().date()}, expected "
                f"~{pd.Timestamp(holdout_end).date()} — truncated or wrong "
                "slice in the lockbox.")
    if set(trainval_prices.columns) != set(holdout.columns):
        _refuse("train/val and holdout column sets differ — a mismatched "
                "universe would silently mis-measure via NaN-filled returns.")

    full = pd.concat([trainval_prices, holdout]).sort_index()
    full = full[~full.index.duplicated(keep="first")]

    out = _load_provider(provider_spec)(full, config)
    weights, signal = out["weights"], out.get("signal")
    rets = full.pct_change(fill_method=None)
    h0 = holdout.index.min()

    perf = {}
    for bps in (5.0, 10.0, 20.0):
        port = backtest_panel(weights, rets, cost_bps_per_side=bps)
        perf[f"{bps:g}bps"] = performance_summary(port.loc[port.index >= h0])
    port5 = backtest_panel(weights, rets, cost_bps_per_side=5.0)
    hold5 = port5.loc[port5.index >= h0]

    # DSR deflated by the EFFECTIVE trial count — ρ̄ from the same persisted
    # trial return matrix the validation-time PBO used (v2 spec, amendment 1).
    rho = mean_pairwise_correlation(load_trial_matrix(returns_store))
    dsr_out = dsr_from_ledger(hold5["net"].values, _ledger_trial_sharpes(),
                              trial_corr_mean=rho)

    canary_out = {"all_passed": None, "note": "no signal panel provided"}
    if signal is not None:
        fwd = (full.shift(-5) / full - 1.0)
        sig_h = signal.loc[signal.index >= h0]
        canary_out = run_signal_canaries(sig_h, fwd.loc[fwd.index >= h0])

    g = perf["5bps"]["net"]
    gates = {
        "net_sharpe": g["ann_sharpe"] >= GATES["min_net_sharpe"],
        "max_drawdown": g["max_drawdown"] >= GATES["max_drawdown"],
        "years_positive": g["pct_years_positive"] > GATES["min_years_positive"],
        "survives_10bps": perf["10bps"]["net"]["ann_sharpe"] >= GATES["min_net_sharpe"],
        "dsr": dsr_out["dsr"] > GATES["min_dsr"],
        "canaries": bool(canary_out["all_passed"]),
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "verdict": verdict, "gates": gates, "performance": perf,
        "dsr": dsr_out, "canaries": canary_out,
        "provider": provider_spec, "config": config,
        "holdout_span": [str(holdout.index.min().date()),
                         str(holdout.index.max().date())],
        "n_ledger_trials": int(trial_count()),
        "holdout_sha256": hashlib.sha256(holdout_bytes).hexdigest(),
        "note_pbo": "PBO is a train/val construct over the trial matrix; it "
                    "gated holdout ELIGIBILITY at validation time and is not "
                    "recomputable on the single holdout run.",
        "audit_trigger_sharpe_gt_1.5_presumed_bug": g["ann_sharpe"] > 1.5,
    }

    # Write order matters: the one-shot marker (result bundle) FIRST, so a
    # crash mid-write can never leave holdout P&L on disk with a retry open.
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, RESULT_NAME), "w") as f:
        json.dump(result, f, indent=2, default=str)
    hold5["net"].to_csv(os.path.join(out_dir, "holdout_net_returns_5bps.csv"))
    _audit(f"FINAL_EVAL RUN: verdict={verdict} provider={provider_spec} "
           f"net_sharpe_5bps={g['ann_sharpe']:.4f} -> {out_dir}", access_log,
           critical=True)
    log_trial(experiment_id="FINAL_EVAL", phase="final", config=config,
              metrics={"verdict": verdict, "gates": gates,
                       "net_sharpe_5bps": g["ann_sharpe"],
                       "net_sharpe_10bps": perf["10bps"]["net"]["ann_sharpe"]},
              notes=f"one-shot holdout evaluation -> {out_dir}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--prices", required=True,
                   help="CSV close panel, train/validation dates ONLY")
    p.add_argument("--operator-token", required=True)
    p.add_argument("--out", default=OUT_DIR)
    a = p.parse_args()
    config = json.load(open(a.config))
    prices = pd.read_csv(a.prices, index_col=0, parse_dates=True).sort_index()
    result = run_final_eval(a.provider, config, prices, a.operator_token,
                            out_dir=a.out)
    print(json.dumps({k: result[k] for k in
                      ("verdict", "gates", "holdout_span", "n_ledger_trials")},
                     indent=2, default=str))
    print(f"\nFINAL VERDICT: {result['verdict']} — this was the one and only "
          "holdout evaluation. The result stands either way.")


if __name__ == "__main__":
    main()
