"""M4 pre-registration dry-run: per-gate outcomes of the 6 S1 null variants.

Precedent: the D15 power analysis — a backtest-only dry-run on COMMITTED,
FROZEN referee code, run BEFORE freezing a spec, so the spec's assertion
values are measured rather than intuited. Synthetic null panels only;
throwaway ledger and runs-store (decision D14) — nothing here touches the
real ledger, real data, or the holdout.

Replicates tests/referee/test_referee_calibration.py::test_size_null_panels_pass_nothing
exactly (same MODULE providers, same NULL_VARIANTS, same panel construction),
and dumps every variant's full per-gate dict + key metrics to JSON.

Run (from repo root):
    /workspace/venv/bin/python research/m4_dryrun/m4_null_gate_analysis.py
Output: research/m4_dryrun/m4_null_gate_results.json (committed evidence;
        first produced 2026-07-09, exit 0, results byte-identical to the
        Phase-0 record for the overlapping values — v3 net Sharpe +0.7226).
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
os.chdir(REPO)

import validation.ledger as vledger
from validation.run_battery import run_battery
from validation.synthetic import null_panel

from tests.referee.test_referee_calibration import (  # byte-identical providers
    MODULE, N_ASSETS, N_DAYS, NULL_VARIANTS)

OUT = os.path.join(HERE, "m4_null_gate_results.json")
WORK = tempfile.mkdtemp(prefix="m4_dryrun_")           # throwaway (D14)
vledger.LEDGER = os.path.join(WORK, "throwaway_ledger.jsonl")

rows = []
for i, (seed, lb) in enumerate(NULL_VARIANTS):
    prices = (1 + null_panel(N_DAYS, N_ASSETS, seed=seed)).cumprod()
    res = run_battery(f"{MODULE}:momentum_provider",
                      {"lookback": lb, "panel_seed": seed}, prices,
                      os.path.join(WORK, "null_runs", f"v{i}"),
                      f"M4-NULL-{i}")
    row = {
        "variant": i, "seed": seed, "lookback": lb,
        "all_gates_passed": res["all_gates_passed"],
        "gates": res["gates"],
        "net_sharpe_5bps": res["performance"]["5bps"]["net"]["ann_sharpe"],
        "net_sharpe_10bps": res["performance"]["10bps"]["net"]["ann_sharpe"],
        "pbo": res["pbo"].get("pbo"),
        "dsr": res["dsr"].get("dsr"),
        "canaries_all_passed": (res.get("canaries") or {}).get("all_passed"),
    }
    rows.append(row)
    print(json.dumps(row, default=str), flush=True)

with open(OUT, "w") as f:
    json.dump(rows, f, indent=2, default=str)
print(f"\nwrote {OUT}")
