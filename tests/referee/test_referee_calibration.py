"""Referee power/size calibration through the FULL battery (mission §3.3d:
synthetic no-edge and known-edge panels). Acceptance criteria are
pre-registered in specs/REFEREE-SELFTEST-2026-07-08.md as amended by
specs/REFEREE-SELFTEST-2026-07-08-v2.md (S2 grid + effective-N DSR; logged
operator approval) and specs/REFEREE-SELFTEST-2026-07-09-v3.md (per-gate S1
null assertions S1c/S1d/S1e; operator approval logged as decision D20) and
may not be adjusted to pass. All trials land in throwaway ledgers (never the
real one — synthetic calibration does not feed the DSR count, decision D14)."""
import json

import numpy as np
import pandas as pd
import pytest

import validation.ledger as vledger
from validation.run_battery import run_battery
from validation.synthetic import null_panel, planted_signal_panel

MODULE = "tests.referee.test_referee_calibration"
N_DAYS, N_ASSETS = 1750, 40
# S1 variants: (panel seed, momentum lookback), fixed a priori.
NULL_VARIANTS = [(101, 20), (101, 60), (101, 120), (101, 250),
                 (202, 60), (202, 120)]
# S2 signal-noise multipliers (quality gradient), clean variant LAST so the
# PBO trial matrix holds all 12 columns when it is evaluated. v2-amended grid,
# selected by the pre-registration power analysis (dry-run DSR 0.9968).
NOISE_GRID = [4, 3, 2.5, 2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0.0]
PLANTED_IC, PLANTED_SEED = 0.10, 42


def _rank_weights(sig: pd.DataFrame) -> pd.DataFrame:
    """Dollar-neutral, gross-1 rank weights — the M0-style toy book."""
    ranks = sig.rank(axis=1)
    z = ranks.sub(ranks.mean(axis=1), axis=0)
    gross = z.abs().sum(axis=1)
    return z.div(gross.where(gross > 0), axis=0).fillna(0.0)


def momentum_provider(prices, config):
    """No-true-edge provider: momentum computed FROM null-panel prices."""
    lb = int(config["lookback"])
    weekly = prices.index[::5]
    mom = (prices / prices.shift(lb) - 1.0).reindex(weekly)
    return {"weights": _rank_weights(mom), "signal": mom}


def jittered_signal_provider(prices, config):
    """Known-edge provider: planted signal + config-scaled noise (read from
    disk so the importlib-loaded module copy needs no shared globals)."""
    sig = pd.read_csv(config["signal_csv"], index_col=0, parse_dates=True)
    rng = np.random.default_rng(int(config["seed"]))
    m = float(config["noise_mult"])
    noisy = sig + m * float(sig.stack().std()) * pd.DataFrame(
        rng.normal(0, 1, sig.shape), index=sig.index, columns=sig.columns)
    weekly = prices.index[::5]
    return {"weights": _rank_weights(noisy.reindex(weekly)),
            "signal": noisy.reindex(weekly)}


@pytest.fixture()
def throwaway_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(vledger, "LEDGER", str(tmp_path / "ledger.jsonl"))


def test_size_null_panels_pass_nothing(tmp_path, throwaway_ledger):
    """S1: the full battery must reject momentum mined from pure noise."""
    results = []
    for i, (seed, lb) in enumerate(NULL_VARIANTS):
        prices = (1 + null_panel(N_DAYS, N_ASSETS, seed=seed)).cumprod()
        res = run_battery(f"{MODULE}:momentum_provider",
                          {"lookback": lb, "panel_seed": seed}, prices,
                          str(tmp_path / "null_runs" / f"v{i}"), f"NULL-{i}")
        results.append(res)
    fails = [json.dumps(r["gates"], default=str) for r in results]
    assert all(not r["all_gates_passed"] for r in results), fails      # S1a
    assert sum(r["gates"]["net_sharpe"] for r in results) <= 1, fails  # S1b
    # v3 spec (operator approval D20): rejection may not hinge on the
    # single-trial-store PBO leg, and the size gates hold per-gate.
    assert all(not all(v for k, v in r["gates"].items() if k != "pbo")
               for r in results), fails                                # S1c
    assert sum(r["gates"]["dsr"] for r in results) == 0, fails         # S1d
    assert sum(r["gates"]["survives_10bps"] for r in results) <= 1, fails  # S1e
    assert len(vledger.read_ledger()) == len(NULL_VARIANTS)            # S3


def test_power_planted_signal_recovered(tmp_path, throwaway_ledger):
    """S2: the full battery must pass the clean planted-signal variant."""
    rets, sig = planted_signal_panel(N_DAYS, N_ASSETS, seed=PLANTED_SEED,
                                     ic=PLANTED_IC)
    prices = (1 + rets).cumprod()
    sig_csv = tmp_path / "planted_signal.csv"
    sig.to_csv(sig_csv)
    results = []
    for i, m in enumerate(NOISE_GRID):
        cfg = {"signal_csv": str(sig_csv), "noise_mult": m, "seed": 1000 + i}
        res = run_battery(f"{MODULE}:jittered_signal_provider", cfg, prices,
                          str(tmp_path / "planted_runs" / f"v{i}"), f"PLANT-{i}")
        results.append(res)
    assert NOISE_GRID[-1] == 0.0
    clean = results[-1]
    detail = json.dumps({"gates": clean["gates"], "pbo": clean["pbo"],
                         "dsr": clean["dsr"]["dsr"],
                         "net_5bps": clean["performance"]["5bps"]["net"]["ann_sharpe"]},
                        default=str)
    assert clean["all_gates_passed"], detail                            # S2a
    assert clean["pbo"]["pbo"] is not None and clean["pbo"]["pbo"] <= 0.2, detail  # S2b
    assert len(vledger.read_ledger()) == len(NOISE_GRID)               # S3
