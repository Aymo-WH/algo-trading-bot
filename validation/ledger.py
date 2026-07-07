"""Append-only experiment ledger (mission §3.4).

Every configuration evaluated — including discards and ablations — is one row in
research/experiments.jsonl. The row count feeds the Deflated Sharpe correction;
an incomplete ledger makes DSR wrong and any "edge" an artifact. Backtest entry
points call log_trial() unconditionally, so nothing runs uncounted.
"""
import datetime
import hashlib
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(REPO, "research", "experiments.jsonl")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


def log_trial(*, experiment_id: str, phase: str, config: dict,
              metrics: dict | None = None, hypothesis: str | None = None,
              seed: int | None = None, notes: str | None = None) -> dict:
    """Append one trial row. Returns the row. Never overwrites, never deletes."""
    row = {
        "id": experiment_id,
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phase": phase,                 # preregistered | trial | result | canary
        "commit": _git_commit(),
        "config_hash": config_hash(config),
        "config": config,
        "seed": seed,
        "hypothesis": hypothesis,
        "metrics": metrics,
        "notes": notes,
    }
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def read_ledger() -> list[dict]:
    if not os.path.exists(LEDGER):
        return []
    rows = []
    with open(LEDGER) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def trial_count(phases: tuple = ("trial", "result")) -> int:
    """Count of logged evaluations — the N that feeds DSR."""
    return sum(1 for r in read_ledger() if r.get("phase") in phases)
