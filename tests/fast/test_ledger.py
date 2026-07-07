"""Ledger integrity: append-only, complete provenance, correct counting."""
import json

import validation.ledger as ledger_mod
from validation.ledger import config_hash, log_trial, read_ledger, trial_count


def test_log_trial_appends_and_reads_back(tmp_path, monkeypatch):
    path = tmp_path / "experiments.jsonl"
    monkeypatch.setattr(ledger_mod, "LEDGER", str(path))
    row1 = log_trial(experiment_id="EXP-000", phase="trial",
                     config={"a": 1}, metrics={"sharpe": 0.1}, seed=42)
    row2 = log_trial(experiment_id="EXP-000", phase="result",
                     config={"a": 1}, metrics={"sharpe": 0.1}, seed=42)
    rows = read_ledger()
    assert len(rows) == 2
    assert rows[0]["id"] == "EXP-000"
    assert rows[0]["config_hash"] == row1["config_hash"] == row2["config_hash"]
    assert rows[0]["commit"]  # provenance present
    assert rows[0]["ts"]
    # append-only: second write did not clobber the first
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2 and json.loads(lines[0])["phase"] == "trial"


def test_trial_count_counts_evaluations_not_preregistrations(tmp_path, monkeypatch):
    path = tmp_path / "experiments.jsonl"
    monkeypatch.setattr(ledger_mod, "LEDGER", str(path))
    log_trial(experiment_id="EXP-001", phase="preregistered", config={})
    log_trial(experiment_id="EXP-001", phase="trial", config={"x": 1})
    log_trial(experiment_id="EXP-001", phase="trial", config={"x": 2})
    log_trial(experiment_id="EXP-001", phase="result", config={"x": 2})
    assert trial_count() == 3  # 2 trials + 1 result; preregistration is not an evaluation


def test_config_hash_is_order_invariant():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
