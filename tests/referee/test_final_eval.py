"""Final-eval one-shot mechanics — pre-registered S4c/S4d in
specs/REFEREE-SELFTEST-2026-07-08.md. All runs on synthetic data with
throwaway lockboxes/ledgers; the real quarantine is never touched."""
import json
import os

import pandas as pd
import pytest

import validation.final_eval as fe
import validation.ledger as vledger
from validation.final_eval import RESULT_NAME, run_final_eval
from validation.ledger import log_trial, read_ledger
from validation.lockbox import build_lockbox
from validation.synthetic import null_panel

PROVIDER = "tests.referee.test_final_eval:momentum_provider"


def momentum_provider(prices, config):
    """Pure toy provider: weekly cross-sectional momentum rank weights."""
    lb = int(config["lookback"])
    weekly = prices.index[::5]
    mom = (prices / prices.shift(lb) - 1.0).reindex(weekly)
    ranks = mom.rank(axis=1)
    z = ranks.sub(ranks.mean(axis=1), axis=0)
    gross = z.abs().sum(axis=1)
    w = z.div(gross.where(gross > 0), axis=0).fillna(0.0)
    return {"weights": w, "signal": mom}


def _setup(tmp_path, monkeypatch, name="a", seed_trial=True):
    d = tmp_path / name
    d.mkdir()
    prices = (1 + null_panel(360, 12, seed=7)).cumprod()
    trainval, holdout = prices.iloc[:250], prices.iloc[250:]
    build_lockbox(holdout.to_csv().encode(), lockbox_dir=str(d / "lb"),
                  token_path=str(d / "tok.txt"), access_log=str(d / "access.log"))
    token = (d / "tok.txt").read_text().strip()
    monkeypatch.setattr(vledger, "LEDGER", str(d / "ledger.jsonl"))
    if seed_trial:
        log_trial(experiment_id="seed-trial", phase="trial", config={"x": 1},
                  metrics={"net_sharpe_5bps": 0.3})
    kw = dict(out_dir=str(d / "out"), lockbox_dir=str(d / "lb"),
              access_log=str(d / "access.log"), require_frozen=False,
              holdout_start=str(holdout.index.min().date()),
              holdout_end=str(holdout.index.max().date()),
              returns_store=str(d / "store"))    # isolation: never the real one
    return trainval, holdout, token, kw, d


def test_happy_path_writes_bundle_then_one_shot_refusal(tmp_path, monkeypatch):
    trainval, holdout, token, kw, d = _setup(tmp_path, monkeypatch)
    res = run_final_eval(PROVIDER, {"lookback": 20}, trainval, token, **kw)
    assert res["verdict"] in ("PASS", "FAIL")          # honest either way
    bundle = json.load(open(os.path.join(kw["out_dir"], RESULT_NAME)))
    assert bundle["verdict"] == res["verdict"]
    rets = pd.read_csv(os.path.join(kw["out_dir"], "holdout_net_returns_5bps.csv"),
                       index_col=0, parse_dates=True)
    assert rets.index.min() >= holdout.index.min()     # holdout dates only
    log = (d / "access.log").read_text()
    assert "FINAL_EVAL ATTEMPT" in log and "FINAL_EVAL RUN" in log
    assert any(r["phase"] == "final" for r in read_ledger())
    with pytest.raises(RuntimeError, match="already run|once"):
        run_final_eval(PROVIDER, {"lookback": 20}, trainval, token, **kw)


def test_access_log_run_line_alone_blocks(tmp_path, monkeypatch):
    trainval, _, token, kw, d = _setup(tmp_path, monkeypatch, name="b")
    with open(d / "access.log", "a") as f:
        f.write("2026-01-01T00:00:00 FINAL_EVAL RUN: verdict=FAIL (prior)\n")
    with pytest.raises(RuntimeError, match="once, ever"):
        run_final_eval(PROVIDER, {"lookback": 20}, trainval, token, **kw)


def test_wrong_token_refused_and_nothing_written(tmp_path, monkeypatch):
    trainval, _, _, kw, d = _setup(tmp_path, monkeypatch, name="c")
    with pytest.raises(PermissionError):
        run_final_eval(PROVIDER, {"lookback": 20}, trainval, "not-the-token", **kw)
    assert not os.path.exists(os.path.join(kw["out_dir"], RESULT_NAME))
    assert "OPEN REFUSED" in (d / "access.log").read_text()


def test_boundary_violations_refused(tmp_path, monkeypatch):
    trainval, holdout, token, kw, _ = _setup(tmp_path, monkeypatch, name="d")
    full = pd.concat([trainval, holdout])
    with pytest.raises(RuntimeError, match="overlaps"):
        run_final_eval(PROVIDER, {"lookback": 20}, full, token, **kw)
    kw_late = dict(kw, holdout_start=str(
        (holdout.index.min() + pd.Timedelta(days=40)).date()))
    with pytest.raises(RuntimeError, match="wrong slice"):
        run_final_eval(PROVIDER, {"lookback": 20}, trainval, token, **kw_late)
    kw_short = dict(kw, holdout_end=str(
        (holdout.index.max() + pd.Timedelta(days=40)).date()))
    with pytest.raises(RuntimeError, match="truncated"):
        run_final_eval(PROVIDER, {"lookback": 20}, trainval, token, **kw_short)
    with pytest.raises(RuntimeError, match="column sets differ"):
        run_final_eval(PROVIDER, {"lookback": 20},
                       trainval.iloc[:, :-2], token, **kw)


def test_empty_ledger_refused(tmp_path, monkeypatch):
    trainval, _, token, kw, d = _setup(tmp_path, monkeypatch, name="e",
                                       seed_trial=False)
    with pytest.raises(RuntimeError, match="ledger is empty"):
        run_final_eval(PROVIDER, {"lookback": 20}, trainval, token, **kw)


def test_unfrozen_referee_refused(tmp_path, monkeypatch):
    trainval, _, token, kw, d = _setup(tmp_path, monkeypatch, name="f")
    monkeypatch.setattr(fe, "REPO", str(d))            # no validation/.frozen here
    with pytest.raises(RuntimeError, match="frozen"):
        run_final_eval(PROVIDER, {"lookback": 20}, trainval, token,
                       **dict(kw, require_frozen=True))
