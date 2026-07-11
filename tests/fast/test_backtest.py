"""Backtester causality, cost, and accounting proofs."""
import numpy as np
import pandas as pd
import pytest

from validation.backtest import backtest_panel, performance_summary

IDX = pd.bdate_range("2020-01-01", periods=100)


def _panel(vals):
    return pd.DataFrame(vals, index=IDX, columns=["A", "B"])


def test_weights_cannot_earn_same_bar_return():
    """A weight set on the day of a +10% jump must NOT capture that jump."""
    rets = _panel(0.0)
    rets.iloc[50, 0] = 0.10
    w = _panel(0.0)
    w.iloc[50, 0] = 1.0            # decided at the close of the jump day
    out = backtest_panel(w, rets, cost_bps_per_side=0.0)
    assert out["gross"].iloc[50] == 0.0          # same-bar: nothing
    assert out["gross"].sum() == pytest.approx(0.0)


def test_next_bar_return_is_captured():
    rets = _panel(0.0)
    rets.iloc[51, 0] = 0.10
    w = _panel(0.0)
    w.iloc[50, 0] = 1.0
    out = backtest_panel(w, rets, cost_bps_per_side=0.0)
    assert out["gross"].iloc[51] == pytest.approx(0.10)


def test_costs_charged_on_turnover_both_ways():
    rets = _panel(0.0)
    w = _panel(0.0)
    w.iloc[10] = [0.5, -0.5]       # enter: |Δw| = 1.0
    w.iloc[20] = [0.0, 0.0]        # exit:  |Δw| = 1.0
    out = backtest_panel(w, rets, cost_bps_per_side=10.0)
    assert out["costs"].sum() == pytest.approx(2.0 * 10.0 / 1e4)
    assert out["net"].sum() == pytest.approx(-2.0 * 10.0 / 1e4)


def test_weights_persist_between_rebalances():
    rets = _panel(0.01)
    w = pd.DataFrame([[1.0, 0.0]], index=[IDX[10]], columns=["A", "B"])
    out = backtest_panel(w, rets, cost_bps_per_side=0.0)
    assert np.allclose(out["gross"].iloc[11:], 0.01)  # held throughout
    assert (out["gross"].iloc[:11] == 0).all()


def test_dollar_neutral_book_ignores_common_moves():
    common = np.full(100, 0.02)
    rets = _panel(np.column_stack([common, common]))
    w = _panel(np.tile([0.5, -0.5], (100, 1)))
    out = backtest_panel(w, rets, cost_bps_per_side=0.0)
    assert abs(out["gross"].iloc[1:]).max() < 1e-15


def test_performance_summary_maxdd_and_years():
    port = pd.DataFrame({
        "gross": np.concatenate([np.full(50, 0.01), np.full(50, -0.005)]),
        "net": np.concatenate([np.full(50, 0.01), np.full(50, -0.005)]),
        "turnover": np.zeros(100), "costs": np.zeros(100)}, index=IDX)
    s = performance_summary(port)
    expected_dd = (1 - 0.005) ** 50 - 1
    assert s["net"]["max_drawdown"] == pytest.approx(expected_dd, rel=1e-6)
    assert s["net"]["ann_sharpe"] != 0
