"""CSCV / Probability of Backtest Overfitting.

Bailey, Borwein, López de Prado, Zhu (2017), J. Computational Finance. Clean
reimplementation of the v1 core (src/core/pbo_validator.py, audited faithful in
research/v1_forensics.md) moved inside the frozen referee boundary, with block
aggregates precomputed so the full C(16,8)=12,870 sweep stays fast for N<=500
trial columns.

Input M: T×N matrix — T synchronous periods of returns for N logged trial
configurations (the WHOLE ledger, discards included).
"""
from itertools import combinations

import numpy as np


def _block_aggregates(M: np.ndarray, S: int):
    """Per-block per-column (count, sum, sumsq) so any block-union Sharpe is O(N)."""
    T = M.shape[0]
    blocks = np.array_split(np.arange(T), S)
    cnt = np.array([[np.sum(~np.isnan(M[b, j])) for j in range(M.shape[1])] for b in blocks], dtype=float)
    Mz = np.nan_to_num(M, nan=0.0)
    ssum = np.array([Mz[b].sum(axis=0) for b in blocks])
    ssq = np.array([(Mz[b] ** 2).sum(axis=0) for b in blocks])
    return cnt, ssum, ssq


def _sharpe_from_agg(cnt, ssum, ssq, idx):
    n = cnt[idx].sum(axis=0)
    s = ssum[idx].sum(axis=0)
    q = ssq[idx].sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
        var = q / n - mean**2
        sd = np.sqrt(np.maximum(var, 0.0))
        sr = np.where((n > 1) & (sd > 0), mean / sd, -np.inf)
    return sr


def cscv_pbo(M: np.ndarray, S: int = 16) -> dict:
    """Probability of Backtest Overfitting via CSCV.

    Returns PBO, the logit distribution, and degradation stats. Metric = Sharpe
    (per paper's default). Requires N >= 2 columns; meaningful with dozens+.
    """
    M = np.asarray(M, dtype=float)
    T, N = M.shape
    if N < 2:
        raise ValueError("CSCV needs >= 2 trial columns")
    if S % 2 != 0 or T < 2 * S:
        raise ValueError("S must be even and T >= 2S")
    cnt, ssum, ssq = _block_aggregates(M, S)
    all_blocks = list(range(S))
    logits = []
    oos_sr_of_is_best = []
    for is_combo in combinations(all_blocks, S // 2):
        is_idx = list(is_combo)
        oos_idx = [b for b in all_blocks if b not in is_combo]
        sr_is = _sharpe_from_agg(cnt, ssum, ssq, is_idx)
        sr_oos = _sharpe_from_agg(cnt, ssum, ssq, oos_idx)
        n_star = int(np.argmax(sr_is))
        # OOS relative rank of the IS winner (1 = worst ... N = best)
        rank = 1 + np.sum(sr_oos < sr_oos[n_star]) \
             + 0.5 * np.sum((sr_oos == sr_oos[n_star])) - 0.5
        omega = rank / (N + 1)
        omega = min(max(omega, 1e-9), 1 - 1e-9)
        logits.append(np.log(omega / (1 - omega)))
        oos_sr_of_is_best.append(sr_oos[n_star])
    logits = np.array(logits)
    return {
        "pbo": float(np.mean(logits <= 0)),
        "n_combinations": len(logits),
        "n_trials": N,
        "logit_mean": float(logits.mean()),
        "logit_std": float(logits.std()),
        "oos_sr_of_is_best_mean": float(np.nanmean(oos_sr_of_is_best)),
    }
