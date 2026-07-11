"""Sharpe-family statistics: PSR, DSR, effective trial count, Newey-West t.

Formulas from Bailey & López de Prado (2014), "The Deflated Sharpe Ratio", JPM
40(5), and Newey-West (1987). All Sharpe inputs here are PER-PERIOD (not
annualized) unless the function says otherwise.
"""
import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def sharpe(returns: np.ndarray) -> float:
    """Per-period Sharpe of a return series (population std)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return np.nan
    sd = r.std()
    return float(r.mean() / sd) if sd > 0 else np.nan


def annualized_sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    return sharpe(returns) * np.sqrt(periods_per_year)


def psr(sr_hat: float, sr_ref: float, n_obs: int,
        skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_ref), non-normality-adjusted.

    kurt is RAW kurtosis (normal = 3). Bailey-LdP 2014, Eq. for PSR.
    """
    if n_obs < 2 or np.isnan(sr_hat):
        return np.nan
    denom = np.sqrt(1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2)
    if denom <= 0 or np.isnan(denom):
        return np.nan
    z = (sr_hat - sr_ref) * np.sqrt(n_obs - 1.0) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: float, var_trial_sr: float) -> float:
    """E[max SR] across n_trials of zero-true-SR strategies (Gumbel approx).

    Bailey-LdP 2014: SR0 = sqrt(V[SR_n]) * ((1-γ)·Z⁻¹(1-1/N) + γ·Z⁻¹(1-1/(N·e))).
    Returns 0 for n_trials <= 1 (a single trial deserves no deflation).
    """
    if n_trials <= 1 or var_trial_sr <= 0:
        return 0.0
    g = EULER_MASCHERONI
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_trial_sr) * ((1.0 - g) * z1 + g * z2))


def effective_trials(n_trials: int, mean_corr: float) -> float:
    """N̂ = ρ̄ + (1-ρ̄)·M — effective number of independent trials (BLdP Eq. 9)."""
    rho = float(np.clip(mean_corr, 0.0, 1.0))
    return rho + (1.0 - rho) * n_trials


def mean_pairwise_correlation(M: np.ndarray | None) -> float:
    """Mean off-diagonal correlation across trial return columns (T×N).

    Feeds effective_trials(); estimated from the same persisted trial return
    matrix CSCV uses. Returns 0.0 (conservative: maximal deflation, N̂ = M)
    when no matrix, < 2 usable columns, or degenerate columns.
    """
    if M is None:
        return 0.0
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2 or M.shape[0] < 3:
        return 0.0
    sd = np.nanstd(M, axis=0)
    valid = M[:, sd > 0]
    if valid.shape[1] < 2:
        return 0.0
    C = np.corrcoef(valid.T)
    off = C[np.triu_indices_from(C, k=1)]
    off = off[~np.isnan(off)]
    return float(off.mean()) if len(off) else 0.0


def dsr(sr_hat: float, trial_sr_variance: float, n_trials: float, n_obs: int,
        skew: float = 0.0, kurt: float = 3.0) -> float:
    """Deflated Sharpe Ratio = PSR with sr_ref = E[max SR under the null].

    n_trials should be the EFFECTIVE trial count (use effective_trials()).
    Accept threshold per contract: DSR > 0.95.
    """
    sr0 = expected_max_sharpe(n_trials, trial_sr_variance)
    return psr(sr_hat, sr0, n_obs, skew, kurt)


def dsr_from_ledger(returns: np.ndarray, trial_sharpes: np.ndarray,
                    trial_corr_mean: float = 0.0) -> dict:
    """DSR of a selected strategy given the full ledger of trial Sharpes.

    returns: per-period return series of the SELECTED strategy.
    trial_sharpes: per-period Sharpe of EVERY logged trial (discards included).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    sr_hat = sharpe(r)
    n_eff = effective_trials(len(trial_sharpes), trial_corr_mean)
    var_sr = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1)) \
        if len(trial_sharpes) > 1 else 0.0
    skew = float(stats.skew(r)) if len(r) > 2 else 0.0
    kurt = float(stats.kurtosis(r, fisher=False)) if len(r) > 3 else 3.0
    value = dsr(sr_hat, var_sr, n_eff, len(r), skew, kurt)
    return {"dsr": value, "sr_hat": sr_hat, "n_trials": len(trial_sharpes),
            "n_effective": n_eff, "trial_corr_mean": float(trial_corr_mean),
            "trial_sr_variance": var_sr,
            "sr0_expected_max": expected_max_sharpe(n_eff, var_sr),
            "skew": skew, "kurt": kurt, "n_obs": len(r)}


def newey_west_tstat(series: np.ndarray, lags: int = 5) -> float:
    """t-stat of the mean of ``series`` with Newey-West (1987) HAC standard error."""
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < max(3, lags + 2):
        return np.nan
    e = x - x.mean()
    var = float(e @ e) / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(e[lag:] @ e[:-lag]) / n
    if var <= 0:
        return np.nan
    return float(x.mean() / np.sqrt(var / n))
