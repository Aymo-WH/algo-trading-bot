"""Static-vs-timing decomposition (specs/EXP-003-s1-s2-timing-decomposition.md).

Splits an already point-in-time signal panel into a per-asset expanding-mean
"static" component and a "timing" residual, to test whether a signal's
graduation IC is carried by a static per-asset tilt or by genuine timing.

Not a signal itself -- a diagnostic transform applied to an existing Tier-1
signal (S1/S2). Inputs/outputs follow src/signals.py's convention: DataFrames
indexed by date, one column per asset, NaN where undefined.
"""
import pandas as pd

MIN_PRIOR_OBS = 52   # ~1 year of weekly readings (specs/EXP-003, frozen)


def expanding_static_timing(signal: pd.DataFrame, min_prior_obs: int = MIN_PRIOR_OBS
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-asset, point-in-time split of ``signal`` into:

    - static(t,a): mean of signal(tau,a) over all STRICTLY PRIOR non-NaN
      observations tau < t for that asset, defined only once >= min_prior_obs
      such observations exist; NaN otherwise.
    - timing(t,a): signal(t,a) - static(t,a), defined only where both terms
      are defined.

    Implemented via shift(1) + cumulative sum/count rather than
    ``DataFrame.expanding()`` so the NaN handling is explicit and
    independently verifiable, not dependent on pandas' internal min_periods
    semantics. shift(1) excludes each date's own value from its own mean --
    the same device src.signals.seasonality_s4 uses for its expanding,
    strictly-prior-years mean.
    """
    shifted = signal.shift(1)
    prior_count = shifted.notna().cumsum()
    prior_sum = shifted.fillna(0.0).cumsum()
    static = prior_sum.div(prior_count.where(prior_count > 0))
    static = static.where(prior_count >= min_prior_obs)
    timing = signal - static
    return static, timing
