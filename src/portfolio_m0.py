"""Phase 3 M0 combiner (design.md §6-7; frozen construction in
specs/EXP-002-phase3-m0-combiner.md, amended per the no-trade-band scope
correction below). Provider contract for `validation.run_battery`:
``m0_provider(prices, config) -> {"weights", "signal"}``.

Construction: equal-weight composite of S1 (momentum) + S2 (TS trend), made
dollar-neutral AND beta-neutral by taking the cross-sectional OLS residual
against rolling beta (exact identity: the residual sums to zero and is
orthogonal to beta, per date, before any clipping, PROVIDED the composite
and beta NaN masks coincide -- see `neutralize_against_beta`), then
z-scored, clipped at +/-2.5, and turned into weights via iterative
position/category-cap waterfilling to a 200% gross target. All signal math
above is point-in-time (reuses `src.signals`, already tested); this module
adds only the combiner + portfolio-construction layer.

Turnover control (the no-trade band) is deferred to Phase 4 alongside
vol-targeting and the drawdown brake (mission §5 assigns "turnover control"
to Phase 4, not Phase 3) -- `apply_no_trade_band`/`_freeze_small_changes`
remain here, tested, for that reuse, but are NOT invoked by
`compute_signal_and_weights`. A prior version called it inside Phase 3's
pipeline; a fresh-context reviewer found that interaction reintroduced real
cap violations (up to 1.34% of gross) via an unsanctioned post-freeze
rescale not in the frozen spec. Removing the call eliminates the bug at its
source rather than patching around it (see decisions.md D26)."""
import json
import os

import numpy as np
import pandas as pd

from src.panel_factory import PANEL_DIR, rebalance_dates
from src.signals import (_cross_sectional_zscore, equal_weight_market_return,
                         low_beta_s3, momentum_s1, trend_s2)

GROSS = 2.0
POSITION_CAP_FRAC = 0.10
CATEGORY_CAP_FRAC = 0.40
NO_TRADE_BAND = 0.005
CLIP = 2.5
MAX_CAP_ITER = 50
CAP_TOL = 1e-9


def _manifest() -> dict:
    with open(os.path.join(PANEL_DIR, "MANIFEST.json")) as f:
        return json.load(f)


def load_canonical_trainval() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """The frozen Phase-1 train/validation artifacts (never the holdout)."""
    close = pd.read_csv(os.path.join(PANEL_DIR, "trainval_close.csv"),
                        index_col=0, parse_dates=True).sort_index()
    elig = pd.read_csv(os.path.join(PANEL_DIR, "eligibility_trainval.csv"),
                       index_col=0, parse_dates=True).sort_index().astype(bool)
    manifest = _manifest()
    categories = pd.Series(manifest["universe"])
    return close, elig, categories, manifest


def neutralize_against_beta(composite: pd.DataFrame, beta: pd.DataFrame,
                            elig: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional OLS residual of `composite` on [1, beta], via
    the centered-covariance closed form (single regressor). Exact identity:
    ``sum(resid) == 0`` and ``sum(resid * beta) == 0`` per date over the
    names where BOTH are defined — a portfolio weighted proportional to
    `resid` is dollar-neutral AND beta-neutral before any clipping.

    Both inputs are masked to their JOINT notna() (not just `elig`) before
    computing cross-sectional moments: `elig` alone does not guarantee
    `composite` and `beta` are defined on the same names (beta additionally
    needs 252 trailing RETURNS, composite needs 252 trailing PRICES), and a
    mask mismatch breaks the exact-zero identity (found by fresh-context
    review: 5/1149 real dates, max |sum(resid)| = 1.83 before this fix)."""
    joint = elig & composite.notna() & beta.notna()
    Cm = composite.where(joint)
    Bm = beta.where(joint)
    mu_c = Cm.mean(axis=1)
    mu_b = Bm.mean(axis=1)
    Cc = Cm.sub(mu_c, axis=0)
    Bc = Bm.sub(mu_b, axis=0)
    cov = (Cc * Bc).mean(axis=1)
    var = (Bc * Bc).mean(axis=1)
    slope = cov.div(var.where(var > 0), axis=0)
    return Cc.sub(Bc.mul(slope, axis=0))


def _cap_and_scale_row(raw: pd.Series, categories: pd.Series) -> pd.Series:
    """Scale a single date's demeaned/clipped score to GROSS exposure, honoring
    the position and category caps via iterative waterfilling (approximate,
    not machine-precision — see spec §6 for the tolerance this is checked at).

    Names/categories that hit a cap are LOCKED at that value; only names that
    have never hit either cap absorb further redistribution. Without this,
    redistributing to "not currently over cap" names can re-inflate a name
    whose category was just scaled down, oscillating instead of converging."""
    w = raw.dropna()
    if w.empty or w.abs().sum() == 0:
        return raw * 0.0
    w = w / w.abs().sum() * GROSS
    pos_cap = POSITION_CAP_FRAC * GROSS
    cat_cap = CATEGORY_CAP_FRAC * GROSS
    cats = categories.reindex(w.index)
    locked = pd.Series(False, index=w.index)

    for _ in range(MAX_CAP_ITER):
        changed = False
        over = (w.abs() > pos_cap + CAP_TOL) & ~locked
        if over.any():
            w.loc[over] = w.loc[over].clip(lower=-pos_cap, upper=pos_cap)
            locked.loc[over] = True
            changed = True
        for cat in cats.dropna().unique():
            idx = cats[cats == cat].index.intersection(w.index)
            sub = w.loc[idx]
            gross_c = sub.abs().sum()
            if gross_c > cat_cap + CAP_TOL:
                w.loc[idx] = sub * (cat_cap / gross_c)
                locked.loc[idx] = True
                changed = True
        current_gross = w.abs().sum()
        headroom = GROSS - current_gross
        free = ~locked
        if headroom > CAP_TOL and free.any():
            free_w = w.loc[free]
            free_gross = free_w.abs().sum()
            if free_gross > CAP_TOL:
                w.loc[free] = free_w * (1 + headroom / free_gross)
                changed = True
        if not changed:
            break
    else:
        raise RuntimeError(
            f"cap waterfilling did not converge within {MAX_CAP_ITER} iterations "
            f"for date-row with {len(w)} names -- do not silently ship an "
            "unconverged cap solution")
    return w.reindex(raw.index).fillna(0.0)


def build_capped_weights(clipped: pd.DataFrame, categories: pd.Series) -> pd.DataFrame:
    rows = {date: _cap_and_scale_row(row, categories)
            for date, row in clipped.iterrows()}
    return pd.DataFrame(rows).T.reindex(columns=clipped.columns)


def _freeze_small_changes(target: pd.Series, prev: pd.Series, band: float) -> pd.Series:
    """A name keeps its previously HELD weight if the desired change is
    < `band` of NAV (design.md §7 turnover control); otherwise it trades to
    the fresh target."""
    keep = (target - prev).abs() < band
    return target.where(~keep, prev)


def apply_no_trade_band(weights: pd.DataFrame, band: float = NO_TRADE_BAND) -> pd.DataFrame:
    """Sequential pass applying `_freeze_small_changes` date over date.
    NOT called by `compute_signal_and_weights` (see module docstring) — kept
    here, tested, for Phase 4 (turnover control is Phase-4 scope per the
    mission's phase plan, alongside vol-targeting/drawdown-brake).

    Freezing only SOME names' positions while others still move to their
    freshly-capped targets breaks the aggregate gross/neutrality guarantee
    `build_capped_weights` established for a fully self-consistent solution.
    The uniform rescale here restores the gross target but can still push a
    name/category past its cap by a small margin (found by fresh-context
    review to exceed the Phase-3 spec's stated tolerance on real data) — a
    Phase-4 adoption of this function should account for that rather than
    reuse it as-is."""
    out = weights.copy()
    prev = None
    for date in out.index:
        target = out.loc[date]
        if prev is None:
            prev = target.copy()
            continue
        adjusted = _freeze_small_changes(target, prev, band)
        gross_now = adjusted.abs().sum()
        if gross_now > CAP_TOL:
            adjusted = adjusted * (GROSS / gross_now)
        out.loc[date] = adjusted
        prev = adjusted
    return out


def compute_signal_and_weights(close: pd.DataFrame, elig: pd.DataFrame,
                              categories: pd.Series,
                              window: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full M0 construction. Returns (weights, signal), both indexed at
    `window` rebalance dates only."""
    ret = close.pct_change(fill_method=None)
    market = equal_weight_market_return(ret, elig)
    composite = 0.5 * (momentum_s1(close, elig) + trend_s2(close, elig))
    beta = -low_beta_s3(close, elig, market=market)
    resid = neutralize_against_beta(composite, beta, elig)
    z = _cross_sectional_zscore(resid, elig)
    clipped = z.clip(-CLIP, CLIP)
    demeaned = clipped.sub(clipped.where(elig).mean(axis=1), axis=0)

    signal = demeaned.reindex(window)
    weights = build_capped_weights(signal, categories)
    return weights, signal


_CONFIG_CONSTANTS = {"gross": GROSS, "position_cap_frac": POSITION_CAP_FRAC,
                    "category_cap_frac": CATEGORY_CAP_FRAC, "clip": CLIP}


def m0_provider(prices: pd.DataFrame, config: dict) -> dict:
    """`validation.run_battery` provider entry point. `prices` is whatever
    date-sliced panel the CLI was invoked with (never the holdout) — this
    function verifies it is a faithful slice of the canonical Phase-1
    train/validation artifact before using the canonical file's full history
    (needed for trailing lookbacks the slice alone cannot support).

    `config` is logged to the ledger as this trial's parameters but was
    previously never checked against what the module actually uses —
    asserting equality here means the ledger's recorded config is never
    silently decorative (fresh-context review finding)."""
    for key, expected in _CONFIG_CONSTANTS.items():
        if key in config and config[key] != expected:
            raise ValueError(f"config[{key!r}]={config[key]!r} does not match the "
                             f"module constant {expected!r} this provider actually uses")

    close, elig, categories, manifest = load_canonical_trainval()
    rebal = rebalance_dates(close.index)
    window = rebal[(rebal >= manifest["panel_start"]) & (rebal <= manifest["trainval_end"])]

    if not prices.index.isin(close.index).all():
        raise ValueError("prices argument has dates outside the canonical trainval panel")
    if not prices.columns.isin(close.columns).all():
        raise ValueError("prices argument has columns outside the canonical universe")
    if len(prices.index) == 0 or len(prices.columns) == 0:
        raise ValueError("prices argument is empty -- refusing to trade on it")
    a = prices.loc[prices.index, prices.columns]
    b = close.loc[prices.index, prices.columns]
    if not a.isna().equals(b.isna()):
        raise ValueError("prices argument's missing-value pattern does not match the "
                         "canonical trainval panel -- refusing to trade on it")
    mismatched = (a - b).abs() > 1e-9
    if bool(mismatched.any().any()):
        raise ValueError("prices argument does not match the canonical trainval panel "
                         "at one or more overlapping cells -- refusing to trade on it")

    weights, signal = compute_signal_and_weights(close, elig, categories, window)
    weights = weights.reindex(columns=prices.columns).fillna(0.0)
    signal = signal.reindex(columns=prices.columns)
    return {"weights": weights, "signal": signal}
