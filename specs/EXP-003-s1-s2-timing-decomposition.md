# EXP-003 — S1/S2 Static-vs-Timing Decomposition (frozen before first run)

Frozen: 2026-07-10, before any component of this decomposition is computed on
real data. This is operator item "B" from the 2026-07-10 session resume,
sequenced to run before EXP-004 (Tier-2 carry, D28) — cheap, no new data
dependency, and directly calibrates expectations for carry. Refined per a
2026-07-10 design-reviewer (Fable) consult on the proposed sequencing before
freezing (journal.md).

## Hypothesis

S1 (momentum) and S2 (TS trend) each contain a **timing** component — the
part of the signal that varies over time, net of that asset's own historical
average signal level — that independently clears the unchanged Tier-1
graduation rule (`validation.ic.passes_graduation`: mean_ic>=0.01, t_nw>=2.0,
pct_years_positive>=0.60), the same three-leg rule S1/S2 themselves cleared
at EXP-001 (D23).

This directly answers a question left open by EXP-002's audit (journal.md,
2026-07-10): S1/S2's measured time_shift canary retention (81%/96%,
"Retroactive canary check on S1/S2 individually") is *consistent with* either
(a) a genuinely slow-decaying but real timing signal, or (b) a signal
substantially explained by a per-asset static tilt that a 26-week shift
barely disturbs — mirroring M0's own decomposition finding (constant tilt
t=5.28 vs timing-only t=2.81, from the EXP-002 leak-hunter audit). This
experiment tests (b) directly rather than inferring it from the canary alone.

**Explicit non-retroactivity clause (design-reviewer-required):** this
result, whichever way it comes out, does NOT reopen or change S1/S2's
EXP-001 graduation (D23 stands, unconditionally). It tests a DIFFERENT
constructed object (the residual after removing each asset's own historical
mean signal level) and informs interpretation of the existing result,
nothing more.

## Data window

Identical to EXP-001: `data/panel/trainval_close.csv` +
`eligibility_trainval.csv`, rebalance dates from
`src.panel_factory.rebalance_dates`, intersected with
`[panel_start=1999-12-22, trainval_end=2021-12-31]`. Same `fwd5` label
(`close.shift(-5)/close - 1`). Holdout never touched.

## Exact construction (frozen; to be implemented in `src/decomposition.py`)

Let `W` = the weekly-reindexed signal panel for S1 or S2 — i.e.
`momentum_s1(close, elig).reindex(window)` or
`trend_s2(close, elig).reindex(window)`, using the SAME frozen
`src/signals.py` functions EXP-001 used, unmodified. For asset `a`, let the
rebalance dates where `W(*, a)` is not NaN be indexed in order
`tau_1 < tau_2 < ... < tau_n` (asset-specific: an asset can be ineligible or
lack sufficient lookback at some dates).

- **Static component:**
  `static(tau_k, a) = mean(W(tau_1,a), ..., W(tau_{k-1},a))` if
  `k-1 >= MIN_PRIOR_OBS` (i.e. at least `MIN_PRIOR_OBS` strictly-prior
  non-NaN observations of that asset's OWN signal history exist), else NaN.
  `MIN_PRIOR_OBS = 52` (~1 year of weekly readings — chosen to give the
  static estimate a full annual cycle beyond the signal's own ~12m
  construction lookback; a fixed, pre-committed round number, not tuned on
  any real-data result).
  Implementation: a NaN-aware expanding mean via `shift(1)` (excludes the
  current observation from its own mean, same device `seasonality_s4` uses)
  followed by cumulative-sum / cumulative-count (avoids relying on pandas'
  `.expanding()` NaN-counting semantics; independently verifiable by hand in
  unit tests).
- **Timing component:** `timing(tau_k, a) = W(tau_k, a) - static(tau_k, a)`,
  defined only where both terms are defined (NaN otherwise).
- **Point-in-time property:** `static`/`timing` at `tau_k` use only
  `W`-values at `tau_1 .. tau_{k-1}` (strictly prior), and `W` itself is
  already point-in-time (unchanged from EXP-001) — no new look-ahead is
  introduced. The leak-hunter audit is asked to verify this claim against
  the actual `src/decomposition.py` code, not accept it as asserted
  (matching EXP-001's practice, including an empirical truncation attack if
  applicable).

Applied to S1 and S2 independently — no cross-signal construction, no new
hyperparameter search.

## Metric, graduation rule, and canaries (all pre-existing, none reparameterized)

- **Confirmatory test (counts against the trial budget):** `timing(S1)` and
  `timing(S2)`, each scored via the UNCHANGED
  `validation.ic.cross_sectional_ic` -> `ic_summary` -> `passes_graduation`
  pipeline EXP-001 used: same `fwd5` label, same `nw_lags=2`, same
  thresholds (`min_ic=0.01, min_t=2.0, min_year_share=0.60`).
  **Planned trial count: 2** (`EXP-003-S1_timing`, `EXP-003-S2_timing`).
- **Descriptive only, NOT part of the pass/fail rule** (mirrors EXP-001's
  inter-signal-correlation treatment): the STATIC component's own IC
  (`static(S1)`, `static(S2)`), reported for interpretation alongside the
  timing result but not tested against a threshold and not logged as a
  separate trial — a static per-asset tilt is not a timed prediction, so
  grading it against a rule built for timed signals would be a category
  error, not a real test.
- **Canaries (D30 — diagnostic, non-blocking):**
  `validation.canaries.run_signal_canaries` (label_shuffle, time_shift with
  the unchanged default `shift_rows=26`, random_feature) run against BOTH
  timing components, reported in full; does not gate `passes_graduation`.
  Direct comparability target: the already-reported raw-signal canary
  numbers from the 2026-07-10 retroactive check (S1 base_t=3.9187,
  lagged_t=3.1802, 81.1% retention; S2 base_t=3.1622, lagged_t=3.0283, 95.8%
  retention). If the timing components show materially lower time_shift
  retention than the raw signals, that supports the static-tilt reading of
  the original result; if retention is similar, the slow-decay-but-real
  reading is favored instead. Neither outcome reopens D23 (see above).

## Falsification criterion

Per-component: failing any one of the three graduation legs falsifies that
timing component (logged as a `result`, not a `discard`; not eligible for
rescue by reparameterizing `MIN_PRIOR_OBS` without a new logged trial, same
rule as `design.md` §5). Whole-hypothesis falsified only if BOTH
`timing(S1)` and `timing(S2)` fail — in which case the honest reading is
"this universe's Tier-1 signals are substantially static tilts, not timed
predictions," feeding directly into the EXP-004 carry conversation per D28's
sequencing rationale. If either or both pass, the reading is "genuine timed
information exists alongside the static tilt" — informative either way, per
mission §1's "both outcomes are success."

## Seeds / config

None for the IC/graduation test — deterministic on already-frozen real
historical data, same as EXP-001. `label_shuffle_canary` and
`random_feature_canary` use their existing default `seed=0` (unchanged from
`validation/canaries.py`), for exact reproducibility.
