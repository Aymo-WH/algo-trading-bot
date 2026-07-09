# EXP-001 — Tier-1 Signal Library IC Screen (frozen before first run)

Frozen: 2026-07-09, before `src/signals.py` computes anything on real data.
Scope: honest cross-sectional IC measurement of the four Tier-1 signals named
in `research/design.md` §5, on real train/validation data (never the holdout).
This is the Phase 2 deliverable (mission §5): "measure standalone
cross-sectional IC honestly on walk-forward; keep only those with stable,
real IC."

## Hypothesis

Each of the four Tier-1 signal constructions already specified in
`research/design.md` §5 (operator-approved 2026-07-07) — computed exactly as
written below, with no free hyperparameter — clears the graduation rule
already coded in `validation/ic.py::passes_graduation` when measured on the
real train/validation panel. Falsified per-signal if that signal misses any
leg of the rule; falsified for the Tier-1 hypothesis as a whole only if ALL
FOUR miss (design §10, kill criterion K1).

## Data window (train/validation only — the holdout stays sealed)

`data/panel/trainval_close.csv` + `eligibility_trainval.csv` (Phase 1,
commit 84929f2). Rebalance dates = `src.panel_factory.rebalance_dates` on the
trainval calendar, intersected with `[panel_start=1999-12-22,
trainval_end=2021-12-31]` (the exact Phase-1 boundary; identical to
`validation.final_eval.HOLDOUT_START/END`, checked by
`tests/data/test_panel_integrity.py::test_boundaries_agree_with_frozen_referee`).
A date's cross-section is masked to names with `eligibility_trainval[name,
date] == True`; all cross-sectional statistics (z-scoring, market proxy,
demeaning) use only that masked set. No look-ahead: every construction below
uses trailing or expanding windows computed strictly at/before the date it is
assigned to.

## Exact constructions (frozen; implemented verbatim in `src/signals.py`)

Let `P` = trainval close panel, `E` = eligibility mask, `skip = 21` trading
days.

- **S1 — cross-sectional momentum** (design §5 row S1, "trailing return
  skip-month, z-scored per date"): for `LB ∈ {252, 126, 63}` (12m/6m/3m),
  `r_LB(t,a) = P(t-skip,a)/P(t-skip-LB,a) - 1`; `z_LB(t,·)` = per-date
  cross-sectional z-score over eligible names only; `S1(t,a) =
  mean(z_252, z_126, z_63)` over whichever of the 3 are defined for `a` at
  `t` (NaN only if all 3 are undefined).
- **S2 — time-series trend** (design §5 row S2, "sign/strength of 12m & 3m
  trend... book stays neutral"): for `LB ∈ {252, 63}`, **no skip** (TS trend
  literature does not apply XS momentum's reversal skip — Baz et al. 2015):
  `r_LB(t,a) = P(t,a)/P(t-LB,a) - 1`; per-date z-score as in S1;
  `S2(t,a) = mean(z_252, z_63)`.
- **S3 — low-beta/BAB** (design §5 row S3, "rank by trailing 252d beta vs
  equal-weight market factor"): daily eligible-equal-weight market return
  `m(t) = mean_a{ret(t,a) : E(t,a)}`; `beta(t,a)` = OLS slope of `ret(·,a)`
  on `m(·)` over the trailing 252 trading days ending at `t` inclusive;
  `S3(t,a) = -beta(t,a)` (sign-flipped so higher S3 = lower beta = the BAB-
  favored side; no z-scoring — a single-component signal's z-score is a
  no-op under Spearman rank IC, so it is omitted rather than computed and
  discarded).
- **S4 — calendar seasonality** (design §5 row S4, "same-calendar-month mean
  relative return, ≥5y history required"): relative return
  `rr(t,a) = ret(t,a) - m(t)` (eligible equal-weight demeaned, same `m` as
  S3); for asset `a`, calendar month `k∈{1..12}`, year `y`: `M(a,k,y) =`
  mean of `rr(·,a)` over calendar days in month `k` of year `y`. At date `t`
  in month `k(t)`, year `y(t)`: `S4(t,a) = mean{ M(a,k(t),y) : y < y(t) }`
  using only years strictly before the current one (expanding, point-in-time
  by construction), defined only once **≥ 5** such prior years exist for
  that `(a,k)` pair; NaN otherwise.

## Exact metric (formula + code path)

Label: `fwd5(t,a) = P(t+5,a)/P(t,a) - 1` (5-trading-day forward return on the
daily panel — identical convention to `validation/final_eval.py`'s
`full.shift(-5)/full - 1`). Cross-sectional demeaning of the label (design
§4's "cross-sectionally demeaned" target) is **not applied**: subtracting a
per-date constant from every name cannot change a per-date Spearman rank
correlation, so it is a no-op for this metric — noted here so its absence is
a documented equivalence, not a silent deviation from §4.

Per signal: `validation.ic.cross_sectional_ic(signal, fwd5, min_names=10)`
(Spearman rank IC per rebalance date; `signal.align(fwd5, join="inner")`
restricts to the weekly dates where both exist — no manual reindexing
needed) → `validation.ic.ic_summary(ic, nw_lags=2)` (mean IC, Newey-West t,
per-calendar-year sign consistency) → `validation.ic.passes_graduation(...)`
with the codebase's existing defaults (`min_ic=0.01, min_t=2.0,
min_year_share=0.60)` — **unchanged from `validation/ic.py`, not
reparameterized for this test.**

`nw_lags=2` (the `ic_summary` default) matches
`research/validation_methodology.md` §7's "non-overlapping weekly: 1–2"
guidance, since the rebalance is weekly and the 5-trading-day label
approximately equals one rebalance interval (non-overlapping in the
relevant sense).

## No CPCV/walk-forward fold split at the IC-measurement step (justified)

Every construction above is a zero-parameter, point-in-time transform: S1–S3
are trailing rolling windows recomputed fresh at each date (no value fit on
a training set and reused out-of-sample — the same category as this repo's
existing point-in-time PCA/beta discipline); S4 is an explicit *expanding*,
prior-years-only calendar mean with a hard 5-year minimum, which is exactly
the walk-forward discipline `design.md` §5's "walk-forward" qualifier is
asking for, enforced inside the feature itself rather than via an external
split. There is no hyperparameter selection happening (no lookback search,
no threshold tuning) for a train/test split to protect against. `CPCV`/`PBO`
(`validation/splits.py`, `validation/cscv.py`) remain reserved for Phase 3+,
when there are multiple *competing configurations* to rank against each
other — not applicable to a single fixed construction per signal. The
leak-hunter audit (below) is asked explicitly to attack this claim against
the actual `src/signals.py` code, not accept it as asserted.

## Pass threshold (pre-committed, per signal)

`passes_graduation()` returns `True`, i.e. `mean_ic >= 0.01 AND t_nw >= 2.0
AND pct_years_positive >= 0.60`, all three simultaneously, on the real
train/validation panel as scoped above.

## Falsification criterion

Per-signal: failing any one of the three legs falsifies that signal — it is
dropped from the library, logged as a `result` (not a `discard`), and is
**not** eligible for rescue by changing its lookback/construction without a
brand-new logged trial (design §5: "no post-hoc rescue by re-parameterization
without a new logged trial"). Whole-hypothesis: falsified (Tier-1 null, design
§10 K1 → attempt Tier-2 carry, else write the rigorous null) only if all four
signals fail. `mean_ic > 0.15` on any signal is a pre-declared leakage red
flag (`research/validation_methodology.md` §7) — triggers the mission §3.3f
adversarial-audit path before the result is trusted, positive or not.

## Planned trial count

**4** — one per signal (S1, S2, S3, S4), no sub-variant scan (every
lookback/window choice above is fixed by the already-approved design, not
searched). Logged as 4 `phase="result"` ledger rows against the ≤250 budget
(design §10); this is the first real-trial usage of that budget (Phase 0/1
work logged as `preregistered`/`data_build`/synthetic-throwaway, none of
which count — decision D14).

## Seeds / config

None — every construction is deterministic on already-frozen real historical
data (`data/panel/trainval_close.csv`, sha256 in
`data/panel/MANIFEST.json`); no randomness anywhere in this test.
