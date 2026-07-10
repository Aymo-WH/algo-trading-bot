# EXP-002 — Phase 3: M0 Combiner (S1+S2), market-neutral portfolio (frozen before first run)

Frozen: 2026-07-10, before any weight/return is computed on real data. Operator
sign-off: D25 (research/decisions.md) — proceed to Phase 3 with S1+S2 as-is; M0
built exactly per design.md §6 with no adjustment for the 0.796 S1/S2 correlation
beyond the construction already specified below (design-reviewer's stated view:
the clipped z-score composite already sits near the GLS optimum for two
similarly-scaled ICs at this correlation level — a nearly-flat-objective argument,
not a claim of exact GLS-weight equivalence).

Scope: the Phase 3 deliverable (mission §5): "Combine signals -> cross-sectional
predictions -> market-neutral portfolio; ML as combiner/sizer... a working
strategy evaluated on walk-forward only." M0 is the **mandatory first** combiner
(design.md §6) — a transparent, near-unoverfittable equal-weight composite that
every later ML model (M1/M2) must beat before being admitted. Vol-targeting and
the drawdown brake (design.md §7) are explicitly OUT of scope here — mission §5
assigns them to Phase 4 ("Costs & risk overlay"), not Phase 3.

## Hypothesis

The M0 combiner (S1 momentum + S2 TS trend, equal-weighted, beta-neutralized,
dollar-neutralized, capped, weekly-rebalanced per the exact construction below)
clears kill criterion K2 (design.md §10: combined book net Sharpe >= 0.2 on
walk-forward) when evaluated on train/validation only via the existing
`validation.run_battery` referee (walk-forward + CPCV + PBO + DSR + canaries).
Falsified if net Sharpe (5bps) < 0.2 on the full train/validation walk-forward
window, OR if the referee's canary suite trips (indicating the "signal" driving
weights is not honestly point-in-time).

This is the specific test the prior session's redesign consult (D25) proposed in
place of blocking on signal-library expansion: let the pre-registered gate decide
whether ~1.1-1.4 effective breadth (S1/S2 correlation 0.796) is survivable, rather
than assuming it in either direction.

## Exact construction (frozen; implemented verbatim in `src/portfolio_m0.py`)

Let `close`/`elig` be the full train/validation panel (`data/panel/trainval_close.csv`
+ `eligibility_trainval.csv`, Phase 1, commit 84929f2 — same artifacts EXP-001 used),
`market` = `src.signals.equal_weight_market_return` on that panel, `rebal` =
`src.panel_factory.rebalance_dates` intersected with
`[panel_start=1999-12-22, trainval_end=2021-12-31]` (identical window to EXP-001).

1. **Composite:** `C = 0.5 * (momentum_s1(close, elig) + trend_s2(close, elig))`
   — the "equal-weight composite of surviving signals" design.md §6 mandates for
   M0, using S1/S2 exactly as frozen in EXP-001 (no re-parameterization).
2. **Beta:** `beta = -low_beta_s3(close, elig, market=market)` (recovers the raw
   signed 252d rolling beta; `low_beta_s3` itself returns the sign-flipped BAB
   score, per EXP-001's frozen construction).
3. **Neutralization (dollar AND beta, exact pre-clip):** per rebalance date,
   cross-sectional (across eligible names) OLS of `C` on `[1, beta]`; keep the
   residual. Implemented via the centered-covariance closed form (single
   regressor): `Cc = C - mean_elig(C)`, `Bc = beta - mean_elig(beta)`,
   `slope = mean_elig(Cc*Bc) / mean_elig(Bc**2)`, `resid = Cc - slope*Bc`. This
   is algebraically exact: `sum(resid) = 0` and `sum(resid*beta) = 0` per date
   (standard OLS-residual identity, not an approximation) — i.e. a portfolio
   weighted proportional to `resid` is exactly dollar-neutral AND exactly
   beta-neutral, before any clipping.
4. **Re-standardize + clip:** per-date cross-sectional z-score of `resid` over
   eligible names (`src.signals`-style z-score), clipped to **±2.5** (design.md
   §7). Clipping breaks the exact zero-sum/beta-orthogonality above (both
   restored/quantified in step 6-7); this is expected and reported, not hidden.
5. **Restore exact dollar-neutrality post-clip:** subtract the per-date eligible
   mean of the clipped z once more. (Beta-neutrality is NOT re-forced after this
   point — the realized ex-ante portfolio beta after clipping is reported as a
   diagnostic in the results, not hard-constrained to exactly zero.)
6. **Position + category caps, gross target (design.md §7: gross 200%, position
   cap 10% of gross, category cap 40% of gross):** iterative waterfilling —
   scale the demeaned-clipped score proportionally to hit gross exposure 2.0,
   then repeatedly (i) clip any name exceeding 10% of gross, (ii) scale down any
   category exceeding 40% of gross, (iii) rescale the *not-yet-capped* remainder
   to restore the gross target, until no cap is violated or 50 iterations are
   reached (whichever first). This is an approximate, not machine-precision,
   enforcement — the realized max position, max category exposure, and net
   dollar exposure after capping are reported as diagnostics in the results
   (falsification-relevant if the caps are violated by more than a negligible
   numerical tolerance, e.g. >0.1% of gross, which would indicate a construction
   bug, not an accepted approximation).
7. **No-trade band (design.md §7, turnover control):** sequential pass over
   rebalance dates — if a name's desired change from its previously HELD weight
   is `< 0.5%` of NAV, keep the previous weight instead of trading it.
8. Categories: `data/panel/MANIFEST.json["universe"]` (ticker -> category,
   already committed Phase-1 metadata — not new data).

## Referee invocation (unchanged validation/ code, frozen)

`validation.run_battery.run_battery` — full battery: walk-forward performance at
the cost grid (5/10/20 bps), CPCV Sharpe distribution (`splits.cpcv_splits`,
N=8/k=2, purge 5/embargo 10), PBO via CSCV (needs >=10 stored trials — this will
be the ~7th real trial including EXP-001's 4 + this consult's preregistration
row, so PBO will likely report "insufficient trials" rather than a number; not a
falsification condition by itself), DSR with effective-N, and the signal-canary
suite (label-shuffle / time-shift / random-feature) on the `resid`-based signal
from step 4. Gates are the existing frozen `GATES` dict in `run_battery.py`
(`min_net_sharpe=0.5` full-gate reference; K2's `0.2` is this experiment's own,
looser pre-registered pass/fail line — see Falsification below, these are
different bars for different purposes).

Prices fed to the referee's `--prices` argument are the train/validation close
panel **sliced to start one trading day before `panel_start` (1999-12-21 or the
nearest prior session) through `trainval_end`** — NOT the full 1993-2021 history
— so performance/Sharpe/years-positive are measured over the live-tradable
window only. The provider itself internally loads the FULL close/eligibility
history from the canonical Phase-1 artifacts (needed for signals' trailing
lookbacks) independent of the sliced `--prices` argument, and asserts the sliced
argument's values match the canonical file exactly at every overlapping cell
(integrity check against silent staleness) before returning weights aligned to
the sliced index.

## Falsification criterion

**K2 (design.md §10, the specific pre-registered kill criterion for this
question):** net Sharpe (5bps, full train/validation walk-forward) `< 0.2` ->
Tier-1+M0 null; do not proceed to M1/M2 without a new logged trial and design
consult. `0.2 <= net Sharpe < 0.4`: ambiguous zone — per D25/design-reviewer,
this specific outcome (not intuition) is the pre-registered trigger to consider
escalating the S6 long-horizon-reversal candidate (running in parallel as a
non-blocking, non-trial-counted `factor-screener` exploratory pass, see below)
to a confirmatory pre-registration. `>= 0.4`: proceed toward M1 admission
(design.md §6: CPCV-median net Sharpe >= M0 + 0.1 with PBO no worse, a SEPARATE
future trial, not decided here). Canary trip on the `resid` signal at step 4
halts "validated" status regardless of the Sharpe reading (mission §3.3d/K4).
`net Sharpe > 1.5` is the mission's §3.3f pre-declared suspicious-good-news bound
— triggers the adversarial-audit path before any positive reading is trusted.

## Non-blocking parallel exploratory work (explicitly NOT part of this trial)

Per D25, one `factor-screener` exploratory pass on **S6** (long-horizon reversal,
`-zscore(P(t-252)/P(t-1260) - 1)`, skip-12-months) may run in parallel on the
same train/validation data. Its result does not influence this trial's gates,
is not trial-counted, and per the design-reviewer's refinement #3, its launch is
sequenced AFTER this spec is frozen (this file's commit) precisely so the
non-influence claim is enforced by ordering, not stated intent alone.

## Planned trial count

**1** — the M0 combiner is a single fixed construction (no hyperparameter scan:
S1/S2 already frozen by EXP-001, caps/gross/no-trade-band are the already
operator-approved design.md §7 constants, not searched). Logged via
`validation.run_battery`'s own `log_trial` call (`phase="trial"`,
`experiment_id="PHASE3-M0"`) — the referee's standard logging path, not a
separate manual log_trial call. This is the 5th real trial against the <=250
budget (EXP-001 spent 4).

## Seeds / config

Deterministic given the frozen train/validation panel (no randomness in the
construction above); the referee's own canary suite uses `seed=0` internally
(`validation/canaries.py` default), unchanged.
