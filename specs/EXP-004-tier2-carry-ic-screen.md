# EXP-004 — Tier-2 Carry (Bond Sleeve) IC Screen (frozen before first run)

Frozen: 2026-07-11, before any FRED data is pulled or any carry value is
computed. This is item "A" from the 2026-07-10 session resume (Tier-2 carry,
design.md §5 S5, deferred at D6), green-lit at D28, sequenced after EXP-003
(closed out, D31). Scope broadened at operator direction (this session,
AskUserQuestion) beyond D28's literal "FRED treasury yields" to include
FRED credit-spread series for the high-yield/EM/investment-grade names —
logged as **D32** in decisions.md, with the operator's explicit awareness
that this is a second new FRED series beyond the one named at D28. Freezes
the 5 requirements from the 2026-07-11 design-reviewer consult on EXP-003
(journal.md) plus D33's canary-interpretation ratification and D29's
sign-flip prior-exposure clause (not directly triggered here, but the
non-rescue discipline it embodies applies equally).

## Hypothesis

A single combined carry signal (S5), built from FRED yield/spread levels
across the 10-name bond sleeve, clears the SAME unchanged Tier-1 graduation
rule S1-S4/EXP-001 used (`validation.ic.passes_graduation`: mean_ic>=0.01,
t_nw>=2.0, pct_years_positive>=0.60), measured within the bond sleeve only
(not the full 70-name panel — see "Cross-sectionalization" below).

**Explicit scope note (economic honesty, not hedging):** the treasury-leg
component of S5 is, in substance, closer to a duration/term-premium timing
signal (all six names sit on one shared curve) than to classic multi-
asset-class carry. Combining it with the credit-leg component (LQD/HYG/JNK/
EMB, genuinely a different risk premium: compensation for credit risk, not
duration) is what makes S5 a reasonable single "carry" construction in the
KMPV/AQR sense. This distinction is stated here so it is not silently lost —
if S5 graduates, the report to the operator must characterize which leg is
carrying the result (see "Descriptive diagnostics" below), not just the
combined number.

**FX (UUP) is explicitly OUT OF SCOPE for this experiment.** FX carry needs
foreign short-term rates, a data dependency distinct from anything named at
D28/D32 regardless of which carry-scope option was chosen. Deferred to a
separate, future, explicitly-flagged data-dependency conversation — not
smuggled in here.

## Data dependency (NEW — not yet pulled)

FRED series needed (economic description; **exact series IDs to be confirmed
against the live FRED API at implementation time, not asserted in this
frozen spec** — verifying a wrong ticker now would freeze an error):
- Nominal Treasury constant-maturity yields spanning the treasury-leg names'
  approximate durations (short/1-3y for SHY, intermediate/7-10y for IEF,
  long/20y+ for TLT; AGG/BND approximated via a duration-weighted blend or
  nearest point — exact mapping is an implementation-time detail, not a
  free parameter tuned on results).
- TIPS constant-maturity REAL yields (for TIP specifically — nominal yields
  are the wrong economic quantity for an inflation-protected instrument).
- ICE BofA option-adjusted spread (OAS) indices for investment-grade (LQD),
  high-yield (HYG, JNK), and emerging-market (EMB) credit.

**Holdout-period (2022-01-01 to 2026-06-30) FRED data is stored IN THE
CLEAR, alongside train/val — not lockboxed.** Rationale, resolving the
open question from the prior consult: the Prime Directive (CLAUDE.md,
FABLE_MISSION.md §3.1) permits computing features on every date, including
test dates — what is forbidden is computing or viewing STRATEGY PERFORMANCE
(returns/Sharpe/P&L) on the holdout. FRED yield/spread LEVELS are a feature
input, not a return series of OUR universe; they cannot by themselves
reveal strategy P&L the way holdout CLOSE prices could (which is why those
specifically are lockboxed, D12/D13). This mirrors how the existing price
panel keeps train/val prices in the clear and only locks the holdout
close-price/return combination. Binding constraint: no code anywhere may
combine clear-text holdout FRED data with the LOCKED holdout price/return
panel before the one-shot final eval — the quarantine guard's existing
lockbox block on `data/lockbox/` remains the mechanical enforcement; this
spec adds no new lockbox.

## Exact construction

**Step 1 — per-asset carry level.** For each of the 10 bond names, a single
point-in-time carry level per rebalance date: yield (nominal or real) for
the 6 treasury-leg names, OAS spread for the 4 credit-leg names (LQD, HYG,
JNK, EMB).

**Step 2 — per-asset TS z-score (point-in-time, expanding).** Because yield
levels (percent) and OAS spreads (basis points) are not directly comparable
across the two legs, each name's OWN carry level is expressed as an
expanding, point-in-time z-score against that asset's own history — same
`MIN_PRIOR_OBS=52` convention as `src/decomposition.py` (52 prior weekly
observations of that asset's own carry level required before its z-score is
defined; this is a NEW small helper alongside `expanding_static_timing`,
since z-scoring needs an expanding std as well as an expanding mean — to be
implemented, not invented at spec-freeze time). Sign convention: higher
z-score = more attractive carry (yield or spread higher than that asset's
own typical level).

**Step 3 — cross-sectional rank (the actual S5_carry signal).** At each
rebalance date, Spearman-rank the TS-z-scored carry values across whichever
of the 10 names have a defined value that date. This is `S5_carry(t, a)`,
tested via the unchanged `validation.ic.cross_sectional_ic` machinery
exactly as S1-S4/EXP-001 were, against the same `fwd5` label.

## Cross-sectionalization (pinned, per the prior consult's requirement)

**Within-sleeve ranking, NOT the full 70-name panel.** Carry as constructed
here is only economically meaningful compared within the bond sleeve (a
carry-attractiveness rank has no defined meaning against, say, an equity
sector ETF, which has no carry construction in this experiment). Explicit
power caveat, stated so a marginal result is not over-read: the existing
t>=2.0 graduation bar was observed against a ~70-name cross-section
(S1-S4); with at most 10 names, per-date rank IC is measurably noisier
(fewer points per Spearman correlation), so std_ic is expected to run
higher than S1-S4's, and the SAME nominal threshold is a harder bar to
clear here than it was for the wide-panel signals. This is disclosed
ex-ante, not discovered post-hoc to explain a result either way.

## Descriptive diagnostics (NOT part of the pass/fail rule, NOT separate trials)

- **Leg attribution:** S5_carry's IC recomputed restricted to (a) the
  6-name treasury leg only and (b) the 4-name credit leg only. Answers which
  leg is carrying any result — required given the scope note above, so a
  graduating S5 is not silently reported as "carry works" when it might be
  entirely duration-leg or entirely credit-leg.
- **Static/timing decomposition** (per the prior consult's requirement 1):
  `src/decomposition.py::expanding_static_timing` applied to `S5_carry`
  UNCHANGED (`MIN_PRIOR_OBS=52`, no retuning). Both components' IC reported,
  neither graded against the graduation rule, neither logged as a trial —
  same descriptive treatment EXP-003 gave S1/S2's static legs.
- **Correlation vs the existing library** (requirement 4): `S5_carry` (raw,
  static, timing) against S1/S2 (raw from EXP-001, static/timing from
  EXP-003). This is the number that actually answers whether the new FRED
  dependency bought real breadth or just re-discovered the same ordering
  the existing signals already carry.

## Canaries (diagnostic per D30/D33, non-blocking, pre-declared BEFORE running)

Per the prior consult's requirement 2 and this session's D33 ratification,
the expected signature is stated here in advance, with the two-condition
adjudication test that determines whether a trip is benign or a live issue:

- **Raw S5_carry is EXPECTED to plausibly fail time_shift via the
  long-persistence mechanism already seen in S1/S2's raw signals** (yield
  and spread LEVELS move slowly; a 26-week-stale reading of either can
  retain much of its live correlation, same family as S1's 81%/96%
  retention, journal.md 2026-07-10 retroactive check) — a decay-but-retain
  pattern, not the stronger tilt-re-injection pattern.
- **The timing component of S5_carry is EXPECTED to plausibly fail
  time_shift via tilt re-injection**, the mechanism leak-hunter identified
  for EXP-003 (`timing(t-26) = [W(t-26)-static(t)] + [static(t)-static(t-26)]`,
  the second term re-acquiring part of the static tilt).
- **Adjudication (D33's two-condition test, applied here for the first
  time since ratification):** either canary trip is attributable to its
  expected benign mechanism ONLY IF (i) an identity decomposition
  quantitatively accounts for the lagged-vs-live gap (the same arithmetic
  leak-hunter used, computed directly on the real S5_carry panel) AND
  (ii) an empirical truncation attack confirms the construction uses no
  future information. Failing either condition means the trip is treated
  as a live, unexplained finding requiring full root-cause before any
  further characterization — exactly the same standard any other canary
  trip gets, no special exemption for this signal class.

## Falsification criterion

S5_carry falsifies (this Tier-2 candidate fails) if it misses any leg of
the unchanged graduation rule on the within-sleeve cross-section. Not
eligible for rescue by reparameterizing the TS-z-score window, the carry
construction, or the cross-sectionalization choice without a new logged
trial (same rule as design.md §5 and EXP-001/EXP-003's falsification
clauses) — the leg-attribution and correlation diagnostics above may explain
*why* it failed or passed, but do not license silently retrying a variant.

## Planned trial count

**1** — `EXP-004-S5_carry`, the combined within-sleeve rank IC against the
unchanged graduation rule. Leg attribution, static/timing legs, canaries,
and cross-library correlation are descriptive/diagnostic only, logged in
the results file but not as separate ledger trials — same pattern as
EXP-001's inter-signal correlation and EXP-003's static components. Trial
budget entering this experiment: 8/250 (per D31); this adds 1, bringing the
count to 9/250 regardless of outcome.

## Seeds / config

None for the IC/graduation test — deterministic once the FRED data and
existing frozen signal code are fixed. Canary functions use their existing
default `seed=0` (unchanged from `validation/canaries.py`), for exact
reproducibility.
