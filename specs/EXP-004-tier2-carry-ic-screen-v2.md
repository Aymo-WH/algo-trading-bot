# EXP-004 — Tier-2 Carry (Bond Sleeve) IC Screen v2 (frozen before first run)

**Supersedes `specs/EXP-004-tier2-carry-ic-screen.md` (v1), which is ABANDONED
before any FRED data was pulled or any real computation happened.** v1 was
reviewed by a design-reviewer consult (2026-07-11) that found it substantively
broken: the proposed per-asset TS z-score construction quietly turned "raw
carry" into a timing-residual test (defeating the whole point of the
static/timing decomposition diagnostic), the holdout-period FRED
clear-storage rationale inverted its own mirror argument and silently
contradicted D18's logged precedent, publication-lag/look-ahead handling was
unaddressed, and several forks (AGG/BND duration mapping, TIP's real-vs-
nominal mismatch) were left unpinned. v1's file cannot be edited once written
(the quarantine guard freezes any file under `specs/` on creation, not on
review) — hence this new versioned file, matching the
`REFEREE-SELFTEST-2026-07-08-v2.md` precedent. All of v1's substance is
carried forward corrected; nothing here has touched real data.

Frozen: 2026-07-11. Sequenced per D28 (green-lit), D31 (5 requirements from
the EXP-003 consult), D32 (broadened scope, credit-spread data included),
D33 (canary-interpretation rule), D34 (the deferred book-identity decision
this experiment's diagnostics feed evidence into).

## Hypothesis

A single combined carry signal (S5), built from FRED yield/spread LEVELS
(not any per-asset own-history normalization — see "Construction" below)
across a 10-name bond sleeve, clears the SAME unchanged Tier-1 graduation
rule S1-S4/EXP-001 used (`validation.ic.passes_graduation`: mean_ic>=0.01,
t_nw>=2.0, pct_years_positive>=0.60), measured within the sleeve
(`min_names=10`, full-coverage dates only — see "Cross-sectionalization").

**Scope note, unchanged from v1:** the treasury-leg component is, in
substance, closer to a duration/term-premium signal (six names on one
shared curve) than to classic multi-asset carry; the credit leg (genuinely
a different risk premium) is what makes S5 defensible as a single "carry"
construction. If S5 graduates, report which leg carries the result (the
leg-attribution diagnostic exists exactly for this) — not just the combined
number. FX (UUP) remains explicitly OUT OF SCOPE (needs foreign short rates,
a separate future data dependency regardless of anything decided here).

## Construction — LEVEL-based excess carry, not TS z-scoring (v1's error)

v1 z-scored each asset's carry against its own expanding history before
ranking. That operation IS `src/decomposition.py`'s timing residual
(subtracting a per-asset expanding mean) — so v1's "raw" signal was already
a timing-like object, which would have made the static/timing diagnostic
near-circular and the canary pre-declaration internally wrong (a residual
input predicts residual-class canary behavior on what was labeled the "raw"
leg). Fixed here: both legs are expressed as a LEVEL, in the same economic
unit (excess yield, % p.a.), so cross-sectional ranking is a genuine
raw-carry test, and the unchanged `expanding_static_timing` decomposition
(applied downstream, descriptively) does its actual intended job of
splitting THIS level into static tilt vs timing.

- **Treasury leg (6 names: SHY, IEF, TLT, AGG, BND, TIP):** excess carry =
  duration-matched nominal Treasury CMT yield minus the 3-month T-bill
  yield (a short-rate reference within the same FRED Treasury-yield family
  approved at D28). **TIP is kept in the sleeve** (not dropped — see
  "Why TIP stays" below), using a nominal-equivalent construction: TIPS
  real CMT yield at the matched tenor PLUS the FRED breakeven-inflation
  rate at the same tenor (e.g., 10y TIPS real yield + 10y breakeven —
  both are standard, closely-related FRED series derived directly from the
  same Treasury/TIPS curves already being pulled; this is disclosed as an
  ADDITIONAL FRED series, breakeven inflation, beyond D28/D32's named ones
  — same platform, same data family, flagged for completeness not hidden).
- **AGG/BND/TIP duration mapping (pinned, not left to implementation):**
  each fund's carry point uses the nearest CMT/TIPS tenor to that fund's
  PUBLISHED effective duration as of this spec's freeze date (a fixed,
  stated reference number per fund, recorded in the implementation
  alongside the confirmed FRED series IDs); ties broken by rounding DOWN
  to the shorter tenor. This is a fixed rule, not a free parameter.
- **Credit leg (4 names: LQD, HYG, JNK, EMB):** excess carry = the ICE BofA
  option-adjusted spread (OAS) for the matching credit bucket (IG for LQD,
  HY for HYG/JNK, EM for EMB) directly — already an excess-yield-over-
  treasuries quantity, no further adjustment.
- **Point-in-time / publication-lag handling (pinned; unaddressed in v1):**
  FRED Treasury/TIPS CMT and ICE BofA OAS series typically post with
  roughly a 1-business-day lag relative to their observation date. Each
  name's carry value used at rebalance date `t` is the most recent
  FRED-dated value at or before `t minus 1 business day` (a conservative
  as-of merge), with a max-staleness rule: if the most recent available
  value is more than 5 business days stale, the name's carry is NaN at
  that date (falls out of that date's cross-section under
  `min_names`, below). These are market-quote series (not survey data)
  and are not meaningfully revised after publication, so no ALFRED
  vintage-tracking dependency is needed — noted once, not re-derived.
- **History source (pinned):** carry values are computed from FRED's FULL
  published history (which predates several ETFs' own listing dates) —
  this is point-in-time public index/curve data, not derived from our
  panel, so using its full history introduces no look-ahead. Tradability
  gating happens separately, through the existing `eligibility_trainval`
  mask and `fwd5` alignment `cross_sectional_ic` already applies — carry
  history is not additionally masked to each ETF's own eligibility window.

**Why TIP stays in a 10-name sleeve rather than dropping to 9 (resolves a
tension the prior consult's own two recommendations left unstated):**
`validation/canaries.py`'s `label_shuffle_canary`/`time_shift_canary`/
`random_feature_canary` call `cross_sectional_ic` internally WITHOUT
passing `min_names`, so they always use `validation/ic.py`'s hardcoded
module default (`MIN_NAMES = 10`) — frozen code, not overridable from
here. A 9-name sleeve could never satisfy "at least 10 defined values,"
so the entire canary suite would silently return undefined/degenerate
results for every date if TIP were dropped. Keeping the sleeve at exactly
10 (via the breakeven-adjusted TIP construction above, rather than
dropping it) is what makes "diagnostic canaries, same frozen machinery"
actually true rather than aspirational.

## Cross-sectionalization (pinned)

**Within-sleeve ranking, full coverage.** Confirmatory test:
`cross_sectional_ic(S5_carry, fwd5, min_names=10)` — explicitly passed
(matching the sleeve's actual size, not silently relying on the module
default coinciding with it), and consistent with what the frozen canary
functions assume internally. Consequence, disclosed ex-ante: a date scores
only when ALL 10 names have a defined (non-stale) carry value AND are
panel-eligible — the binding constraint is each ETF's OWN listing date in
our panel (EMB 2008-12-17, JNK 2008-12-02, HYG/BND 2008-04, per the prior
consult's direct measurement), not any signal-construction burn-in (the
level-based fix has none). Effective start ≈ late 2008/early 2009 —
recovering most of 2009 (a highly carry-informative year) that a
z-score-burn-in construction would have excluded, though still missing the
most acute Sept-Oct 2008 crisis moves. **Same power caveat as v1, restated:**
the t>=2.0 bar was observed on a ~70-name cross-section; with 10 names,
per-date rank IC is measurably noisier, so the same nominal threshold is a
harder bar here than it was for S1-S4. Disclosed ex-ante, not invoked
post-hoc to explain a result either way.

## Descriptive diagnostics (NOT part of the pass/fail rule, NOT separate trials)

- **Leg attribution:** S5_carry's IC recomputed restricted to (a) the
  6-name treasury leg (`min_names=6`, explicitly overridden — the 10-name
  default would make this all-NaN by construction) and (b) the 4-name
  credit leg (`min_names=4`, same reason). Disclosed: a 4-point Spearman
  correlation takes values on a coarse discrete grid — reported as
  descriptive texture, not equivalent-strength evidence to the 10-name
  confirmatory test.
- **Static/timing decomposition** (D31 requirement 1): `src/decomposition.py
  ::expanding_static_timing` applied to `S5_carry` UNCHANGED
  (`MIN_PRIOR_OBS=52`, no retuning) — now meaningful because S5_carry is a
  genuine level, not an already-mean-subtracted object. Both components'
  IC reported, neither graded, neither logged as a trial. **This diagnostic
  is the evidence stream for D34's deferred book-identity decision** (how
  much of any carry edge is static tilt vs timed) — stated explicitly here
  since v1 never tied the two together.
- **Correlation vs the existing library** (D31 requirement 4): `S5_carry`
  (raw, static, timing) against S1/S2 (raw from EXP-001, static/timing from
  EXP-003) — the number that answers whether the FRED dependency bought
  real breadth or re-found the same static ordering the existing signals
  already carry.

## Canaries (diagnostic per D30/D33, non-blocking, pre-declared BEFORE running)

Corrected expectation, now consistent with a level-based raw signal:

- **Raw S5_carry is EXPECTED to plausibly fail time_shift via the
  long-persistence/decay-but-retain mechanism** (yield and spread LEVELS
  move slowly; a 26-week-stale reading can retain much of its live
  correlation — same family as S1's raw 81% retention), NOT via tilt
  re-injection (that mechanism applies to residual-class objects; raw
  S5_carry here is a genuine level, unlike v1's accidental z-scored
  version).
- **The timing component of S5_carry is EXPECTED to plausibly fail
  time_shift via tilt re-injection** (D33's named mechanism): the residual
  re-acquires part of the static tilt when staled, the same structural
  property `src/decomposition.py`'s construction has for any input.
- **Adjudication (D33's two-condition test, ratified this session):**
  either canary trip is attributable to its expected benign mechanism ONLY
  IF (i) an identity decomposition quantitatively accounts for the
  lagged-vs-live gap (the same arithmetic leak-hunter used for EXP-003,
  computed directly on the real S5_carry/timing panels) AND (ii) an
  empirical truncation attack confirms no look-ahead. Failing either
  condition, the trip is a live, unexplained finding requiring full
  root-cause — no exemption for this signal class, matching D33 exactly.

## Falsification criterion

S5_carry falsifies (this Tier-2 candidate fails) if it misses any leg of
the unchanged graduation rule on the within-sleeve cross-section. Not
eligible for rescue by reparameterizing the duration-mapping rule, the
publication-lag buffer, the breakeven adjustment, or the
cross-sectionalization choice without a new logged trial (same rule as
design.md §5 and EXP-001/EXP-003's falsification clauses). The
leg-attribution and correlation diagnostics may explain why it failed or
passed; they do not license silently retrying a variant.

## Planned trial count

**1** — `EXP-004-S5_carry`, the combined within-sleeve rank IC against the
unchanged graduation rule. Leg attribution, static/timing legs, canaries,
and cross-library correlation are descriptive/diagnostic only — same
pattern as EXP-001/EXP-003. Trial budget entering this experiment: 8/250
(per D31); this adds 1, bringing the count to 9/250 regardless of outcome.

## Holdout-period FRED data (corrected — v1's rationale was wrong)

v1 proposed keeping 2022-2026 FRED data in the clear, reasoning it mirrors
how train/val prices are clear. That inverted its own analogy: the price
panel keeps train/val prices clear and LOCKS holdout-period prices — the
correct analogue is train/val FRED clear, HOLDOUT-PERIOD FRED LOCKED. This
also would have silently contradicted D18 (holdout-period volume
deliberately not retained, "a needless quarantine surface"), unacknowledged
in v1. Corrected here, no new operator decision needed (this restores
consistency with D18's already-logged principle rather than deviating from
it): FRED data is pulled once for full history (train/val + holdout, for
§3.3b reproducibility — a future re-pull is not guaranteed bit-identical),
but holdout-period (2022-01-01 onward) rows are encrypted into the lockbox
via the same `validation.lockbox.build_lockbox` mechanism the price panel
uses (D12/D21 precedent), not kept in a clear working file. Implementation
detail (which script, which commit) follows Phase 1's `panel_factory.py`
pattern; not re-litigated here.

## Seeds / config

None for the IC/graduation test — deterministic once the FRED data and
existing frozen signal code are fixed. Canary functions use their existing
default `seed=0` (unchanged from `validation/canaries.py`), for exact
reproducibility.

## Compute estimate (§3.6)

EXP-003's canary suite took 11m44s on the 70-name panel's ~1096-1148-date
window (canaries.py's per-date Python loop, repeated ~150x per signal).
This experiment's confirmatory + canary battery runs on a 10-name sleeve
over a shorter effective window (~2009-2021, ~650 rebalance dates vs
EXP-003's ~1096) — expect meaningfully cheaper, roughly 3-6 minutes for
the full battery (1 raw signal + 1 timing diagnostic, each canaried),
scaling down from EXP-003's per-signal cost by both fewer dates and (for
the per-date IC loop specifically) fewer columns to rank per date. Will be
stated as an estimate before the real run per the existing rule, and any
2x+ divergence flagged when reporting.
