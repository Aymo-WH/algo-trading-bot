# EXP-004 — Tier-2 Carry (Bond Sleeve) IC Screen v3

Frozen: 2026-07-11. Supersedes `specs/EXP-004-tier2-carry-ic-screen-v2.md`
(itself superseding v1). Sequenced per D36 (v2's credit leg found
structurally infeasible pre-holdout, live-verified), D37 (operator decision
to build a yfinance-derived credit-leg proxy rather than drop the leg or
defer the experiment), and D38 (operator decision on holdout-period storage:
a second, independent lockbox). Reviewed by a design-reviewer consult on the
full draft before freezing (REFINE verdict — 4 blocking corrections
incorporated: a false "no splits" claim corrected, LQD's duration re-sourced
directly, the holdout-storage gap resolved via D38, and the falsification
dial-list extended; plus 4 non-blocking refinements incorporated: the v2
scope note restated, ex-date-count diagnostics added, the disclosed-noise
episode attribution corrected to measured evidence, and a tz-handling note).

**Supersedes `specs/EXP-004-tier2-carry-ic-screen-v2.md`, which is not wrong but
INFEASIBLE**: v2's credit leg (LQD/HYG/JNK/EMB) sourced OAS directly from FRED's
ICE BofA series, which turned out (D36, live-verified this session) to have a
rolling ~3-year trailing history window — first observation 2023-07-11,
entirely inside the frozen 2022-01-01..2026-06-30 holdout. Zero usable
train/val dates. No point-in-time rescue exists (ALFRED vintage archives
checked and equally truncated). This is a data-access wall, not a construction
error — v2's logic is sound, it just cannot be fed real data pre-holdout.

Per D37 (operator decision): rather than drop the credit leg (treasury-only
is a free but severely underpowered fallback — 6 names collapse to 4 distinct
carry values under the pinned tenor rule) or defer Tier-2 carry entirely
(literature rates carry as the single highest-conviction candidate signal in
the library, D6), build a credit-leg proxy from data already inside the
project's approved perimeter: `yfinance` (Phase-1-approved, no new
dependency). The treasury leg (6 names) is UNCHANGED from v2 — this document
only replaces the credit-leg construction and re-states what's needed for
the combined 10-name sleeve.

## Hypothesis (unchanged from v2)

A single combined carry signal (S5_carry), built from FRED yield LEVELS
(treasury leg) and a yfinance-derived trailing-yield-spread proxy (credit
leg) across the 10-name bond sleeve, clears the unchanged Tier-1 graduation
rule (`validation.ic.passes_graduation`: mean_ic>=0.01, t_nw>=2.0,
pct_years_positive>=0.60), measured within the sleeve (`min_names=10`,
full-coverage dates only).

## Construction — treasury leg (UNCHANGED from v2 + D35 addendum)

Excess carry = duration-matched nominal Treasury CMT yield minus the
3-month T-bill yield (DGS3MO). Duration-tenor mapping (published effective
duration as of this spec's freeze date, nearest tenor, ties round down):

| Fund | Published eff. duration | Source (dated) | Mapped tenor |
|------|--------------------------|-----------------|--------------|
| SHY  | ~1.9-2.0y | issuer fact sheet (qualitative, well-established) | 2y (DGS2) |
| IEF  | 7.2y | issuer/third-party fact sheet | 7y (DGS7) |
| TLT  | 15.20y (2026-07-08) | issuer data via search | 20y (DGS20) |
| AGG  | 5.78y (2026-03-31) | issuer fact sheet | 5y (DGS5) |
| BND  | ~5.7-5.8y (2026-03/05) | Vanguard fact sheet | 5y (DGS5) |
| TIP  | 6.41y (2026-03-31) | issuer fact sheet | 7y (DGS7) |

TIP's carry = DGS7 minus DGS3MO directly (D35: breakeven = nominal-minus-real
by definition, so no separate TIPS real-yield/breakeven series is needed —
TIP's carry is arithmetically identical in form to the other treasury names').
IEF and TIP share tenor 7 (predicted by D35, confirmed live) — a disclosed,
permanent rank tie between those two names on every date, not a defect.

Point-in-time / publication-lag handling (unchanged from v2): each name's
carry value at rebalance date `t` = the most recent FRED-dated value at or
before `t minus 1 business day`; NaN if the most recent value is more than
5 business days stale.

## Construction — credit leg (NEW: yfinance trailing-yield-spread proxy)

**Definition.** For each credit name (LQD, HYG, JNK, EMB), excess carry at
date `t`:

```
ttm_yield(t)   = sum(distributions with ex-date in (t - 366d, t - 1 business day]) / raw_close(t)
excess_carry(t) = ttm_yield(t) - matched_treasury_cmt(t)
```

- `raw_close`: yfinance daily Close with `auto_adjust=False` (split-adjusted,
  NOT dividend-adjusted — using a total-return-adjusted price as the
  denominator would silently deflate the yield estimate over a 15+ year
  window; this is a SEPARATE pull from the committed panel, which is
  dividend-adjusted for return-calculation purposes and unsuitable here).
  **Correction (design-reviewer caught this — an earlier draft's "verified:
  no splits" claim was false):** JNK had a 1-for-3 reverse split on
  2019-05-06. This is harmless to the construction, not ignored: yfinance
  back-adjusts BOTH `raw_close` (auto_adjust=False still applies split
  adjustment, just not dividend adjustment) and `Ticker.dividends` to the
  same post-split basis throughout history (verified: 2018 monthly dividends
  show ~0.48/share post-split-basis vs the true as-traded ~0.16/share,
  a consistent 3x, matching the price series' equivalent rescaling) — so the
  ttm_yield RATIO is split-invariant across the 2019-05-06 boundary. Any
  future split on any of the 4 names would rescale numerator and denominator
  identically; benign by the same argument, not something to re-check ad hoc.
- `distributions`: yfinance `Ticker(t).dividends` (ex-dividend amounts,
  indexed by ex-date, tz-aware America/New_York 09:30 — normalize to a plain
  date before comparing against rebalance dates or window boundaries, or
  they silently fail to align). Verified this session: LQD from 2002-09, HYG
  from 2007-05, JNK from 2007-12, EMB from 2007-12 — all monthly-ish, ample
  history.
- **Why TTM (12-month trailing), not a shorter window:** this is the
  industry-standard fund-yield definition (Morningstar/ETF.com "TTM Yield"),
  chosen specifically BECAUSE it is an external, pre-existing convention —
  not something whose window length was chosen by looking at this signal's
  IC. A shorter window would react faster to genuine spread-widening (a
  point in its favor) but is more exposed to lumpy/uneven monthly payouts
  and would look like a parameter picked to fit the test. Not eligible for
  retuning without a new logged trial (falsification clause, below).
- **Ex-date, not payment date, is the information-availability boundary**
  (ex-dividend amounts and dates are publicly announced in advance of the
  ex-date itself, so using ex-date is conservative, not optimistic) — same
  spirit as the treasury leg's t-1 business-day buffer, applied here too
  (a distribution with ex-date on or after `t` is excluded from the trailing
  sum used at `t`).
- **Duration-tenor mapping for the matched treasury (published effective
  duration, same nearest-tenor/ties-round-down rule as the treasury leg):**

| Fund | Published eff. duration | Source (dated) | Mapped tenor |
|------|--------------------------|-----------------|--------------|
| LQD  | 7.88y (design-reviewer pulled iShares' own LQD page directly, correcting an earlier draft's sibling-fund proxy citation) | iShares, live-checked this session | 7y (DGS7) |
| HYG  | 2.91y | issuer data, ~2026-04 | 3y (DGS3) |
| JNK  | ~3.08y | third-party source, ~2026-04 | 3y (DGS3) |
| EMB  | 6.61y (2026-03-31) | issuer product brief | 7y (DGS7) |

  Subtracting a duration-matched treasury yield (rather than just DGS3MO,
  which is what the treasury leg subtracts) is deliberate, and confirmed
  correct by design-reviewer: OAS is *already* curve/duration-adjusted, so a
  like-for-like proxy must strip out generic term premium too — otherwise
  the credit leg would mostly re-rank HYG/JNK (~3y duration) against
  LQD/EMB (~7-8y) on DURATION, not credit risk, re-absorbing exactly the
  term-premium exposure D32 added the credit leg to escape. This makes the
  two legs' excess-carry definitions asymmetric in form (treasury leg: yield
  minus cash; credit leg: yield minus duration-matched curve) — inherited
  from v2's own OAS-vs-CMT asymmetry, not introduced here; disclosed, not a
  redesign.
- **Minimum data-sufficiency floor (pinned, not tuned):** `ttm_yield(t)` is
  defined only once at least 8 distinct ex-dividend dates fall within the
  trailing 366-day window (handles near-monthly payers with some tolerance
  for uneven schedules; avoids a partial-history burn-in artifact at each
  fund's own inception). Undefined (NaN) otherwise — falls out of that
  date's cross-section under `min_names`.

**Disclosed limitation (not solved, stated honestly; episode attribution
corrected by design-reviewer's own measurement):** yfinance's `dividends`
field does not distinguish ordinary income from return-of-capital or
special/capital-gains distributions. A genuine OAS is a clean,
option-adjusted market-implied spread; TTM distribution yield is a coarser,
backward-looking proxy that can be distorted by lumpy special distributions.
**Measured, not assumed:** the largest such distortions are YEAR-END SPECIAL
distributions, not general credit-stress episodes — e.g. JNK paid 2.718 on
2010-12-29 vs a ~0.539 median (~5x, roughly +1.9pp of TTM yield for the
following 12 months, large enough to flip JNK-vs-HYG's rank in-sample) and
EMB paid 1.29 on 2018-12-18 (~3x its typical monthly amount). No attempt is
made to detect or filter these after the fact (that would risk tuning the
construction against the very outcome being tested); this is reported as a
standing construction-quality caveat on any S5_carry result, not corrected
away. A descriptive diagnostic — each credit name's derived TTM-yield series
min/max/any negative values, PLUS the min/max count of distinct ex-dividend
dates found in any trailing 366-day window per name (the direct detector for
a Yahoo missing/duplicate-distribution data artifact, distinct from the
special-distribution phenomenon above) — is reported in results as a
data-sanity check, informational only, not part of the pass/fail rule and
not used to adjust the construction.

## Cross-sectionalization (unchanged from v2)

Within-sleeve ranking, full coverage: `cross_sectional_ic(S5_carry, fwd5,
min_names=10)`, explicitly passed. A date scores only when all 10 names have
a defined, non-stale carry value AND are panel-eligible. Unlike the
treasury-only fallback (6 names collapsing to 4 distinct values under the
tenor rule), this construction has only ONE exact tie (IEF/TIP at tenor 7,
inherited from the treasury leg) — the 4 credit names contribute
name-specific yields, not shared curve levels, so no analogous degeneracy
there.

## Scope note (carried forward from v2, restated since this is now the
standalone frozen document)

The treasury leg alone is closer to a duration/term-premium signal (names on
one shared curve) than to classic multi-asset carry; the credit leg is what
makes S5 defensible as a single "carry" construction (D32). If S5 graduates,
report which leg carries the result (leg attribution exists exactly for
this), not just the combined number. The 10-name cross-section is a harder
statistical bar than the ~70-name universe S1-S4 graduated on (same t>=2.0
nominal threshold, noisier per-date rank IC) — disclosed ex-ante, not
invoked post-hoc. FX (UUP) remains out of scope.

## Descriptive diagnostics (unchanged in kind from v2)

Leg attribution (`min_names=6` treasury / `min_names=4` credit), static/timing
decomposition (`src/decomposition.py`, unchanged), correlation vs S1/S2's
static/timing legs — same as v2. Per design-reviewer's Q1 answer: the
credit leg's cross-name ranking is expected to sit closer to a persistent
HY>EM>IG ordering (closer to static tilt) than OAS-based carry would, because
the 366-day numerator smooths fast-moving spread information — but the
denominator (spot price) still moves daily and spikes the yield estimate
when spreads genuinely widen, so timing information is not expected to be
zero. The static/timing decomposition measures this directly and is the
evidence stream D34 needs either way — stated as an expectation, not a
result.

## Canaries (diagnostic per D30/D33, non-blocking, pre-declared BEFORE running)

Same expected-signature framework as v2: raw S5_carry may plausibly fail
time_shift via long-persistence/decay-but-retain (yield levels move slowly);
the timing component may plausibly fail time_shift via tilt re-injection
(D33's named mechanism). D33's two-condition adjudication test applies
unchanged. One addition specific to this construction: the credit leg's TTM
yield is itself a 366-day trailing average, so it is *mechanically* slow-
moving/persistent in a way the treasury leg (instantaneous CMT levels) is
not — this is disclosed ex-ante as an ADDITIONAL, expected reason
(alongside D33's tilt-re-injection) the raw signal specifically might retain
correlation under a 26-week shift; not invoked post-hoc to explain a result
either way.

## Falsification criterion (unchanged from v2, dial list extended)

S5_carry falsifies if it misses any leg of the unchanged graduation rule.
Not eligible for rescue by reparameterizing the TTM window length, the
minimum-distributions floor, the duration-tenor mapping, the
cross-sectionalization choice, the ex-date information-availability
convention, or the raw-close (vs. adjusted-close) denominator choice
without a new logged trial. All of these are locked dials, not just the
ones inherited unchanged from v2.

## Planned trial count

**1** — `EXP-004-S5_carry` (same trial ID as v2 would have used; v2 never ran,
so this is not a re-run, it is the first real execution of this hypothesis
under a corrected, runnable construction). Budget entering this experiment:
8/250 (D31); this adds 1, bringing the count to 9/250 regardless of outcome.

## Holdout-period data (corrected per design-reviewer + D38 operator decision)

Both FRED and yfinance data are pulled for full history (train/val +
holdout, for §3.3b reproducibility). Holdout-period (2022-01-01 onward) rows
are encrypted, NOT kept in any clear working or committed file — and this
now covers what an earlier draft missed: not just the derived S5_carry
values, but the RAW yfinance inputs too (daily close + dividends for
LQD/HYG/JNK/EMB), since those raw closes are themselves holdout-period price
history for 4 panel names — exactly the surface D12/D18/D21 exist to
protect for the committed panel.

**D38 (operator decision): a SECOND, independent lockbox**, distinct from
the existing price-panel lockbox (`data/lockbox/holdout.enc` /
`/workspace/OPERATOR_TOKEN.txt`, which stay untouched — `build_lockbox`
refuses to overwrite an existing enc/token pair by design, so reusing those
paths is not an option regardless). New paths: lockbox directory
`data/lockbox_carry/` (holding `holdout.enc`), token file
`/workspace/OPERATOR_TOKEN_CARRY.txt` (outside the repo, same convention as
the existing token). The operator secures this second token off-pod after
the lockbox is built, same procedure as the original (D20-L1 precedent).
Consequence, disclosed: `validation/final_eval.py` currently only knows
about the one price lockbox; consuming this second lockbox's contents at
Phase 6 requires a logged, reviewed change to final_eval at that time — not
now, and not something this experiment's run needs today (train/val never
touches either lockbox). Chosen over the D18-pattern no-retention
alternative because it keeps holdout-period carry bit-reproducible now
(rather than depending on a yfinance/FRED re-pull years from now being
stable), at the cost of one more token to secure and one more piece of
frozen-code surface to touch later.

## Seeds / config

None for the IC/graduation test — deterministic once FRED/yfinance data and
existing frozen signal code are fixed. Canary functions use `seed=0`
(unchanged default).

## Compute estimate (§3.6)

Same battery as v2 would have run (1 raw signal + 1 timing diagnostic, each
canaried) on the same ~10-name sleeve, ~2009-2021 effective window (~650
rebalance dates). Additional one-time cost: a fresh yfinance pull for 4
tickers' daily raw close + dividends (small, seconds-scale) and a TTM-yield
computation (simple rolling sum, no per-date Python loop). Expect the same
3-6 minutes v2 estimated for the confirmatory+canary battery, plus under a
minute for the additional data pull/construction. Will be stated precisely
before the real run; any 2x+ divergence flagged when reporting.
