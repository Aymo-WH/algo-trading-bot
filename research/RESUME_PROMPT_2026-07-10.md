# Resume prompt — paste this to start the next Gordian v2 session

Resume Gordian v2. Read CLAUDE.md and FABLE_MISSION.md in full, then read
research/journal.md and research/decisions.md (through **D27**) to confirm
current state before doing anything. Repo: /workspace/algo-trading-bot,
branch `research/gordian-v2`, last commit **d5c0d8d** (working tree was
clean, everything pushed, as of 2026-07-10).

## What happened last session (summary — full detail is in journal.md)

1. Fixed a stale test assertion (`trial_count()==0`) that had gone stale
   after EXP-001 legitimately logged real trials — confirmed with a fresh
   reviewer, fixed, committed.
2. Resolved D22 (a prior unilateral test fix) per operator ruling:
   revert-and-redo with reviewer sign-off obtained *before* recommitting
   this time — logged as **D24**.
3. Ran the S1/S2 redesign consult (design-reviewer verdict: REFINE).
   Operator approved proceeding to Phase 3 with S1+S2 as-is, plus a
   non-blocking exploratory screen on a new candidate (long-horizon
   reversal, named **S6**) — logged as **D25**. S6 failed as specified
   (real continuation, not reversal, in this universe); a sign-flipped
   variant would pass but was correctly NOT promoted (HARKing risk) —
   operator agreed to treat it as a legitimate idea to test properly later,
   not via the sign-flip shortcut.
4. Built and ran **EXP-002** (Phase 3 M0 combiner: S1+S2 equal-weight
   composite, beta-neutralized, capped to 200% gross). **Result: NOT
   validated.** The `time_shift` leak canary tripped (a signal artificially
   staled by 6 months scored HIGHER than the live one) — leak-hunter ruled
   out actual leakage (bit-identical truncation attack) and root-caused it
   as the edge being substantially an untimed static tilt plus a
   beta-hedge-decay artifact. A separate reviewer pass found and fixed real
   implementation bugs (cap violations from an unsanctioned no-trade-band
   rescale) — fixed by dropping the no-trade band to Phase 4 (**D26**,
   operator-approved) and re-running (v2). Net Sharpe (5bps) landed at 0.21,
   in the pre-registered K2 "ambiguous 0.2–0.4" zone, but that's moot — the
   canary trip blocks "validated" status regardless, per the frozen spec.
   Trial budget: 6/250 spent (4 from EXP-001 + 2 from EXP-002 v1/v2, the
   extra run disclosed honestly as an audit-driven bugfix rerun).
5. Per operator request, retroactively ran the canary suite on S1 and S2
   individually (diagnostic only, no new trial): **both also fail
   time_shift** (S1 retains 81% of its base t-stat after staling, S2 retains
   96%) — confirming the canary trip isn't purely a combiner artifact. But
   the failure pattern differs from M0's: S1/S2 merely decay too slowly
   (plausibly mechanical, from ~12-month lookback windows sharing most days
   across a 26-week shift), while M0 uniquely *strengthens* (a
   combiner-specific beta-hedge-decay effect).
6. **D27** (operator-directed governance change): broadened the
   `design-reviewer` consult from an event-triggered "before presenting"
   check to a **standing rule across every phase** — consult at phase
   boundaries, after major audit findings, and whenever weighing a new
   signal/combiner/direction, with authority to proactively flag or
   redirect. Formalized in FABLE_MISSION.md §3.7/§9 and CLAUDE.md (both
   **local-only, not tracked in git** — the repo is public and these are
   deliberately excluded, per the existing `.gitignore` comment; this was
   confirmed intentional, not an oversight, this session).
7. Demonstrated D27 immediately via a **Fable-pinned architecture consult**
   on signal diversification, combiner alternatives, and backtesting's
   limitations. Headline finding: decomposing M0's canary result together
   with S3's and S6's wrong-sign results suggests the current signal library
   substantially captures the **unconditional cross-asset risk-premium
   ordering** (risky beats safe) rather than genuine timed alpha — and the
   quarantined 2022–2026 holdout begins with the regime shock (rate hikes)
   that inverted exactly that ordering. Flagged as the project's single
   largest known risk, not solvable by more validation rigor — a bet on
   whether that ordering persists, not a bug.

## Open decision points — NOT yet acted on, need operator direction

**A. Tier-2 carry signal (bond + FX legs) — Fable's top recommended next
action.** Needs a **new data dependency (FRED treasury yields)** — per
design.md's own rule, this requires explicit operator awareness/sign-off
before pulling, same as any new data source. Fable recommended building a
static-vs-timing decomposition diagnostic into this screen from the start
(given the now-established pattern that carry-type signals often show some
static-tilt behavior too), rather than being surprised by it again.

**B. S1/S2 static-vs-timing decomposition** — cheap (1–2 trials), needs NO
new data. Splits each signal into a static component (expanding-window mean
tilt) and a timing component (residual), tests the timing component alone
against the existing graduation rule. Directly answers "is there any real
timed information in this universe at all" — recommended as a quick,
low-cost complement to (A), not mutually exclusive.

**C. Explicitly declined by Fable's recommendation, pending operator
ratification:** any sign-flipped S3 (BAB) or S6 (reversal→continuation)
variant — named as the same risky-beats-safe trade the canary already
caught, and a HARKing trap. Not touched; operator has NOT yet formally
ratified declining these, just implicitly agreed via the "treat as normal,
test properly later" stance on S6 specifically.

**D. M1 (ML combiner)** — Fable recommends deferring until ≥3 economically
distinct signals exist (currently only 2, highly correlated). Not started.

**E. Phase 4 (costs/risk overlay: vol-targeting, drawdown brake, the
deferred no-trade band)** — not started, lower priority than A/B per
Fable's recommendation.

## What to actually ask the operator first, next session

1. Green-light (or discuss further) the FRED data dependency for carry (A)?
2. Proceed with the cheap S1/S2 decomposition (B) regardless, in parallel or
   first?
3. Ratify (C) — formally decline sign-flip rescues as a standing rule, or
   revisit?
4. Anything from Fable's "additive backtesting confidence" suggestions
   (regime-stratified P&L reporting, a pre-registered decay haircut on
   expectations, a pre-registered live/paper-monitoring kill-trigger
   contract) worth acting on now, independent of any new signal work?

Apply D27 from the start of the new session: consult `design-reviewer`
(pinned to Fable) at this phase boundary before committing to a direction on
A–D, not just once something is fully decided.
