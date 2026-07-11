Resume Gordian v2. Read CLAUDE.md and FABLE_MISSION.md in full, then read
research/journal.md and research/decisions.md (through **D36**) to confirm
current state before doing anything. Repo: /workspace/algo-trading-bot,
branch `research/gordian-v2`, last commit **cf9ae6b**. Working tree has
UNCOMMITTED doc-only changes as of this pause (`research/decisions.md`,
`research/journal.md`, `research/lockbox_access.log` — the last is just an
auto-appended benign BLOCKED-attempt audit line from a quarantine-guard hit,
not a real access). No code changes, no data pulled, nothing to lose by
picking this up in a fresh session; do not discard these edits.

## Where this session stopped: EXP-004 is BLOCKED, operator decision pending

This session began implementing EXP-004 (Tier-2 carry) per the frozen
`specs/EXP-004-tier2-carry-ic-screen-v2.md`. **Before any FRED data was
pulled or any construction code written**, live verification of the exact
FRED series IDs (the spec's own required implementation-time step) turned up
a real, verified blocker:

- **Treasury leg is fine.** Nominal CMT (DGS3MO/6MO/1/2/3/5/7/10/20/30) and
  TIPS real CMT (DFII5/7/10/20/30) both have long, clean daily history.
  Fund effective durations looked up from current issuer fact sheets
  (WebSearch, dated sources): SHY ~1.9-2.0y, IEF 7.2y, TLT 15.20y
  (2026-07-08), AGG 5.78y (2026-03-31), BND ~5.7-5.8y, TIP 6.41y
  (2026-03-31). Frozen nearest-tenor rule maps: SHY->2, IEF->7, TLT->20,
  AGG->5, BND->5, TIP->7 (IEF/TIP tie, already predicted by D35's addendum).
- **Credit leg is BLOCKED.** FRED's ICE BofA OAS series (IG: BAMLC0A0CM, HY:
  BAMLH0A0HYM2, EM: BAMLEMCBPIOAS, and every rating-bucket variant checked)
  all first-observe **2023-07-11** — confirmed via direct `fred.get_series()`
  pulls, not just search metadata. This is FRED/ICE's documented 2022
  licensing event: FRED dropped historical vintages of ICE-sourced index
  data and now carries only a rolling ~3-year trailing window (2023-07-11 is
  exactly "today minus 3y" — the gap never closes by waiting). ALFRED
  vintage archives were also checked (`get_series_as_of_date`,
  `get_series_all_releases`) and confirmed equally truncated — no
  point-in-time rescue exists anywhere on FRED.
- **Consequence:** 2023-07-11 onward sits entirely inside the frozen holdout
  (2022-01-01..2026-06-30, D8). The credit leg has ZERO usable dates on
  train/val, so the `min_names=10` confirmatory test — and even the
  `min_names=4` credit-only leg-attribution diagnostic — cannot score a
  single pre-holdout date. This is a data-infeasibility wall, not a weak
  result. Logged as **D36** (architect finding, zero trials consumed,
  ledger untouched).

**Design-reviewer consult (Fable, D27 standing rule) ran on this finding**
before it went to the operator: verdict ENDORSE-the-pause / REFINE-the-options.
Independently re-verified everything above, additionally ruled out a
Moody's-proxy substitute (BAA10Y only rescues the IG name; HYG/JNK/EMB stay
dead; dropping below 10 names silently degenerates `validation/canaries.py`'s
hardcoded `min_names=10` default — the exact problem D35 kept TIP in-sleeve
to avoid), and confirmed the TIP/breakeven tenor question is already fully
closed by D35 (not a new fork — dropped from the presentation).

**Presented to the operator as a genuine fork (AskUserQuestion, not decided
silently), with two live options:**
1. **Defer Tier-2 carry, return to Phase 3.** Log the blocker (already done,
   D36), abandon the credit leg for now, go build the Phase-3 M0 combiner on
   S1+S2 (already graduated, D23). Zero new spec, zero trial cost.
   Design-reviewer's recommendation.
2. **Narrow to a 6-name treasury-only v3 spec**, reverting D32's credit-leg
   broadening. Two defects must be pinned ex-ante if chosen: (a) a 6-name
   sleeve breaks the frozen canary suite's hardcoded `min_names=10` default
   — needs a resolution before this is runnable; (b) the tenor-mapping rule
   collapses 6 names to only 4 distinct carry values every date (SHY=2,
   AGG=BND=5, IEF=TIP=7, TLT=20). Design-reviewer's read: probably not worth
   one of the 250 trials given how coarse this is, but operator's call.

**The operator has not yet answered this question — hit a token/time limit
and asked to pause, resuming in ~3 hours (this same conversation may simply
continue; if instead this is a fresh session reading this file, re-ask the
same AskUserQuestion above before doing anything else).** Do NOT pick an
option unilaterally. Do NOT attempt a third rescue option (e.g. a different
data source, a shrunk-window test, silently patching `min_names`) without
routing it through the operator the same way — every rescue either reverts
an operator-approved decision (D32), substitutes a data family never
approved (D28/D32 named FRED specifically), or silently breaks frozen
validation code.

## Notes for whichever model drives this session

- Two stray Agent calls with literal "placeholder" content appeared during
  the 2026-07-10/11 session — root-caused as a prior driver's own erroneous
  pattern for yielding a turn while waiting on a background task, not an
  external anomaly. If waiting on a background task, just end the turn with
  plain text — no tool call needed.
- Background subagents can fail mid-run on session/API limits — retry
  cleanly, it isn't a signal anything is wrong with the work itself.
- `specs/` files freeze the instant they're written (quarantine guard blocks
  further Edit/Write regardless of review status) — a v3 spec (if option 2
  is chosen) must be a new versioned file, matching the
  `REFEREE-SELFTEST-v2`/`EXP-004-...-v2` precedents.
- The quarantine guard blocks ANY tool call whose path/command blob mentions
  `data/lockbox` or `OPERATOR_TOKEN` (except a Bash command literally
  invoking `validation/final_eval.py`) — this is real enforcement, not a
  false positive, confirmed again this session (a plain `ls -la
  data/lockbox/` was correctly blocked). Don't retry or work around it;
  scripts that only *import* `validation.lockbox` from within a Python file
  (not naming the path in the Bash command string itself) are unaffected,
  matching how `src/panel_factory.py` already does it.
- The guard's Bash write-heuristic can false-positive on read-only commands
  that merely contain a `>` character alongside a `specs/`/`validation/`
  path mention elsewhere in the same command string — rephrase rather than
  assume something is wrong.
