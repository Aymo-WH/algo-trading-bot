# PRE-REGISTRATION v3 — S1 per-gate null assertions (PROPOSED, NOT ACTIVE)

**Status: PROPOSED 2026-07-09 — awaiting operator approval.** This file is
frozen at proposal time (guard blocks edits to existing specs). Activation is
recorded EXTERNALLY, exactly like D15: a new row in research/decisions.md
logging the operator's decision. Until that row exists, the current self-test
(v1 spec as amended by v2) remains the governing registration and
tests/referee/test_referee_calibration.py may not be modified.

Amends S1 ONLY. S1a, S1b, S2 (as amended by v2), S3, S4, every gate threshold,
and all referee code carry over verbatim and unchanged. validation/ stays
frozen — this change touches one test file.

## Why (M4, flagged by the Phase-0 fresh-context reviewer)

S1a asserts `all_gates_passed == False` for all 6 null variants, but each
variant runs against its own single-trial runs-store, where the PBO gate is
`False` by construction (PBO needs a populated trial matrix). S1a is therefore
**vacuously satisfiable**: it would still pass if every other gate were broken
always-True. Real size discrimination currently rests on S1b (net-Sharpe
chance-tail ≤ 1/6) alone.

## Evidence (measured, not intuited — D15 precedent)

Dry-run on the committed frozen referee, byte-identical S1 providers/seeds:
`research/m4_dryrun/m4_null_gate_analysis.py` →
`research/m4_dryrun/m4_null_gate_results.json` (2026-07-09). Determinism
cross-check: v3's net Sharpe +0.7226 matches the Phase-0 record exactly.
Per-gate outcomes across the 6 null variants:

| gate           | passes | note                                        |
|----------------|--------|---------------------------------------------|
| net_sharpe     | 1/6    | v3 (+0.7226) — the registered S1b allowance |
| survives_10bps | 1/6    | v3 (+0.5817)                                |
| dsr            | 0/6    | max null DSR 0.7943 vs gate 0.95            |
| pbo            | 0/6    | vacuous (single-trial store → None → False) |
| years_positive | 3/6    | not a size gate; no assertion               |
| max_drawdown   | 6/6    | diversified null books don't hit −20%; no assertion |
| canaries       | 6/6    | correct: canaries detect LEAKAGE, not null edge — a null-mined-but-honest strategy must pass them; asserting rejection here would be miscalibrated |

## New criteria (added to S1; exact, frozen)

- **S1c — non-vacuous rejection:** for EVERY null variant, rejection must not
  hinge on the vacuous PBO leg:
  `not all(v for k, v in gates.items() if k != "pbo")` for all 6 of 6.
- **S1d — DSR size:** the `dsr` gate passes for **0 of 6** null variants.
- **S1e — cost-stress size:** the `survives_10bps` gate passes for **at most
  1 of 6** (same deterministic chance-tail allowance as S1b).

Explicit NON-assertions (pre-registered so a future session doesn't "tighten"
them in error): no null assertions on `canaries`, `max_drawdown`, or
`years_positive` — per the table, those gates pass on honest null variants by
design and their rejection is not what they measure.

## Implementation & execution (upon approval only)

Three assert lines appended to
`tests/referee/test_referee_calibration.py::test_size_null_panels_pass_nothing`
(no other file, no threshold, no referee code). Then:
`/workspace/venv/bin/python -m pytest tests/referee/test_referee_calibration.py -q`
must be green (S1 re-run with the new assertions; S2 unchanged and re-verified
by the same command), expected deterministic per the dry-run. If any new
assertion fails against these measured values, the discrepancy goes to the
operator — no self-amendment (v2 §"no further self-amendment" carries over).
