# PRE-REGISTRATION — Phase 0.6 Referee Self-Test (frozen before first run)

Frozen: 2026-07-08, before the calibration suite was first executed.
Scope: acceptance criteria for the full-pipeline referee self-test
(`tests/referee/test_referee_calibration.py`, `tests/referee/test_final_eval.py`,
`tests/fast/test_lockbox.py`). If any criterion fails, the REFEREE gets
root-caused and fixed (documented in research/journal.md); these criteria do
not move. All runs are seed-fixed and deterministic; synthetic-panel trials go
to throwaway ledgers and are excluded from the real DSR trial count (they never
inform strategy selection — decision D14).

## S1 — Size (the referee must find nothing in noise; mission §3.3d "synthetic NO-edge")
Six momentum-provider variants (lookbacks 20/60/120/250 d, fixed seeds, two
independent null panels ~1750 d × 40 assets) run through the FULL battery
(`validation.run_battery.run_battery`):
- **S1a:** `all_gates_passed == False` for **all 6 of 6** variants.
- **S1b:** at most **1 of 6** clears the raw `net_sharpe` gate alone
  (deterministic chance-tail allowance at SR se ≈ 0.38 ann).

## S2 — Power (the referee must recover a planted edge; mission §3.3d "synthetic KNOWN-edge")
Twelve variants of a planted-signal strategy (true rank IC ≈ 0.10 at 5-day
horizon, ~1750 d × 40 assets) with a genuine quality gradient (signal noise
multipliers from 0 to 12), degraded variants run first, the clean variant LAST
(so PBO computes over the full 12-column trial matrix):
- **S2a:** the clean variant's battery result has `all_gates_passed == True` —
  i.e. net Sharpe ≥ 0.5 at 5 bps, survives 10 bps, MaxDD ≥ −20%, majority of
  years positive, PBO < 0.5, DSR > 0.95, all canaries pass.
- **S2b:** PBO over the 12-column matrix ≤ 0.2 (the quality gradient makes the
  IS winner a genuine OOS winner).

## S3 — Trial accounting (mission §3.4 "nothing runs uncounted")
- Ledger row count (throwaway ledger) increases by exactly 1 per battery
  invocation: **18 invocations → 18 rows**.

## S4 — Lockbox & final-eval mechanics (mission §3.3h)
- **S4a:** encrypt→decrypt round-trip is byte-identical; a wrong/garbled token
  raises `PermissionError`; no plaintext holdout bytes on disk.
- **S4b:** rebuilding an existing lockbox (or overwriting an existing token)
  raises `FileExistsError`.
- **S4c:** `final_eval` refuses (RuntimeError) when: a prior result bundle
  exists; the access log records a prior RUN; the ledger is empty; the
  train/val panel overlaps or crosses the holdout boundary; the decrypted
  slice does not start at the expected holdout boundary.
- **S4d:** a valid one-shot run on synthetic data writes the result bundle +
  audit RUN line + ledger `phase="final"` row, and a second invocation then
  refuses.

## Execution
`/workspace/venv/bin/python -m pytest tests/ -q` must be fully green
(existing 41 tests + the new self-test) before `validation/.frozen` is created.
