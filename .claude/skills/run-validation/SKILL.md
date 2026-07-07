---
name: run-validation
description: Run the full validation referee battery on a candidate strategy (OOS walk-forward → CPCV → PBO/CSCV → DSR → canaries) and produce an evidence bundle. Use before claiming ANY result "validated".
---

# /run-validation — the referee battery

A result is "validated" only if EVERY step below passes. A single metric is gameable;
the battery is not (mission §8).

1. **Preconditions:** working tree clean (commit first); the candidate has a
   pre-registered spec (`specs/EXP-*.md`); referee tests green:
   `/workspace/venv/bin/python -m pytest tests/ -q`.
2. **Run the battery** (train/validation data only — the engine cannot see the
   lockbox):
   `/workspace/venv/bin/python -m validation.run_battery --config <candidate-config> --out research/runs/<ts>/`
   This computes: walk-forward net performance (5/10/20 bps), CPCV distribution,
   PBO via CSCV (S=16) over the full trial ledger, DSR with effective trial count,
   seed/perturbation stability, and runs all five canaries.
3. **Gates** (from specs/DESIGN-v2-2026-07-07.md — do not reinterpret):
   net Sharpe ≥ 0.5 · MaxDD ≤ 20% · majority of years positive · PBO < 0.5
   (target ≤ 0.2) · DSR > 0.95 · survives 10 bps · all canaries pass.
4. **Ledger:** the battery auto-appends every trial to research/experiments.jsonl;
   verify the row count increased accordingly.
5. **Adversarial audit (mandatory for any PASS):** spawn the `leak-hunter` agent on
   the diff + spec + evidence bundle. Its verdict goes in the journal.
6. **Journal:** append hypothesis → commands → raw outputs → verdict to
   research/journal.md. Audit triggers: Sharpe > 1.2 → treat as defect report and
   investigate before any claim; > 1.5 → presumed bug (mission §3.3f).

Never mark anything validated with a tripped canary, an unlogged trial, or a missing
leak-hunter verdict.
