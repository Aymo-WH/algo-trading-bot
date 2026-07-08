# PRE-REGISTRATION v2 — Phase 0.6 Referee Self-Test (AMENDS S2 of REFEREE-SELFTEST-2026-07-08.md)

Frozen: 2026-07-08, before the amended suite's first execution.
**Operator approval:** logged this session — operator selected "Fix referee +
amend test" from the presented options after reviewing the S2 failure analysis
(research/journal.md 2026-07-08 entry). The original spec remains frozen as the
historical record; this file changes ONLY what is listed below. S1, S3, S4 and
every gate threshold carry over verbatim and unchanged.

## Why S2 is amended (root cause, evidenced in the journal)
The original S2 noise grid (multipliers 0→12) produced a trial-Sharpe spread of
sd = 1.26 ann (−0.842..+2.386): mostly-noise variants churn weekly and cost
drag stretches the spread. Under Bailey-LdP, that trial history implies
SR0 ≈ 1.9–2.1 ann; DSR > 0.95 then requires ≈ +0.63 ann Sharpe of margin on
1750 obs — statistically unsatisfiable for ANY faithful DSR implementation.
The referee refused correctly; the registered test asked an incoherent question.

## Amendment 1 — referee fix (completes the already-registered design)
`run_battery` and `final_eval` compute DSR with the **effective trial count**:
ρ̄ = mean pairwise correlation of the persisted trial return matrix
(`_trial_returns`), passed to `dsr_from_ledger` → `effective_trials`
(N̂ = ρ̄ + (1−ρ̄)·M, BLdP). This was prescribed by Phase 0.3, by
`research/validation_methodology.md`, and by the /run-validation skill, but was
never wired in. Conservative default ρ̄ = 0 when < 2 trial columns exist.
**The DSR > 0.95 gate does not change.**

## Amendment 2 — S2 grid (power analysis, not intuition)
Noise multipliers = **[4, 3, 2.5, 2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0]**
(clean LAST; variant seeds 1000+i by position; panel/ic/assets/days unchanged:
seed 42, IC 0.10, 1750 d × 40 assets). Grid chosen by a dry-run power analysis
executed BEFORE this registration (scratchpad/power_analysis_v2_grid.py, using
only committed referee math on the identical panel/seeds):
- cap4 grid → trial SR sd 0.930 ann, ρ̄ 0.385, N_eff 7.8, SR0 1.341 ann,
  **DSR(clean) = 0.9968**; CSCV PBO on the same matrix = **0.0** (12,870 combos).
- cap3 → DSR 0.9993; cap2 → DSR 0.9999. cap4 selected as the widest genuine
  quality gradient that is statistically coherent — the strongest PBO
  discrimination test that does not ask the impossible of DSR.

## Criteria (S2 restated; unchanged in substance)
- **S2a:** the clean variant's battery result has `all_gates_passed == True`
  (all gates incl. DSR > 0.95 at the amended, effective-N-corrected computation).
- **S2b:** PBO over the 12-column matrix ≤ 0.2.
- S1/S3/S4 as originally registered; the full suite
  (`/workspace/venv/bin/python -m pytest tests/ -q`) must be green before
  `validation/.frozen` is created. If any criterion fails again, work halts and
  the failure goes to the operator — no further self-amendment.
