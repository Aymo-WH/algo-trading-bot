---
name: preregister
description: Freeze a hypothesis BEFORE running a confirmatory experiment (mission §3.3c, §8). Use whenever a hypothesis graduates from exploration to a confirmatory test.
---

# /preregister — freeze a hypothesis before testing it

Pre-registration kills HARKing: success is defined before the number is seen.

Steps (all BEFORE running the experiment):

1. Pick the next spec id: `EXP-NNN` (next integer after existing `specs/EXP-*.md`).
2. Write `specs/EXP-NNN-<slug>.md` containing exactly:
   - **Hypothesis** — one falsifiable sentence.
   - **Exact metric** — formula + code path that computes it.
   - **Pass threshold** — pre-committed number(s); no ranges you can reinterpret.
   - **Falsification criterion** — what result kills the hypothesis.
   - **Planned trial count** — how many configs this test will consume from the
     ≤250 budget (ablations count).
   - **Data window** — train/validation only; state the exact dates.
   - **Seed(s)** and config file(s).
3. Append one JSON line to `research/experiments.jsonl`:
   `{"id": "EXP-NNN", "ts": "<iso>", "phase": "preregistered", "hypothesis": "...",
     "metric": "...", "threshold": ..., "planned_trials": N, "commit": "<git rev-parse HEAD>"}`
4. Commit the spec file BEFORE running: `git add specs/EXP-NNN* research/experiments.jsonl && git commit`.
5. Only then run the experiment. Append the real result to the ledger afterwards
   (`"phase": "result"` row citing script + config + output file). The spec file is
   now frozen — the guard hook blocks edits to it.

Never: adjust a threshold after seeing results, re-run with a "better" metric under
the same id, or leave a failed result out of the ledger. A failed pre-registered test
is a valid, valuable outcome — log it and move on.
