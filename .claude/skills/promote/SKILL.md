---
name: promote
description: OPERATOR-ONLY one-shot final holdout evaluation (mission §3.3h/§3.3i, Phase 6). The single sanctioned path that unlocks the quarantine. Claude must never invoke this on its own initiative.
disable-model-invocation: true
---

# /promote — the one-shot final evaluation (operator-triggered only)

Preconditions (verify ALL, refuse otherwise):
1. The candidate passed the FULL /run-validation battery, with the leak-hunter audit
   verdict CLEAN, all documented in research/journal.md.
2. The mandatory human review (mission §3.3i) happened: the operator has read the
   evidence bundle and explicitly typed the /promote command themselves this session.
3. `research/experiments.jsonl` is complete (every trial logged) — DSR at final eval
   uses this count.
4. No prior final-eval invocation exists in research/lockbox_access.log — this runs
   ONCE, ever.

Procedure:
1. Ask the operator to retrieve the token themselves (it was written at lockbox build
   time to /workspace/OPERATOR_TOKEN.txt — Claude is hook-blocked from that path) and
   to run, themselves, via the `!` prefix:
   `! /workspace/venv/bin/python validation/final_eval.py --provider <module:function> --config <candidate-config.json> --prices <train_val_close_panel.csv> --operator-token <TOKEN>`
2. The script decrypts the holdout slice, evaluates the frozen candidate exactly once,
   writes the full evidence bundle to research/final_eval/, and logs the invocation.
3. **Immediately after the run, the operator deletes /workspace/OPERATOR_TOKEN.txt
   themselves.** The token is the only decryption key; destroying it makes any replay
   of the evaluation cryptographically impossible (the token also entered this
   session's transcript via argv — deletion closes that path too).
4. Report the result honestly — pass or fail. **There is no second evaluation.** If it
   fails, that is the mission's honest answer; write the close-out, do not "fix and
   retry" on the holdout.
