Resume Gordian v2. Read CLAUDE.md and FABLE_MISSION.md in full, then read
research/journal.md and research/decisions.md (through **D35**) to confirm
current state before doing anything. Repo: /workspace/algo-trading-bot,
branch `research/gordian-v2`, last commit **4c4d489** (working tree was
clean, everything pushed, as of 2026-07-11).

## What happened last session (summary — full detail is in journal.md)

1. Applied D27 (standing design-reviewer consult) at the start of the
   session on the open items (A-E) inherited from the prior resume prompt,
   before presenting anything to the operator.
2. Operator decided (AskUserQuestion): D28 (FRED carry green-lit,
   sequenced after a cheap no-new-data diagnostic first), D29 (decline
   sign-flip rescues, with a prior-exposure clause), D30 (add the canary
   suite as a diagnostic, non-blocking leg of signal graduation).
3. Pre-registered and ran **EXP-003** (S1/S2 static-vs-timing
   decomposition, `specs/EXP-003-s1-s2-timing-decomposition.md`):
   **S1_timing PASSES** graduation (t_nw=2.372, the weakest graduating
   t-stat in the project so far); **S2_timing FAILS** on the t-stat leg
   alone (1.750 vs 2.0) — a near-miss null, NOT a demonstrated absence of
   timing information (its confidence interval contains both zero and
   S1's value). Both timing components fail the diagnostic time_shift
   canary with the lagged t-stat EXCEEDING live — mechanistically
   root-caused by leak-hunter as benign "tilt re-injection" (subtracting a
   slowly-updated expanding mean re-acquires part of the static tilt under
   a shift), confirmed via bit-identical truncation attacks to need no
   future information.
4. Audited by reviewer (APPROVE) and leak-hunter (CLEAN across all 8
   refutation axes; leak-hunter's first attempt failed mid-run on a
   session/API limit — a clean retry succeeded. Treat any such background-
   agent failure as routine and just retry; it isn't a signal something is
   wrong with the work itself).
5. A design-reviewer consult on the EXP-003 result REFINED the architect's
   first-pass characterization before it reached the operator — most
   importantly correcting an overclaim ("S2 is almost entirely static
   tilt" was not actually supported by the data; the CI is too wide).
6. Operator decided three more items (AskUserQuestion): **D32** (broaden
   EXP-004's carry construction to include FRED credit-spread data for
   LQD/HYG/JNK/EMB, not just the treasury curve — presented as a genuine
   fork per the "present, don't pick silently" instruction, since a
   treasury-only construction would mostly re-test duration/term-premium
   rather than genuine cross-asset carry), **D33** (ratify a two-condition
   rule for interpreting time_shift trips on residual-class signals going
   forward — attributable to a benign mechanism only if an identity
   decomposition quantitatively accounts for the gap AND a truncation
   attack confirms no look-ahead; otherwise treated as a live trip. Has
   BLOCKING force at Phase 5 for any residual-class signal in the eventual
   book), **D34** (book identity: blend the static-tilt and
   timing-residual contributions and report them separately, defer the
   final trade-vs-strip choice until more evidence exists).
7. Drafted the **EXP-004** (Tier-2 carry) pre-registration. First draft
   (`specs/EXP-004-tier2-carry-ic-screen.md`, now ABANDONED) had two real
   errors a design-reviewer consult caught before any data was touched: a
   TS-z-score construction that would have quietly tested carry-*timing*
   instead of raw carry (per-asset z-scoring against an expanding history
   IS `src/decomposition.py`'s residual operation), and a holdout-period
   FRED storage rationale that inverted its own train/val-vs-holdout
   mirror logic and silently contradicted D18. Corrected in
   `specs/EXP-004-tier2-carry-ic-screen-v2.md` (**the active, frozen
   spec**): level-based excess-carry construction (Treasury CMT minus
   3-month bill; OAS directly for credit names), holdout FRED lockboxed
   per the existing D12/D21 mechanism (not clear-text), a pinned
   publication-lag buffer (t-1 business day, 5-day max staleness), pinned
   AGG/BND/TIP duration-tenor mapping. A second, narrower verification
   consult found one further correction (logged as **D35**, an addendum in
   decisions.md, not a v3 spec file): the "breakeven adjustment" for TIP
   is an exact arithmetic identity for the nominal Treasury yield at that
   tenor, not a genuinely separate data source — and TIP's duration is
   close enough to IEF's that they may permanently rank-tie on some/all
   dates (disclosed, power-reducing, not engineered around).

**No FRED data has been pulled. No real computation has run against the
EXP-004 spec.** Last session's work was entirely pre-registration/design;
this session is where implementation starts.

## This session's task: implement and run EXP-004

Per `specs/EXP-004-tier2-carry-ic-screen-v2.md` (+ the D35 addendum in
decisions.md) — this is a fully-specified, frozen construction; the task is
implementation, not further design:

1. **FRED API access is fully configured and tested — nothing to set up,
   just start building.** The operator provided a key; it's stored as
   `FRED_API_KEY` in `/workspace/activate.sh` (added 2026-07-11, outside
   the git repo, never committed — `source /workspace/activate.sh` loads
   it, same existing pod-restart step, nothing new). The key was verified
   live against the real API this session (fetched a real DGS10
   observation via both a raw REST call and the installed client
   library). `fredapi==0.5.2` is installed in the project venv and pinned
   in `requirements.lock.txt` (already flagged/logged this session — same
   dependency tier as `yfinance`; the underlying FRED *data* dependency
   itself was approved at D28/D32, this was just the client library).
   Nothing here needs re-verifying — go straight to building the actual
   EXP-004 data pull.
2. Confirm the exact FRED series IDs for: nominal Treasury CMT yields at
   the tenors the spec's duration-mapping rule selects for SHY/IEF/TLT/
   AGG/BND/TIP, the 3-month T-bill, and ICE BofA OAS indices for IG/HY/EM
   credit (LQD / HYG,JNK / EMB) — the spec deliberately left exact tickers
   for implementation-time verification, not guessed at design time.
   Record each fund's published effective duration (the number the
   tenor-mapping rule needs) and the confirmed series IDs in the
   implementation itself, not just in scratch analysis.
3. Build the data pull: full history (train/val + holdout, for §3.3b
   reproducibility), point-in-time as-of merge (t-1 business day lag,
   5-day max staleness per the spec), holdout-period rows (2022-01-01
   onward) encrypted into the lockbox via
   `validation.lockbox.build_lockbox` — NOT kept in a clear working file
   (this corrects a mistake in EXP-004 v1; don't repeat it).
4. Construct S5_carry exactly per the frozen spec: level-based excess
   carry per name, ranked cross-sectionally within the bond sleeve
   (`min_names=10`, explicitly passed — do not silently rely on a default
   parameter coinciding with the intended value, a mistake corrected last
   session).
5. Run the confirmatory IC/graduation test (1 planned trial,
   `EXP-004-S5_carry`) plus the descriptive/diagnostic legs the spec
   requires: leg attribution (`min_names=6` treasury / `min_names=4`
   credit, explicitly overridden), the static/timing decomposition
   (reusing `src/decomposition.py` unchanged), canaries on both raw and
   timing (pre-declared expected signature + the D33 two-condition
   adjudication test), and correlation vs S1/S2's own static/timing legs.
6. State a duration estimate before running anything non-trivial (§3.6) —
   the spec estimates 3-6 minutes for the full battery on this 10-name
   sleeve's shorter effective window, scaling down from EXP-003's 11m44s;
   flag if actual runtime diverges 2x+.
7. Writer≠verifier: fresh-context reviewer + leak-hunter audits before
   committing, same as every prior experiment. Apply D27 (standing
   design-reviewer consult) at this phase boundary and again once a real
   result exists — do not wait for it to feel like "presenting," per the
   standing rule.
8. Log the audit verdicts to `research/audit_log.jsonl` before committing
   (the `audit_gate` hook requires this mechanically).

## Notes for whichever model drives this session

- Two stray Agent calls with literal "placeholder" content appeared
  mid-way through last session — root-caused as the prior driver's own
  erroneous pattern for yielding a turn while waiting on a background
  task (not an external anomaly, despite an initial, incorrect suspicion
  of prompt injection). If you're waiting on a background task, just end
  your turn with plain text — no tool call needed.
- Background subagents can fail mid-run on session/API limits (happened
  once to leak-hunter last session) — retry cleanly, it isn't a signal
  anything is wrong with the audit itself.
- `specs/` files freeze the instant they're written (the quarantine guard
  blocks further Edit/Write regardless of review status, not just after
  an explicit freeze step) — if a drafted spec needs correction before
  its first real run, write a new versioned file (`-v2`, `-v3`, ...)
  rather than trying to edit the original, matching the
  `REFEREE-SELFTEST-v2` and `EXP-004-...-v2` precedents.
- The quarantine guard's Bash write-heuristic produced several false
  positives last session on read-only commands that merely CONTAIN a `>`
  character (e.g. in an email like `<noreply@...>`, or a mathematical
  comparison like `lagged_t>base_t`) combined with a `specs/`/
  `validation/` path mention elsewhere in the same command string. If a
  read-only Bash command gets blocked, rephrase to remove `>` or split
  the command rather than assuming something is actually wrong.
