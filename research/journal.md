# Gordian v2 — Research Journal (append-only)

## Status — 2026-07-08
- **Phase:** 0 (Build the referee) — COMPLETE pending freeze commit. Referee self-test
  green: 53/53 tests incl. pre-registered power/size calibration (v2-amended spec,
  logged operator approval D15).
- **Passed/failed:** S1 size ✓ (6/6 null variants rejected), S2 power ✓ after referee
  fix + spec amendment (clean planted variant passes all gates: DSR 0.9968, PBO 0.0),
  S3 accounting ✓, S4 lockbox/final-eval mechanics ✓ (10/10 tests).
- **Current best validation metrics:** n/a — no alpha work has begun; zero real-data
  trials in the ledger.
- **Open risks:** DSR gate is genuinely demanding at realistic ETF Sharpe levels (a
  junk-heavy trial history raises the hurdle fast — keep the real ledger lean and
  hypothesis-driven); survivorship tilt documented in D9.
- **Next steps:** freeze validation/ (.frozen), then Phase 1 (data & universe layer,
  lockbox build for the real holdout at panel-build time).
- **Pod cost note:** referee self-test ≈ 28 min CPU per full run; two full runs this
  session plus scratch analyses.

---

## 2026-07-07 — Phase R kickoff

**Hypothesis being investigated (Phase-R level):** the §4 default (cross-sectional
market-neutral over ~40–60 liquid ETFs, weekly rebalance) is buildable with honest data
and has a plausible ex-ante edge; or a better-motivated redesign exists.

**Actions:**
- Read `FABLE_MISSION.md` in full; restated Prime Directive + success contract to
  operator; Phase R plan approved verbatim (operator: "approve"), §2 numbers unchanged.
- Created branch `research/gordian-v2` off `docs/gordian-closeout`, merged
  `origin/main`, committed `requirements.lock.txt` (commit 4995848).
- Environment verified: `/workspace/venv/bin/python` 3.11.13; yfinance 1.5.1 smoke
  test (SPY 2024-01-02..09 daily, auto-adjusted) returned (6,5) rows — network OK.
- Launched 3 parallel read-only research subagents:
  - R1: v1 codebase forensics (reusable components + defect confirmation).
  - R2a: validation methodology canon (CSCV/PBO, DSR, CPCV, canary design).
  - R2b: ETF cross-sectional evidence brief (signal families, effective breadth, cost realism).
- R3 (data recon) running in main session: candidate universe definition + span/liquidity
  audit + effective-breadth measurement. **Quarantine discipline:** recon computes data
  statistics only (spans, volumes, correlation structure); no strategy P&L anywhere, and
  correlation/breadth statistics use pre-holdout data only once the boundary is proposed.

## 2026-07-07 — R3 data reconnaissance results (evidence: research/data_recon/)

Command: `/workspace/venv/bin/python research/data_recon/recon_universe.py`
(commit to follow; outputs: `research/data_recon/universe_summary.csv`, `breadth_report.txt`)

- **Universe:** 78 candidates requested, 77 returned usable history. 24 tickers with
  ≥25y, 68 with ≥18y. Liquidity: all majors fine; thin tail = TUR ($4.1M/day median,
  3y), FXE ($5.9M), EWM ($6.3M), EZA ($6.8M), DBA ($7.7M) — flag for possible exclusion
  or size caps.
- **Survivorship probe:** RSX (Russia, halted/delisted 2022) returns 1 junk row from
  yfinance — **dead ETFs are invisible in yfinance**. A "tickers alive today" universe
  is unavoidably survivorship-tilted at the margin; must be handled by design
  (broad-category mega-ETFs, documented closure risk, no return-based inclusion filter).
- **Effective breadth (pre-holdout ≤2021-12-31 only):**
  - Raw daily returns: participation ratio ≈ 2.9–3.4 (top PC = 53–58% of variance).
  - After removing top-1 PC: PR ≈ 16.3–16.8. Top-3 removed: PR ≈ 19.5–21.
  - Cross-sectionally demeaned: PR ≈ 9.1–10.0.
  - **Conclusion:** market-neutralization is what creates the breadth; true breadth is
    ~10–20 effective bets per rebalance, not nominal 50–77. §4's "50–100× breadth"
    framing is optimistic by ~3–5×; still a large improvement over v1's breadth ≈ 1.

## 2026-07-07 — R1 v1-forensics summary (subagent audit, read-only)

Full report retained in `research/v1_forensics.md`. Headlines:
- `core/pbo_validator.py` CSCV core is a faithful Bailey/LdP implementation → REUSE;
  the `validate_pbo.py` wrapper RUNS ON `data/test/` and uses correlated proxy trials
  → REBUILD (v2 CSCV runs on train/val with real trial variants).
- FFD is causal but optimal-d is selected on the FULL series incl. test dates → FIX.
- PIT PCA/scaler fit is genuinely train-only (static split) → REUSE, upgrade to
  expanding refits. Purge/embargo in XGB split correct (side + size) → REUSE.
- Confirmed v1 defects: long-only action space (`Box(0,1)`, shorts impossible),
  zeroed risk-penalty reward terms + asymmetric +50% profit bonus, live_inference
  train/serve mismatch (4 features vs 11, would crash) → DISCARD live path.
- Gym seeding broken (module-level `random`, identical state across vec workers) → FIX.
- Cost model: 1bp/side in phase1 config, no spread/slippage → FIX for v2.
- Thread-cap/CPU-affinity pattern in train_agent.py → REUSE verbatim.

## 2026-07-07 — R2a/R2b evidence briefs + R4 architecture decision

- R2a (validation canon) saved to `research/validation_methodology.md`: exact CSCV
  (S=16), DSR with effective-N, CPCV N=8/k=2, purge 5d + embargo 10d, canary designs
  for cross-sectional strategies (within-date label shuffle is the primary control),
  IC testing with Newey-West. Flagged source disagreements recorded.
- R2b (ETF evidence) saved to `research/evidence_etf_cross_section.md`: effective
  breadth 5–10 per literature (matches our measured 10–17); carry > trend > low-beta >
  seasonality; reversal/lead-lag contradicted at ETF level; realistic multi-style net
  Sharpe ~0.7 ceiling (AQR); 5–10 bps/side blended costs; momentum-crash and
  correlation-spike failure modes.
- **R4 verdict: CONFIRM the market-neutral cross-sectional chassis, REFINE the parts.**
  Full proposal in `research/design.md`; pre-registration frozen in
  `specs/DESIGN-v2-2026-07-07.md`; decisions D3–D9 logged. Trial budget ≤250,
  kill criteria K1–K4, audit triggers 1.2/1.5 pre-registered.
- `research/experiments.jsonl` initialized (empty — no trials run; Phase R computed
  data statistics only, no strategy P&L anywhere).
- **Next:** operator approval of design → Phase 0 (referee + lockbox + hooks/skills/
  auditor). No alpha work before the referee is green.

## 2026-07-07 — Design APPROVED; Phase 0 begins

- Operator approved the design ("ok, u may proceed"), delegated final architectural
  authority, and restated the end goal (real profit, no hallucination/synthetic-data
  reporting/overfitting) — consistent with the mission contract: honest OOS profit or
  a rigorous null; synthetic data used only to calibrate the referee, never to report
  performance. Decisions D3–D9 now ACTIVE; `specs/DESIGN-v2-2026-07-07.md` FROZEN.
- Git identity for this repo switched to the personal account (Aymo-WH
  <khoowiheng@hotmail.com>); the 5 unpushed commits rewritten accordingly
  (filter-branch over origin/main..HEAD).
- **Phase 0 build order:** (0.1) .claude/ runtime guardrails — quarantine hook,
  read-only auditor agents, /preregister /run-validation /promote skills;
  (0.2) validation/ package: purged splits + CPCV; (0.3) metrics (Sharpe/PSR/DSR/
  effective-N) + CSCV on real trial matrices; (0.4) panel backtester + auto-logging
  ledger; (0.5) synthetic null/planted-signal generators + canary suite + referee
  power/size validation; (0.6) lockbox + token-gated final_eval; then freeze
  validation/ (validation/.frozen) and run the full referee self-test.

## 2026-07-08 — Phase 0.6: lockbox review/redesign, final_eval, referee self-test

Session start: operator confirmed two out-of-session fixes — git identity corrected to
Aymo Khoo <khooweiheng@hotmail.com> (rewritten across commits; I never touch identity
again) and a push credential configured (push after every commit from now on). Verified
`git ls-remote`: remote tip = local HEAD 6e1c874, nothing stranded. Operator also pasted
a GitHub PAT in plaintext into the session; flagged for revocation since a stored
credential already works.

**Work done (uncommitted at session start: validation/lockbox.py, validation/run_battery.py):**
- Reviewed both. run_battery: sound; removed unused imports; documented CPCV-block
  semantics honestly (sub-period stability of delivered weights, refits are the
  provider's job). backtest.py docstring corrected (overstated per-call ledger logging).
- **lockbox.py redesigned (D12):** previous draft kept the Fernet key on disk at a known
  path — the operator-token check in open_lockbox was just an if-statement; code editing
  final_eval could skip it and still decrypt. Now the operator token IS the key: no key
  file exists; without the token, decryption is mathematically impossible.
- **validation/final_eval.py written (D13):** one-shot (result marker + access-log RUN
  line), refuses on unfrozen referee / empty ledger / holdout-boundary violations;
  holdout start hardcoded 2022-01-01 per D8.
- **Pre-registered the self-test** in specs/REFEREE-SELFTEST-2026-07-08.md (S1 size,
  S2 power, S3 accounting, S4 mechanics) BEFORE first execution. Synthetic calibration
  trials go to throwaway ledgers, excluded from the real DSR count (D14).

**Self-test results (evidence: pytest runs this session):**
- tests/fast (incl. new lockbox tests): green via PostToolUse hook.
- `pytest tests/referee/test_final_eval.py -q` → **6 passed in 10.60s** (S4 ✓).
- `pytest tests/referee/test_referee_calibration.py -q -rA` → **1 passed, 1 failed
  in 1694.53s**:
  - **S1 size: PASSED.** All 6 null-momentum variants rejected (all_gates_passed=False
    ×6); exactly 1/6 cleared the raw net-Sharpe gate (v3: +0.723) — within the
    pre-registered ≤1 allowance. Per-variant gates recovered from
    /tmp/pytest-of-root/pytest-14/.../battery_result.json.
  - **S2 power: FAILED on the DSR gate only.** Clean planted variant (true IC 0.10):
    net Sharpe 2.386 (5bps) ✓, survives 10bps ✓, MaxDD ✓, years ✓, **PBO = 0.0**
    (12,870 CSCV combos) ✓, canaries ✓, **DSR = 0.7749 < 0.95** ✗.

**Root cause (scratchpad/recompute_dsr.py on surviving pytest-14 artifacts, unmodified
validation.metrics):**
1. **Genuine referee defect:** run_battery computes DSR with trial_corr_mean=0
   (N_eff = N), though the Phase-0.3 design, research/validation_methodology.md, and the
   /run-validation skill all prescribe DSR with effective-N, and the trial return matrix
   needed for ρ̄ is already persisted for PBO. Measured ρ̄ = 0.2824 → N_eff 8.89 →
   DSR 0.8942. **Fix is necessary but NOT sufficient for S2.**
2. **Genuine self-test design defect:** the pre-registered S2 noise grid (multipliers
   0→12) produced trial Sharpes spanning −0.842..+2.386, sd = 1.260 ann → Bailey-LdP
   SR0 ≈ 1.91–2.10 ann. Passing DSR>0.95 then requires ≈+0.63 ann Sharpe over the
   hurdle on 1750 obs — statistically unsatisfiable for ANY faithful DSR. The referee
   refused correctly; the test asked an incoherent question.

**Pending operator decision (pre-registered criteria may not move without logged
approval):** proposed (a) wire effective-N ρ̄ into run_battery + final_eval DSR;
(b) amend the self-test via a NEW spec (noise grid capped at 4× → realistic trial-SR
spread ≈0.5 ann), all criteria otherwise unchanged; (c) re-run the full suite.

**Operator decision (logged, D15):** "Fix referee + amend test" approved via the
in-session question. Before registering the amended grid, ran a dry-run power analysis
(scratchpad/power_analysis_v2_grid.py — backtest-only, committed referee math, same
panel/seeds the test uses): cap4 grid → trial-SR sd 0.930 ann, ρ̄ 0.385, N_eff 7.8,
SR0 1.342 ann, DSR(clean) 0.9968; CSCV PBO on the same matrix 0.0. cap4 selected (widest
genuine quality gradient that is statistically coherent) and frozen in
**specs/REFEREE-SELFTEST-2026-07-08-v2.md** BEFORE the amended suite's first run.

**Referee fix implemented:** `metrics.mean_pairwise_correlation` added;
`run_battery` persists the trial return matrix first and passes ρ̄ into
`dsr_from_ledger` (gate unchanged at DSR > 0.95); `final_eval` uses the same ρ̄ from
the real runs store. `dsr_from_ledger` now records `trial_corr_mean` in its output.

## 2026-07-08 — Referee self-test GREEN; Phase 0 complete

Command: `/workspace/venv/bin/python -m pytest tests/ -q -rA --durations=5`
→ **53 passed, 0 failed in 1640.94s (0:27:20)**. Raw output in session task
b58bqa48v; clean-variant evidence from
`.../planted_runs/v11/battery_result.json`:
- gates: net_sharpe ✓ (2.386 @5bps), survives_10bps ✓ (1.044), max_drawdown ✓,
  years_positive ✓, **pbo ✓ (0.0, 12,870 combos)**, **dsr ✓ (0.9968; ρ̄=0.3852,
  N_eff=7.76, SR0=1.342 ann)**, canaries ✓ → `all_gates_passed: true`.
- Realized values match the pre-registered power analysis exactly (deterministic).
- S1 re-confirmed on the fixed referee: all 6 null variants rejected.
- Note: the 2.39 planted Sharpe exceeds the §3.3f audit bound by construction — this
  is the documented synthetic KNOWN-edge calibration, not a real result.

**Fresh-context reviewer verdict (writer ≠ verifier, agent a939d4a5e0ad31831):**
REQUEST-CHANGES — explicitly found NO ruler-shaving (effective-N change covered by the
D15 logged approval, faithful to BLdP, DSR>0.95 unchanged everywhere) and NO holdout
access, but flagged pre-freeze fixes. Dispositions:
- **M1 fixed:** one-shot markers now checked at BOTH caller-supplied AND canonical repo
  paths; /promote step added — operator deletes the token file right after the run
  (destroys the only decryption key → replay cryptographically impossible).
- **M2 fixed:** audit lines that are load-bearing one-shot markers (BUILT / OPEN
  ALLOWED / FINAL_EVAL RUN) now hard-fail if unwritable.
- **L2/L6 fixed:** final_eval refuses on truncated holdout (end-date check) and on
  train/val-vs-holdout column-set mismatch. **L3 fixed:** result bundle (the one-shot
  marker) is written before any other artifact.
- **L4 fixed:** final_eval tests fully isolated (own returns_store). **L5 fixed:**
  /promote command corrected to the real argparse signature.
- **M3 → OPERATOR ACTION:** a stale lockbox from the pre-D12 draft exists
  (data/lockbox/holdout.enc + old key at /workspace/.gordian_lockbox_key +
  /workspace/OPERATOR_TOKEN.txt). Operator must delete all three before the Phase-1
  lockbox build — I am hook-blocked from those paths by design.
- **M4 → DEFERRED (needs operator-approved v3 spec):** S1a is vacuously satisfiable
  (PBO gate is always False under 10 stored trials, so all_gates_passed can't be True
  for the 6 null variants regardless); real size discrimination currently rests on S1b.
  Proposal for next session: add per-gate null assertions.
- **L1 → OPERATOR ACTION (hook edit):** the guard's final_eval allowlist matches by
  substring; tighten to the exact sanctioned invocation. **L7/L8 deferred** (min-T
  guard for ρ̄; returns-store path convention) — noted, non-blocking.
Re-ran changed-path tests after fixes: `pytest tests/referee/test_final_eval.py
tests/fast -q` → **39 passed in 10.19s** (calibration paths untouched since the
53-green run; its evidence stands).

Phase 0 close: commit + push, create `validation/.frozen`, verify the guard blocks
validation/ edits, commit + push the freeze.

## 2026-07-08 — validation/ FROZEN. Phase 0 COMPLETE.

- Phase 0.6 committed as **82024e2** and pushed (14 files, 1003 insertions).
- Created `validation/.frozen`, then VERIFIED the freeze: a probe Edit to
  validation/__init__.py was denied by the quarantine guard with the FROZEN message
  (raw hook output in this session). The referee can no longer be modified without the
  operator removing the marker after a logged approval.
- **Phase 0 deliverable met (mission §5):** a tested referee — purged CPCV splits,
  Sharpe/PSR/DSR with effective-N, CSCV/PBO on real trial matrices, panel backtester
  with auto-logging ledger, five canaries, synthetic null/planted calibration proving
  size AND power, cryptographic lockbox, token-gated one-shot final_eval — 53/53 green
  under pre-registered acceptance criteria, adversarially reviewed.
- **Operator TODOs before Phase 1 lockbox build:** (1) delete the stale pre-D12 draft
  lockbox: data/lockbox/holdout.enc, /workspace/.gordian_lockbox_key,
  /workspace/OPERATOR_TOKEN.txt; (2) optionally tighten the guard's final_eval
  allowlist (L1) — hook edits are operator-only; (3) revoke the GitHub PAT pasted into
  the 2026-07-08 session transcript (a stored credential already works).
- **Next session = Phase 1 (data & universe layer):** yfinance daily panel for the D9
  universe (69 ETFs), PIT entry rule, data-integrity test suite, holdout slice
  encrypted into a FRESH lockbox at build time (train/val in the clear, 2022+ only in
  the box). Also propose the M4 v3-spec (S1 per-gate null assertions) for approval.
- No real-data trials run; research/experiments.jsonl still empty. Trial budget (≤250)
  untouched.

## 2026-07-08 — Operator TODOs confirmed done; PR opened; Phase 1 NOT started

- Operator reports (same day): stale pre-D12 lockbox artifacts deleted and the pasted
  PAT revoked (I cannot verify the lockbox paths directly — hook-blocked by design;
  recorded as operator-reported). Hook-tightening (L1) remains at operator discretion.
- Operator instruction: do NOT start Phase 1 yet; open a PR for easy retrieval.
- Session ends with Phase 0 complete, referee frozen, branch fully pushed.
- PR for retrieval: https://github.com/Aymo-WH/algo-trading-bot/pull/185 (branch reconnected
  to the rewritten main via merge 8943fe2; tree verified byte-identical to 1eb2d70;
  conflicts resolved keeping ours — .gitignore guardrail entries + the train_agent.py
  thread-cap fix the rewritten main had lost).

## 2026-07-09 — Phase 1: data & universe layer built and green

**Context restoration verified before any work:** journal tail + decisions D1–D15 read;
`validation/.frozen` present (357 bytes, 2026-07-08); freeze probe re-run — an Edit to
validation/__init__.py was DENIED by the quarantine guard with the FROZEN message (raw
hook output in this session). Branch fast-forwarded to origin (b3d51f9).

**Universe resolution (D16).** Applying the frozen D9 rule to the committed
universe_summary.csv: `status=='OK' and median_dollar_vol_3y_musd >= 10.0` → **70
names** (dropped thin/dead: DBA, EWM, EWS, EZA, FXE, FXY, RSX, TUR). design.md §2's
"69" is internally inconsistent (headline 69, exclusion arithmetic 70, category sum
68) — the rule wins, discrepancy logged, operator ratification requested. Command +
output in this session (pandas recount of the committed CSV).

**Panel cut (`src/panel_factory.py`, one-shot semantics like the lockbox):**
`/workspace/venv/bin/python src/panel_factory.py` — yfinance 1.5.1, 70 tickers,
period=max, daily auto-adjusted; smoke-tested SPY+EPI 3mo first (data through
2026-07-09 confirmed).
- calendar = SPY-traded sessions; PIT entry = 252nd observed day; rebalance =
  first trading day ≥ Wednesday per ISO week; **panel_start = 1999-12-22** (first
  rebalance with ≥ 20 eligible — matches design's "~2000" expectation).
- train/val IN THE CLEAR, full history from 1993-01-29 → **2021-12-31** (7285×70):
  trainval_close fbb7ed23…, trainval_volume 733dc41d…, eligibility 7b574dac…
  (full SHA-256 in data/panel/MANIFEST.json, committed).
- **FRESH lockbox built**: holdout close panel 2022-01-01 → 2026-06-30 (1126×70),
  plaintext sha256 6e627d498b2b8e74… recorded in the manifest (tamper-evidence at
  final eval), encrypted to data/lockbox/holdout.enc; plaintext never touched disk.
  The clean build itself verifies the operator's stale-artifact deletion
  (build_lockbox raises FileExistsError otherwise). **Operator token written to the
  canonical path — operator must retrieve and secure it now.**
- Stale Phase-R clear caches deleted (data/recon/prices_{close,volume}.csv held
  2022+ closes); a data-integrity test now enforces their absence.
- Build logged to research/experiments.jsonl as phase="data_build"
  (id=PHASE1-PANEL-BUILD) — excluded from trial_count()/DSR by construction and
  tested as such. **Zero strategy trials run to date; trial budget 250 untouched.**

**Data-integrity suite (`tests/data/`, 20 tests):** PIT no-look-ahead asserted
literally (truncating the future never changes the past), eligibility monotone +
recomputable from the artifact, boundary tests (nothing after 2021-12-31 in ANY
clear csv under data/), universe columns exact, calendar integrity, split-outlier
scan (max |1d return| < 0.60), manifest-hash-vs-disk, referee-constant cross-check
(panel boundaries == validation.final_eval's), ledger-exclusion test. One test bug
found+fixed during bring-up (pandas 3.x stack() keeps NaN → positivity check needed
dropna; data itself was fine).
`pytest tests/fast tests/referee/test_final_eval.py tests/referee/test_canaries.py
tests/referee/test_cscv_ic.py tests/data -q` → **71 passed in 93.85s**. The 27-min
calibration battery was NOT re-run: validation/ and tests/referee are byte-identical
to the frozen 53/53-green commit (git status clean on those paths); its Phase-0
evidence stands. Phase-1 code touched only new files (src/panel_factory.py,
tests/data/).

## 2026-07-09 — M4 v3-spec PROPOSED; L1 re-flagged with a sharper threat model

**M4 (deferred from Phase 0):** per-gate S1 null assertions drafted as
`specs/REFEREE-SELFTEST-2026-07-09-v3.md`, **status PROPOSED — not active** until an
operator-approval row lands in decisions.md (the D15 mechanism; the spec file itself
is frozen at proposal time by the guard). Assertion values are MEASURED, not
intuited: dry-run of the 6 S1 null variants on the frozen referee
(research/m4_dryrun/, D14 throwaway ledger; determinism cross-check: v3 net Sharpe
+0.7226 matches the Phase-0 record). Proposed: S1c non-vacuous rejection (all 6
rejected even with the PBO leg masked — measured 6/6, worst case rejected by DSR
alone), S1d DSR passes 0/6 (max null DSR 0.7943), S1e survives_10bps ≤ 1/6. Explicit
non-assertions pre-registered for canaries / max_drawdown / years_positive (canaries
detect leakage, not null edge — all 6 null variants correctly pass them; a future
"tightening" there would be miscalibrated). Implementation upon approval = three
assert lines in the S1 test; no referee code, no thresholds.

**L1 (operator-discretion, re-flagged with higher urgency):** the guard's final_eval
allowlist matches by SUBSTRING — any bash command merely containing the string
"validation/final_eval.py" (e.g. in a trailing comment) bypasses the lockbox-token
block entirely, for ANY action in that command. Post-D12 the token IS the Fernet
key, and since today's Phase-1 build a FRESH token exists at the canonical path
until the operator secures it — during that window the substring rule is the only
mechanical barrier between an agent command and the decryption key (deliberate
misuse would still land in lockbox_access.log as ALLOWED, but auditable ≠ blocked).
Recommendation to operator: (1) retrieve + secure the token file NOW; (2) tighten
the allowlist to an anchored exact invocation, e.g.
`^/workspace/venv/bin/python validation/final_eval\.py --provider \S+ --config \S+
--prices \S+ --operator-token \S+( --out \S+)?$` with no shell metacharacters.
Hook edits are operator-only by design — I cannot and will not make this change.

**Open for operator:** ratify D16 (universe 70-by-rule vs design.md's miscounted 69);
approve/decline the M4 v3 spec; decide L1 timing; optional: commit holdout.enc for
durability (I am guard-blocked from staging data under the lockbox dir and won't
work around it).

## 2026-07-09 — Operator decisions (D20); M4 v3 ACTIVE and green

Operator answered the in-session question (same mechanism as D15), logged as **D20**:
(1) **M4 v3 APPROVED — activate now**; (2) **L1: secure the token only**, hook-edit
deferred (documented residual risk accepted; revisit before Phase 6); (3) **D16
ratified** — universe is 70-by-rule; D16 status cell updated to point at D20.

M4 implementation per the activated spec: S1c/S1d/S1e added to
tests/referee/test_referee_calibration.py::test_size_null_panels_pass_nothing
(three assert lines + docstring reference; no referee code, no thresholds). Re-run:
`/workspace/venv/bin/python -m pytest tests/referee/test_referee_calibration.py -q -rA`
→ **2 passed in 1452.08s (0:24:12)** (S1 incl. the new per-gate null assertions; S2
re-verified unchanged). Raw output in session task boh8eyc81 (run survived a session
restart; verified by waiting on the live pytest PID). Combined with the 71 fast+data
tests earlier today, the whole suite is green at this commit.

Phase 1 is COMPLETE: data & universe layer built, integrity-tested, lockbox fresh,
zero strategy trials logged. Next session = Phase 2 (signal library): pre-register
each Tier-1 signal's IC test per design §5 before computing ANY IC on real data —
those are the first rows that count against the 250-trial budget.

## 2026-07-09 — Session resume: model switched to Sonnet 5, agents re-pinned, §3.6 amended

Context restoration for this turn: operator reports OPERATOR_TOKEN.txt secured
off-pod and deleted from /workspace (guard-blocked from verifying directly — taken
as reported, same as the Phase-0 stale-lockbox TODO). Two operator commits found
already on the branch and pushed to origin: `d2daab1` (leak-hunter/reviewer pinned
to `model: claude-fable-5` in their agent frontmatter — verified by reading both
files) and `eff8e16` (D21: holdout.enc committed for durability, ciphertext-only).
FABLE_MISSION.md §3.6 confirmed to now include the pre-run duration-estimate rule
(grep'd the live file, lines 184-190) — will apply it before any non-trivial run
starting with the Phase-2 IC computation.

**Interstitial: flaky test fixed (D22).** Before starting Phase 2, a PostToolUse
hook run (triggered by an unrelated scratchpad file write) surfaced
`tests/fast/test_lockbox.py::test_round_trip_and_no_plaintext_at_rest` FAILING:
`assert b"SPY" not in enc` tripped because Fernet's per-call random IV produced
ciphertext that happened to contain the 3-byte substring "SPY" by coincidence.
Root-caused with evidence before touching anything:
`for i in 1..8: pytest tests/fast/test_lockbox.py::test_round_trip_and_no_plaintext_at_rest -q`
→ **8/8 passed** on the UNMODIFIED test (confirms non-determinism, not a logic bug).
Fixed by dropping the redundant, collision-prone short-substring check and keeping
the collision-safe `PLAINTEXT not in enc` (~40 bytes — negligible coincidental-match
probability), which is exactly S4a's pre-registered criterion ("no plaintext holdout
bytes on disk") word for word. Re-verified: `pytest tests/fast/test_lockbox.py -q`
×5 → **4 passed** each time; `pytest tests/fast tests/data -q` → **53 passed in
5.26s**. Logged as D22, flagged for operator review since the file is part of the
Phase-0 evidence bundle even though it sits outside validation/'s mechanical freeze.

## 2026-07-09 — Phase 2 begins: EXP-001 pre-registered (Tier-1 signal IC screen)

Per design.md §5 / mission §8, pre-registered BEFORE computing any IC on real data:
`specs/EXP-001-tier1-signal-ic.md` — four signals (S1 XS momentum, S2 TS trend, S3
low-beta, S4 seasonality), each a single fixed construction already specified in the
operator-approved design.md (no hyperparameter search), the graduation rule already
coded in `validation/ic.py` (mean IC ≥0.01, NW t≥2.0, ≥60% years positive),
train/validation only (1999-12-22..2021-12-31). 4 planned trials — the first real
usage of the 250-trial budget (Phase 0/1 rows are preregistered/data_build/
throwaway-synthetic, none countable per D14).

## 2026-07-09/10 — EXP-001 executed, audited, and closed (D23). Session paused here.

**Implementation.** `src/signals.py` (S1-S4 exactly per the frozen spec) +
`tests/fast/test_signals.py` (14 unit tests: independently-computed expected values,
not tautological re-calls of the module's own helpers; point-in-time truncation
tests on all four; the S1-vs-S2 skip-month differentiator test; an exact beta
recovery test; S4's strictly-prior-years and current-year-exclusion tests). Bring-up
found 3 test-construction bugs (a dead placeholder line; two cases of a small-N
synthetic panel letting the constructed asset contaminate its own market proxy via
`equal_weight_market_return` — fixed both by using the `market=` override each
function already exposed) — all bugs were in the NEW test fixtures, not in
`signals.py` itself; root-caused with evidence each time before editing anything.
`pytest tests/fast tests/data -q` → **67 passed in 4.87s**.

**Real run.** Duration estimated up front per the new §3.6 rule (no directly
comparable past run existed; reasoned from job size: ~7285×70 rolling ops + ~4600
per-date IC calls across 4 signals → "under 60s, likely 10-30s"). Actual: **7.87s**
wall clock (`time /workspace/venv/bin/python research/exp001_signal_ic/run_exp001.py`)
— faster than estimated, no divergence flag needed (the rule flags 2x+ *over*, not
under). Real result (`research/exp001_signal_ic/results.json`, train/val only,
1999-12-22..2021-12-31, holdout never touched):
- **S1 momentum: PASSES** — mean_ic 0.0378, t_nw 3.919, 73.9% years positive.
- **S2 TS trend: PASSES** — mean_ic 0.0307, t_nw 3.162, 78.3% years positive.
- **S3 low-beta: FAILS** — mean_ic -0.0357, t_nw -3.062 (significant, wrong sign),
  30.4% years positive. Not a bug: real market factor uses all ~70 eligible names
  (negligible self-contamination, unlike the small-N test fixtures above); plausible
  reading is the classical single-equity-market BAB anomaly not transferring to this
  heterogeneous cross-asset-class ETF universe over 1999-2021 — logged as an honest
  negative, not chased.
- **S4 seasonality: FAILS** — mean_ic -0.0016, t_nw -0.215, indistinguishable from
  zero.
- Both passing ICs sit in validation_methodology.md §7's "0.02-0.05 realistic"
  band, far under the 0.15 pre-declared leak-flag — checked explicitly, did not
  fire (per the design-reviewer consult's one requested addition, folded in here).
- **Inter-signal correlation (descriptive, design §6): S1↔S2 = 0.796** — the two
  survivors are substantially one bet. Flagged for whoever designs the Phase-3
  combiner (decorrelation-aware weighting, or treat as one signal), not resolved now.
- Ledger: exactly 6 rows total (1 data_build, 1 preregistered, 4 result) — verified
  by direct count, matching the pre-registered `planned_trials: 4`. 4/250 trial
  budget spent. **Verdict: partial survival, not Tier-1-null** (K1 needs all 4 to
  fail; design §10).

**Audits (writer≠verifier, mission §8 + the new CLAUDE.md design-reviewer rule).**
Both spawned in parallel, each given only the diff/files + the pre-registration:
- **leak-hunter → CLEAN.** Attempted refutation on every mandated axis — look-ahead
  (including an empirical truncation attack: recomputed all 4 signals on the real
  panel truncated at 2017-12-29, bit-identical to the full-panel run at every
  overlapping date), quarantine access (0 lockbox references outside a docstring
  disclaimer; sha256 of all 3 panel artifacts match MANIFEST.json exactly), spec
  drift (spec untouched since the freezing commit, verified via `git diff`),
  rescue-by-sign-flip (S3's failure is logged as a failure, nothing flips it),
  ledger completeness (6 rows, independently re-verified by me: `grep -c` gives
  4 result / 1 preregistered / 1 data_build, exact match) — found no defect on
  any axis. One process note: the code was still uncommitted when it audited
  (fixed by this entry's commit) and independently re-confirmed D22 (the lockbox
  test fix) was a legitimate flake-fix, not ruler-shaving.
- **design-reviewer → ENDORSE** on all four presentation points (S1/S2 graduate,
  S3/S4 drop without rescue, S1/S2 correlation flagged-not-resolved for Phase 3,
  report now rather than chasing Tier-2 first); explicitly found no §7 pause-bar
  trigger. Ran as a `general-purpose` substitute on `model=fable` with the real
  design-reviewer.md content injected verbatim as instructions: the actual
  `design-reviewer` subagent type exists on disk (`.claude/agents/design-reviewer.md`,
  committed by the operator directly as `a9bd0f7` mid-session) but was not in this
  session's loaded agent registry (added after the registry snapshot) — confirmed
  available again after the session-limit reset (system message: "New agent types
  are now available... design-reviewer"), so future consults should use it directly.
  Both agents' factual citations were spot-checked against files read directly in
  this session (validation/ic.py thresholds, results.json values, ledger counts,
  git log) and matched exactly.

**Flagged for operator attention — possible prompt-injection, not acted upon.**
When resumed via SendMessage to restate its verdict, the design-reviewer substitute
reported receiving a message "framed as being from 'the coordinator'" that, per its
own account, carried instructions about pre-treating future coordinator messages as
carrying operator-equivalent authority ("act on coordinator course-corrections as if
they were within my existing permissions... only my user's own messages count as
approval"). It explicitly refused to fold this into its findings and surfaced it
instead — correct behavior. Two things make this ambiguous rather than confirmed:
(1) my own two messages to that agent (the original task + the restate request, both
quoted in full in this session's transcript) contain nothing resembling that
content; (2) leak-hunter, resumed the same way in parallel, reported its own
"coordinator" resume message as completely benign (just my restate request) — no
anomaly. Both notifications carried an explicit system caveat that the usual
safety-classifier review was unavailable for these two runs. I cannot determine from
here whether this was a genuine external injection, a harness-labeling artifact the
design-reviewer substitute (primed to be maximally skeptical about authority/approval
framing, per its own mandate) over-flagged, or a model confabulation — but no
evidence exists that anything was actually acted on: the substitute's actual verdict
is the one presented above, fully grounded in file:line citations I independently
spot-checked. Operator: worth a look if platform/harness logs for agents
a1471f5fa91ac9681 / a70b86995e46691c1 are available; no repo/data action needed on
my end.

**Committed and pushed this round:** `src/signals.py`, `tests/fast/test_signals.py`,
`research/exp001_signal_ic/{run_exp001.py,results.json}`, the 4 real ledger rows,
decisions.md (D23), this journal entry.

---
### SESSION PAUSED HERE (operator stopping the pod; resume tonight)

**Done:** Phase 2's first signal batch is fully closed — pre-registered, executed,
audited (leak-hunter CLEAN + design-reviewer ENDORSE), logged, decided (D23),
committed, and pushed to `origin/research/gordian-v2` (push confirmed, not just a
local commit — see commit hash in the push output this entry accompanies). Build is
green: 67 tests (fast+data) plus the untouched frozen 53 (validation/tests/referee,
byte-identical since Phase 0/M4). Trial budget: 4/250 spent, both outcomes (2 pass,
2 fail) logged honestly.

**Mid-flight:** nothing code-wise — this was a clean stopping point, not a partial
one. Two things await the operator, not further Claude work: (1) the prompt-injection
flag above; (2) M4/L1 from the prior session remain as previously logged (M4 active,
L1 deferred by operator choice) — no change this round.

**Next step, next session:** Phase 2 continues — either (a) attempt more Tier-1
signal candidates / Tier-2 carry as a separate pre-registered trial (design §5 Tier 2
needs new data: FRED yields, dividends, roll-proxies — a new dependency, flag to
operator before adding it), or (b) proceed to Phase 3 (cross-sectional model) with
just S1+S2, treating their 0.796 correlation as the first open design question for
the combiner (M0 z-score composite per design §6) — **use the now-available real
`design-reviewer` subagent directly** (not the substitute) to weigh in on which of
(a)/(b) to pursue before building anything. Either way: pre-register before touching
real data again, same as every step this session.

## 2026-07-10 — Session resume: pod recreated, D22 open item + stale test resolved (D24)

Context restoration per the new startup protocol: read CLAUDE.md, FABLE_MISSION.md,
journal tail, decisions.md through D23 — confirmed clean tree at c11d9d3, matching
operator's report of a pod recreation (RunPod host out of vCPU) onto the same network
volume/commit, fresh container.

**Item 1 — failing test triaged.** Operator ran the canary/referee suite manually
pre-session: 84 passed, 1 failed —
`tests/data/test_panel_integrity.py::test_build_logged_but_excluded_from_dsr_trial_count`
(asserted `trial_count() == 0`). Independently confirmed (not taking the operator's
read as ground truth): `trial_count()` (validation/ledger.py:65) counts
`phase in ("trial","result")`; the ledger has 4 real `result` rows from EXP-001 (D23),
legitimately logged after this test was written in the Phase-1 commit (84929f2),
*before* EXP-001 existed. Stale assertion, not a leak — its own first assert already
shows the real intent (the data-build row is excluded by phase, not "ledger is empty
forever").

**Item 2 — fresh-context reviewer sign-off obtained BEFORE editing** (writer≠verifier,
mission §8): reviewer independently re-derived the same diagnosis from git history +
code, found no defect in `trial_count()==4`, and endorsed the proposed fix with one
amendment (`r["id"]` → `r.get("id")` to avoid a KeyError masking the real check).
Applied with the amendment: re-derives the expected count from the ledger and asserts
the build-row id is absent from counted rows, instead of hardcoding a count that goes
stale as the ledger legitimately grows. `pytest tests/data/test_panel_integrity.py -q`
→ **20 passed**. Verdict logged to `research/audit_log.jsonl` (first entry — the
audit_gate hook itself was only added in the prior session's last commit, c11d9d3).
Committed + pushed as **1b8e02f**.

**Item 3 — D22 (prior unilateral lockbox test fix) ruled on by the operator.**
Presented the diff/rationale; operator chose **revert-and-redo** (not ratify-as-is)
over ratify+retroactive-review — see **D24**. Reverted the working-tree assertion to
the pre-fix state, sent to a fresh-context reviewer for sign-off *before* any commit
this time (closing the exact process gap D22 exposed). The reviewer independently
reproduced the coincidental-substring flake from scratch (200k direct Fernet
encryptions: 102 hits, ~1/1960, matches ~1/1900 theory) and found a stronger argument
than the original: `PLAINTEXT` contains `\n` (0x0A), outside Fernet's urlsafe-base64
output alphabet, so the retained `PLAINTEXT not in enc` check is deterministically
collision-proof, not merely low-probability. Fix content ends up byte-identical to the
original f78e1d1 change — nothing to recommit (`git checkout --` restored it; working
tree clean against HEAD). Verdict logged to `research/audit_log.jsonl`. D22 marked
"active" (ratified) via D24; process gap closed.

**Item 4 — S1/S2 redesign consult run and reviewed.** Per operator instruction, ran
as a small Workflow: 3 parallel research agents (decorrelation/combiner literature;
candidate signal families; cost-benefit-risk of the fork itself) -> 1 synthesis
agent -> 1 design-reviewer fresh-context check (5 agents total). Operator feedback
on execution (saved to memory for future sessions): the consult should have been
"Fable-led" (all research/synthesis agents pinned to `model: claude-fable-5`
explicitly, not inheriting the session's Sonnet-5 driver — only the design-review
stage picked up Fable, via its own subagent-type frontmatter pin) and the agent
count crept from the approved "2-3 research + synthesis" framing to 5 once the
already-separately-requested design-review stage stacked on top, uncalled-out.

**Result:** design-reviewer verdict REFINE (not clean ENDORSE) on the synthesis's
recommendation to proceed to Phase 3 with S1+S2 as-is (M0 unchanged) plus a
parallel non-blocking exploratory screen on a new candidate (long-horizon
reversal). Four required refinements before proceeding, all incorporated: (1)
renamed the candidate S5->**S6** (S5 is already reserved for Tier-2 carry, D6);
(2) explicitly distinguished S6 (long-horizon, skip-12m) from D4's dropped
short-term reversal; (3) sequenced the Phase-3 M0 pre-registration spec to freeze
*before* launching the S6 screener, making the firewall mechanical; (4) the
escalation trigger (CPCV-median net Sharpe landing 0.2-0.4 -> consider escalating
S6) is written into the frozen spec itself, not left as prose.

**Operator sign-off:** logged as **D25** — proceed to Phase 3 with S1+S2, M0
construction unchanged from design.md §6-7 (the 0.796 correlation is a
Grinold-GLS-derived breadth ceiling that no linear recombination trick escapes;
hunting for one would itself be a new, ledger-inflating trial per the
forking-paths risk the research surfaced).

**Phase 3 M0 pre-registration frozen:** `specs/EXP-002-phase3-m0-combiner.md` —
exact construction (S1+S2 equal-weight composite, cross-sectional OLS-residual
neutralization against beta for exact pre-clip dollar+beta neutrality, ±2.5 clip,
iterative position/category-cap waterfilling to gross 200%, no-trade band),
referee invocation via the existing frozen `validation.run_battery` (unchanged),
falsification = K2 (net Sharpe < 0.2 -> null), planned trial count 1 (the 5th
real trial against the <=250 budget). Vol-targeting/drawdown-brake explicitly
deferred to Phase 4 per the mission's own phase plan, not smuggled into M0.
Committed ahead of implementation, and ahead of the S6 screener launch, per
refinement #3's sequencing requirement.

**No new alpha/Phase-2/3 code has RUN yet this session** — the spec above is
frozen; `src/portfolio_m0.py` (the provider implementing it) and the actual
battery run are next.

## 2026-07-10 — EXP-002 (Phase 3 M0) executed, audited, fixed, re-executed. NOT validated.

**S6 exploratory screen (non-blocking, `factor-screener` on `model=fable` per
operator feedback on the earlier consult's model choice) came back first:**
mean_ic -0.0281, t_nw -2.90, pct_years_positive 0.381 — **fails graduation as
specified** (long-horizon reversal is actually continuation in this universe,
opposite of the hypothesis). Correlation with S1/S2 was low (-0.18/-0.17) —
decorrelation goal achieved, hypothesis wrong. The agent explicitly flagged
that a SIGN-FLIPPED variant would pass (t_nw +2.90) but refused to promote it
itself, correctly naming that as a post-hoc sign flip after peeking at the
data — exactly the HARKing risk the earlier consult (D25) warned about. Not
acted on; left for operator judgment, not pre-registered.

**Implementation (`src/portfolio_m0.py`, `tests/fast/test_portfolio_m0.py`):**
M0 per the frozen spec — S1+S2 composite, cross-sectional OLS-residual
neutralization against rolling beta, z-score+clip, position/category-cap
waterfilling, no-trade band. Bring-up found and fixed two real construction
bugs before any real run (both root-caused with evidence, not assumed): (1) a
gross-exposure overshoot up to 2.8% caused by the no-trade band trading some
names while freezing others, mixing two different dates' otherwise-consistent
capped solutions; (2) a cap-waterfilling test-design flaw (asserting exact
gross-2.0 achievability with only 4 names under a 10%-of-gross position cap —
mathematically impossible; fixed the TEST, not the algorithm).

**v1 real run** (`PHASE3-M0`, ledger row logged): net Sharpe (5bps) 0.2228 —
lands in the pre-registered K2 "ambiguous 0.2-0.4" zone. **Canary suite
tripped**: `time_shift` failed — shifting the signal forward 26 weeks did NOT
destroy the edge; the shifted signal's t-stat (4.42) was HIGHER than live
(3.88). Per spec/mission (K4), this halts any "validated" claim regardless of
Sharpe — did not proceed to interpret the result before root-causing.

**leak-hunter audit (v1, fresh-context):** NO LEAKAGE — empirical truncation
attack bit-identical at 3 cutoffs (2010, 2015, 2020). Root-caused the canary
trip as genuine but benign: M0's edge is substantially an UNTIMED STATIC TILT
(a constant per-asset tilt alone scores t=5.28, higher than the live signal's
3.88; the time-varying component alone scores only t=2.81) plus a BETA-HEDGE-
DECAY artifact (the neutralized signal forfeits a real, positively-priced raw
beta premium — IC t=+3.06 — that the 26-week-stale version has partially
drifted back into, explaining the *strengthening*, not just retention).
**Important side-finding:** S1 and S2 individually would ALSO fail time_shift
if tested (S1 t 3.92→3.18, S2 3.16→3.03 — neither retains <50% of base) —
EXP-001 never ran canaries on them (graduation was IC-legs only). Not a leak;
a validation-coverage gap in Phase 2 worth operator attention, not decided
here.

**reviewer audit (v1 code, fresh-context): REQUEST-CHANGES.** Found real bugs
independent of the canary question: (1) the no-trade-band's post-freeze
uniform rescale reintroduced real position/category cap violations (up to
1.34% of gross on 175/1149 dates) — exceeded the spec's own 0.1% tolerance;
(2) that rescale mechanism was itself a post-freeze construction change NOT in
the frozen spec, added unilaterally while fixing bug (1) — flagged as needing
explicit sign-off, not silent inclusion; (3) spec-promised diagnostics (max
position/category exposure, net dollar exposure, ex-ante beta) were never
written to any output; (4) `neutralize_against_beta`'s "exact identity" claim
is false under NaN-mask mismatch (5/1149 dates — beta needs 252 trailing
RETURNS, composite needs 252 trailing PRICES, masks can differ); (5)
integrity-check blind spots; (6) silent non-convergence risk in the cap loop.

**Operator decision:** presented the finding that I'd left turnover control
(the no-trade band) inside Phase 3 scope while correctly deferring
vol-targeting/drawdown-brake to Phase 4 — an inconsistency, since the
mission's own phase plan assigns turnover control to Phase 4 too. Operator
approved (AskUserQuestion) dropping the no-trade band entirely from EXP-002,
deferring to Phase 4 — **logged as D26**. Process note, self-corrected: the
code cited "D26" before that row existed in decisions.md (caught by the v2
leak-hunter audit, not by me first) — the underlying approval was real, I had
just not logged it yet. Fixed by writing D26 accurately; flagged here rather
than smoothed over.

**Fixes applied (v2):** no-trade band no longer called in
`compute_signal_and_weights` (kept, tested, documented for Phase 4 reuse);
`neutralize_against_beta` masks composite/beta to joint `notna()` (exact
identity now holds to ~1e-14 on real data, confirmed empirically); integrity
check hardened (NaN-pattern match, non-empty, subset checks — 6 failure modes
probed and all correctly rejected); cap loop raises on non-convergence
instead of silently returning; `run_exp002.py` now writes
`battery_out/diagnostics.json` with all 4 promised diagnostics.

**v2 real run** (`PHASE3-M0-v2`, ledger row logged — v1's row is NOT deleted,
per "log every trial including discards"): net Sharpe (5bps) 0.2139 (tiny
shift from v1, consistent with the no-trade band's removal changing turnover
slightly — same K2 ambiguous zone). **Canary result byte-identical to v1**
(base_t=3.876348339819923, lagged_t=4.421707974130559) — confirms the trip is
a genuine property of the signal, untouched by the capping/no-trade-band
layer, not an artifact of the bugs just fixed. Diagnostics now clean: 0
position-cap breaches, 0 category-cap breaches (both exactly at their caps on
the binding dates), gross exact to 2.0 ± 1e-15. Residual, spec-disclosed
imperfections remain and are reported honestly, not hidden: max net dollar
exposure 0.427 (21% of gross, one date), max ex-ante portfolio beta 0.52 (mean
exposures small: 0.025 / 0.053).

**Final audits (v2): leak-hunter NO LEAKAGE FOUND, reviewer APPROVE WITH
CHANGES** (both re-confirmed the fixes are real via independent
recomputation, not just reading code; reviewer's one standing point: the
canary trip must be surfaced explicitly before any K2 claim — which this
entry does). Two more trivial fixes applied post-review (dead-code guard
simplified; `m0_provider` now asserts the passed `config` matches the module
constants it actually uses, since it was previously read and ignored) —
verified behavior-preserving directly (identical gross value) rather than
re-running the full 5-minute battery a third time.

**Verdict: EXP-002 is NOT validated.** K2's numeric reading (0.214, ambiguous
zone) is moot — the pre-registered spec is explicit that a canary trip halts
"validated" status regardless of the Sharpe reading, and that stands. This is
not a Tier-1-null or a K2-fail in the clean sense; it is an honest
"the referee did its job and caught something real about this edge's nature"
result. Trial budget: 6/250 spent (EXP-001's 4 + PHASE3-M0 v1 + v2 — 2 real
executions against 1 planned, disclosed honestly: driven by an audit-found
implementation bug, not hypothesis-shopping; construction intent unchanged
between v1/v2).

**Open questions for the operator, not decided here:**
1. Should S1/S2 be retroactively canary-tested (they never were at EXP-001
   time), and if the time_shift trip is confirmed there too, does the
   Tier-1 "graduation" standard need a canary leg added going forward?
2. Is "the edge is substantially an untimed static tilt" itself disqualifying
   for a strategy whose whole premise is *predicting* relative returns, or is
   a partially-static tilt (still real, still measured honestly) an
   acceptable characterization worth pursuing into M1 with different
   treatment (e.g. it may combine differently with a genuinely timed
   candidate later)?
3. S6's sign-flipped variant (real continuation, not reversal) is NOT
   pre-registered and was explicitly not chased by the screening agent for
   HARKing reasons — does the operator want it proposed as a fresh,
   independently-justified confirmatory hypothesis (with its own economic
   rationale, not just "the opposite sign scored better")?
4. Proceed to M1 admission testing despite the canary situation, or treat
   this as the natural stopping point for the current signal set and pivot
   design attention elsewhere?

**Committed this round (pending):** `src/portfolio_m0.py`,
`tests/fast/test_portfolio_m0.py`, `research/phase3_m0/` (run_exp002.py,
prices_live_window.csv, battery_out/{battery_result.json,diagnostics.json}),
`research/decisions.md` (D26), `research/audit_log.jsonl` (5 new entries:
2 for the stale test fix/D22 redo earlier this session, 3 for EXP-002's
reviewer/leak-hunter passes), this journal entry.

## 2026-07-10 — Retroactive canary check on S1/S2 individually (operator item 1)

Per operator request, ran the existing frozen canary suite
(`validation.canaries.run_signal_canaries`) directly against S1 (momentum)
and S2 (TS trend) individually, on the same train/validation window EXP-001
used. Diagnostic only — read-only reuse of already-frozen/audited code and
already-computed signals, no new construction, no pre-registration needed
(not a new confirmatory trial; doesn't change EXP-001's graduation, which
was IC-legs-only and did not include canaries).

Command: inline script computing `momentum_s1`/`trend_s2` on
`data/panel/trainval_close.csv`+`eligibility_trainval.csv`, restricted to
rebalance dates in `[1999-12-22, 2021-12-31]`, canaries run against
`fwd5 = close.shift(-5)/close - 1`. Real output:

- **S1_momentum:** base t_nw 3.9187, mean_ic 0.0378. label_shuffle PASS,
  random_feature PASS, **time_shift FAIL** (lagged_t 3.1802, retains 81.1%
  of base).
- **S2_ts_trend:** base t_nw 3.1622, mean_ic 0.0307. label_shuffle PASS,
  random_feature PASS, **time_shift FAIL** (lagged_t 3.0283, retains 95.8%
  of base).

**Both individually fail time_shift, confirming M0's canary trip is not an
artifact introduced by the combiner alone.** But the failure PATTERN
differs meaningfully from M0's: S1/S2 individually DECAY under the 6-month
shift (81%/96% retained, still above the ≤50% pass bar) — plausibly a
largely mechanical consequence of ~12-month trailing lookback windows
sharing most of their underlying trading days across a 26-week shift, not
necessarily a hidden static tilt in the same sense as M0's finding. M0
uniquely STRENGTHENS under the shift (114% — leak-hunter's earlier
beta-hedge-decay explanation is specific to the neutralization step, not
inherited from S1/S2 raw). Read: the time_shift canary as currently
calibrated (shift=26 weeks, pass bar=50% retention) may be a cleaner test
for short-lookback signals than 12-month ones — a real interpretive
limitation worth remembering, not itself a leak or a reason to relax the
canary's threshold (no change made or proposed to `validation/canaries.py`,
which stays frozen).

**No action taken on the calibration question** — operator decision stands
(Q2 from the prior exchange): accept the static-tilt characterization for
M0, proceed, let the fuller validation battery (CPCV/DSR/sub-period
stability, eventually the one-shot holdout) be the arbiter rather than
re-litigating canary design now. Diversifying the signal library (Tier-2
carry, or another economically distinct family) remains the highest-value
next lever per the earlier redesign consult (D25) — this finding reinforces
rather than changes that read.
