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

## 2026-07-10 — Governance change (D27): standing architecture consult; deep Fable brainstorm

**D27 formalized** (research/decisions.md, FABLE_MISSION.md §3.7/§9, CLAUDE.md):
operator directed that the `design-reviewer` consult broaden from an event-triggered
"before presenting to the operator" check to a standing, whole-project, every-phase
rule — consulted at phase boundaries, after major audit findings, and whenever
weighing a new signal/combiner/direction, with authority to proactively flag or
redirect, not just endorse/refine/reject something finished. Rationale: extend the
existing cost-tiering pattern (cheap main-session driver, strongest-model auditors)
from occasional to continuous, so architecture-level judgment is never left solely
to the cheaper driver. Demonstrated the same day via the consult below.

**Fable-pinned architecture consult** (design-reviewer, model=fable, given full
project context — FABLE_MISSION.md, design.md, decisions.md D1-D26, recent journal
entries — and three open questions: more diverse candidate signals, alternative
combiner architectures, and backtesting's fundamental limitations for this project):

**Key reframe, more informative than anything asked for:** decomposing M0's
canary finding (constant per-asset tilt alone scores t=5.28 vs. live signal's 3.88;
the time-varying component ALONE scores only t=2.81) together with S3's
significant-WRONG-SIGN result (high-beta outperforms) and S6's wrong-sign result
(continuation, not reversal) paints a coherent picture: over 1999-2021 on this
70-ETF universe, price-ranking signals are substantially capturing the
**unconditional cross-asset risk-premium ordering** (risky beats safe, persistently)
rather than genuine timed alpha — and the quarantined holdout (2022-2026) begins
with the rate shock that inverted exactly that ordering. Flagged as the project's
single largest known risk, not solvable by more validation rigor.

**Q1 (diverse signals), ranked:** (1) **Carry (S5), bond+FX legs — the clear
priority**, needs new FRED yield data (already flagged D6, needs operator sign-off),
moderate-high decorrelation confidence, but honestly expected to ALSO show static
tilt (pre-register a static/timing decomposition alongside it so that's not another
surprise); (2) equity/REIT dividend-yield carry, needs new ex-date data; (3)
overnight-vs-intraday return decomposition (genuinely different information
channel — flow/clientele, not price-path), needs OPEN prices not in the current
panel; (4) high-volume return premium — the only candidate needing NO new data,
cheap exploratory pass; (5) fundamentals-anchored value — deferred, hard to get
point-in-time, and S6's wrong-sign result is a real yellow flag for value-style
mean-reversion in this universe. Explicitly recommended AGAINST: VRP (needs options
data we don't have), commodity carry roll-proxies (noise), and — explicitly —
**any sign-flipped S3/S6 variant**, named as the same smuggled-beta trade the
canary already caught and a HARKing trap the screening agent correctly refused.

**Q2 (combiner alternatives):** direct answer to "is there a smarter combiner
trick" — **no, full stop; the constraint is information, not combination
cleverness** (reconfirms D25's GLS-ceiling finding independently). M1 (XGBoost)
recommended DEFERRED until ≥3 economically distinct signals exist — with only 2
correlated inputs, any Sharpe improvement it shows should be read as suspicious,
not celebrated. Risk-parity/Bayesian-shrinkage/disagreement-filter/regime-gating
combiners all specifically ruled out as pointless at N=2 correlated signals. **The
one genuinely new architectural idea:** decompose each signal into its static
component (expanding-window mean) and timing component (residual), and test the
TIMING component alone against the existing graduation rule — 1-2 cheap trials that
directly answer "is there any timed information here at all," dissolving the
time_shift-canary interpretation problem at the root rather than patching around it.

**Q3 (backtesting limits):** candid assessment that this project's machinery
(purged CPCV, DSR/PBO, 5-canary suite, pre-registration, locked one-shot holdout)
is "at or beyond the published best-practice frontier" for what a backtest CAN
offer, and three things are irreducible by more rigor: one historical path (no
statistical trick manufactures independent history), non-stationarity of the
premium itself (the named risk above), and published-anomaly decay
(McLean-Pontiff's 26-58% post-publication haircut, already in our own evidence
brief, applies to this entire signal family). Concrete additive suggestions:
regime-stratified P&L reporting as a first-class deliverable; formally
pre-registering a decay haircut on expectations (e.g. "expected live Sharpe ~ 0.5x
validated backtest Sharpe"); and — the standout suggestion — a pre-registered
live/paper-monitoring contract with a falsifiable kill trigger, written now, which
extends the pre-registration discipline past the backtest boundary into the only
genuine out-of-sample test left: forward time.

**The ONE recommended next action:** build Tier-2 carry (bond+FX legs) as a
pre-registered EXP-003 IC screen, after operator sign-off on the new FRED
dependency, with a static/timing decomposition diagnostic pre-registered into the
screen itself; piggyback a 1-2-trial static-vs-timing decomposition of S1/S2 in the
same effort. Together, ~5 trials (11/250 budget) would answer whether this universe
contains any timed, diversified cross-sectional information at all, or whether the
honest terminus is a static risk-premium book the market-neutral mandate excludes.

**Not yet decided by me** — this is a rich set of recommendations for the operator
to weigh; no new trial, data dependency, or spec has been started as a result of
this consult without further operator direction.

## 2026-07-10 — Session resume: operator decisions on A-E (D28-D30); EXP-003 sequenced next

Context restoration verified before any work: CLAUDE.md + FABLE_MISSION.md re-read
in full; journal tail + decisions.md through D27 confirmed consistent with the
committed session-resume note (`research/RESUME_PROMPT_2026-07-10.md`, commit
9f3b164); `git status`/`git log` confirmed branch `research/gordian-v2` clean,
up to date with origin, HEAD at 9f3b164.

**D27 applied at this phase boundary, before presenting anything.** Rather than
re-running the prior session's open-ended Fable brainstorm, synthesized its
recommendations plus the operator's own open items (A-E) into a concrete proposed
sequencing and sent THAT to a fresh design-reviewer consult (Fable-pinned,
single agent, not a multi-agent workflow this time — the brainstorm stage was
already done) for a groundedness/integrity check before bringing it to the
operator.

**Verdict: REFINE**, not a clean endorse. Confirmed the synthesis was faithful to
the prior consult and to D25/D26/D27, but found five real gaps:
1. My "B before A" framing conflated *gate* (B's result should decide whether to
   proceed with A) with *calibration* (B's result informs A's spec but doesn't
   block it) — two different proposals I hadn't distinguished. Recommended
   calibration as the sounder reading, since carry's timing content comes from
   yield curves, not price paths — a null on momentum/trend timing doesn't
   mechanically predict a null on carry timing.
2. Declining sign-flip rescues (item C) is a HARKing side-door unless it carries
   a **prior-exposure clause**: S3 and S6 are both already data-exposed (t=-3.06
   counted; t=+2.90 sign-flipped, exploratory), so any later "fresh, independently
   motivated" continuation/high-beta hypothesis needs to disclose that exposure
   and clear a stricter bar, not just show up under new wording.
3. The canary-leg question left open at EXP-002's close (should Tier-1 graduation
   itself require a canary leg?) needed a decision now, before EXP-003/EXP-004's
   specs freeze — not after a third surprise trip.
4. B's spec had unpinned degrees of freedom (exact expanding-window definition,
   trial count, and whether a timing-component failure retroactively touches
   S1/S2's D23 graduation — it does not).
5. Trial-budget arithmetic should be stated explicitly, not left implicit: 6/250
   spent; B adds ~2; the carry package adds ~5 → ~13/250 if both proceed.

Also flagged (non-blocking, folded into the specs when written rather than
decided now): a quarantine-adjacent detail for carry (holdout-period FRED yields
need a stated storage answer, per D18's precedent) and a note that
regime-stratified reporting (one of Fable's Q3 suggestions) needs its bucket
definitions frozen once, by rule, or it becomes a post-hoc slicing tool.

**Operator decisions (AskUserQuestion, same mechanism as D15/D20/D26) — logged
as D28, D29, D30 in decisions.md:**
- **A (FRED/carry): green-lit now** (D28). Build order sequenced — EXP-003 (item
  B) runs first since it needs no new data; the carry screen (to be pre-registered
  as **EXP-004**, not EXP-003 as earlier prose in this journal loosely suggested)
  begins once EXP-003 closes out.
- **C (sign-flip rescue): ratified WITH the prior-exposure clause** (D29) — the
  stricter reading design-reviewer required, not the simpler blanket version.
- **Canary leg on graduation: added now, diagnostic only** (D30) — reported at
  graduation time going forward, not blocking, given the still-open question of
  how well the 26-week time_shift canary is calibrated for longer-lookback signal
  families.
- **Sequencing: confirmed B first.**

**Next: EXP-003 (S1/S2 static-vs-timing decomposition) — pre-registration in
progress, not yet run.** Per B's own framing: split each of S1/S2 into a
point-in-time expanding-mean "static" component and a residual "timing"
component, test the timing component alone against the unchanged Tier-1
graduation rule (`validation.ic.passes_graduation`), plus report canaries
diagnostically per D30. Two planned trials (S1_timing, S2_timing); static
components' IC reported descriptively, same pattern as EXP-001's inter-signal
correlation. No code has run on real data yet this session — the spec is being
written next, to be frozen before any real computation, per mission §3.3c.

## 2026-07-10/11 — EXP-003 executed, audited, and consulted (D31). NOT enrolled anywhere yet.

**Implementation.** `src/decomposition.py` (`expanding_static_timing`: NaN-aware
expanding mean via `shift(1)` + cumsum/count, matching `seasonality_s4`'s
device) + `tests/fast/test_decomposition.py` (8 hand-computed tests: basic
expanding mean, NaN gaps excluded from count/sum rather than treated as zero,
the min_prior_obs boundary exactly, static definable when the current signal
is NaN, column independence, PIT truncation invariance, static+timing
reconstruction). All passed on first write; full `tests/fast tests/data` suite
green at 84.

**Real run.** Duration estimated at "under 3 minutes, likely 30-120s" (reasoned
from EXP-001's 7.87s baseline plus the canary suite's ~150 repeat IC
computations per signal). Actual: **11m44s** — a real divergence (6-24x over),
root-caused as the frozen `validation/canaries.py`'s per-date Python loop
inside `cross_sectional_ic`, repeated ~150x per signal by the label_shuffle/
random_feature canaries; not a thread-cap or correctness issue, just an
inherent cost of the existing frozen canary code applied to 2 signals'
canaries where EXP-001 ran none. Real result
(`research/exp003_timing_decomposition/results.json`, train/val only,
1999-12-22..2021-12-29, holdout never touched):
- **S1_timing PASSES**: mean_ic=0.0218, t_nw=2.372, 68.2% years-positive.
- **S2_timing FAILS** on the t_nw leg alone: 1.750 vs the 2.0 bar (mean_ic
  0.0166 clears 0.01; years-positive 63.6% clears 60%).
- Static components (descriptive only): S1 t=3.326, S2 t=3.514 — both higher
  than their timing counterparts.
- Canaries (diagnostic, D30): **both timing components fail time_shift with
  the lagged t EXCEEDING live** (S1 2.372->2.533; S2 1.750->2.469) — a
  stronger anomaly than the raw signals' mere 81%/96% retention.
- Ledger: exactly 2 real rows (EXP-003-S1_timing, EXP-003-S2_timing),
  independently recount-verified; static/canaries correctly excluded from
  the ledger. Budget: 8/250.

**Audits (writer≠verifier).** reviewer (first attempt): **APPROVE** —
independently recomputed via a different algorithm to 1.8e-15, confirmed
exact spec conformance, trial hygiene, and non-tautological tests; flagged
the canary anomaly as a design-reviewer question, not a code defect, plus 4
minor non-blocking nits (year-2000 IC-bucket granularity — confirmed no leg
flips; a docstring gap; a spec-text wording nit with no behavioral effect; a
fragile string-parse in the runner). leak-hunter's first attempt **failed
mid-run** on a session/API limit (post-restart the operator flagged other
things might drop the same way) — retried clean. leak-hunter retry: **CLEAN**
across all 8 mandated refutation axes, including an empirical truncation
attack (bit-identical at multiple cutoffs, both decomposition-only and
end-to-end) proving no look-ahead. Root-caused the canary anomaly
mechanistically as **"tilt re-injection"**: `timing(t-26) = [W(t-26)-static(t)]
+ [static(t)-static(t-26)]`; the second term (26 weeks of expanding-mean
updates missing from the stale static estimate) is itself strongly predictive
(t=2.74/2.70) because it's a slice of the dominant static tilt (t=3.33/3.51).
Live timing subtracts the *most complete* tilt estimate (cleanest, weakest
residual: t=2.27/1.77 on matched dates); shifting partially undoes that
subtraction. Structural property of subtracting a slowly-updated mean, not a
leak — confirmed via truncation attacks it needs no future information.
Both verdicts logged to `research/audit_log.jsonl`.

**Design-reviewer consult (Fable, D27 standing rule — triggered by an
ambiguous result: partial pass + a mechanistically-explained-but-still-present
canary anomaly).** Verdict **REFINE** on the architect's (session driver's)
proposed characterization — a real correction, not a formality:
- **S1: mostly right, one overclaim.** "Genuine timed information on a
  stronger static tilt" is fair and matches the spec's own pre-registered
  reading. But drop "dominant" (IC components are correlated, not additively
  attributable — S1 static 0.0287 + timing 0.0218 = 0.0505 exceeds raw S1's
  0.0378, so proportional-share language claims a measurement never made),
  and don't let "graduates" read as "strong": t=2.372 is the weakest
  graduating t of the project so far, echoes M0's own timing-only t=2.81 from
  EXP-002 (low-novelty confirmation, not new discovery), and would not clear
  the HLZ t>=3.0 bar design.md §9 applies to a final candidate.
- **S2: one real error, corrected.** "Almost entirely static tilt" is
  WRONG. S2_timing's ~95% CI (from the reported NW se) spans roughly
  [-0.002, +0.035] — contains BOTH zero AND S1_timing's 0.0218. The data
  cannot distinguish "S2 has no timing signal" from "S2's timing is
  comparable to S1's." Correct framing: a near-miss null (falsified per the
  frozen spec's rule, correctly not claimable), not a demonstrated absence of
  timing information — the test isn't powered to rule that out.
- **EXP-004 (Tier-2 carry): D28's authorization stands unconditionally** —
  EXP-003 was calibration, not a gate, and came out on the proceed side
  anyway. Five requirements before freezing the spec: (1) reuse
  `src/decomposition.py`/MIN_PRIOR_OBS=52 verbatim as a descriptive leg, no
  retuning; (2) pre-declare the expected tilt-re-injection canary signature
  and its two-condition adjudication test (quantitative identity accounting +
  truncation attack, else treated as a live trip) BEFORE running, for both
  the timing residual and the separately-expected raw long-lookback decay;
  (3) raw carry stays the only confirmatory trial, timing/canaries
  descriptive; (4) add carry-vs-S1/S2 static/timing correlation as a required
  descriptive diagnostic — the number that actually says whether the FRED
  dependency buys real breadth or just re-finds the same static ordering;
  (5) pin the cross-sectionalization method (within the ~10-name bond sleeve
  vs panel-wide-with-NaN) and the holdout-period FRED storage question in the
  spec before pulling data.
- **Canary-interpretation-for-residual-signals: institutionalize as a class,
  but carefully.** This is the SECOND mechanistically-root-caused instance of
  the same structural family (EXP-002's beta-hedge-decay; EXP-003's
  tilt-re-injection — both "X minus a slow estimate of X's persistent
  component" re-acquiring that component under a shift). Keep it separate
  from the unrelated long-lookback-retention limitation (don't blanket-merge
  into "time_shift is unreliable here"). Attach the two-condition adjudication
  test so it's a testable rule, not a waiver. Must be ratified by the
  OPERATOR as a decision row, not just architect prose — because
  `run_signal_canaries.all_passed` still gates strategy-level "validated"
  status and K4 halts on trips, so this returns WITH BLOCKING FORCE at Phase
  5 if any residual-class signal is in the eventual book. No change to the
  frozen canary code itself.
- **Proactive flags:** (1) state explicitly that EXP-003 graduated a
  diagnostic object, not a new tradeable signal — enrolling S1_timing in any
  book is a fresh proposal needing its own decision row and trial, not an
  automatic consequence of clearing the same numeric rule; (2) **the real
  decision this tees up**: does the eventual book trade the static tilt
  (permitted by the mandate — dollar/beta-neutral static risk-premium
  harvesting is a legitimate identity, just a different one than "timed
  alpha," with the edge concentrated in exactly the component the 2022-start
  holdout is known to punish) or strip it and keep only the thinner,
  currently one-signal timing story, or deliberately blend and report both
  separately — a genuine architecture fork for the operator, not resolved by
  EXP-003 itself; (3) pre-commit now: no S2_timing rescue-by-reparameterization
  will be attempted (t=1.75 is a near-miss and the pull to retune
  MIN_PRIOR_OBS will be real; already forbidden by the frozen spec, echoed
  here); (4) 8/250 spent, immaterial as budget but a reminder that a marginal
  t=2.37 at trial 8 is exactly what DSR will deflate later.
- Confirmed §7 bar reached: bundle to the operator (result +
  corrected characterization, the tilt-vs-strip book-identity question, the
  canary-interpretation ratification ask, the EXP-004 5-requirement plan) —
  not a halt, since no leak/contradiction exists and D28 stands.

**Process note, unrelated to the science:** partway through this session, two
stray Agent invocations with literal "placeholder" content (description and
prompt both "placeholder") appeared, attributed to the session driver. First
one was initially (incorrectly) flagged to the operator as a possible
external anomaly/prompt-injection; on the second occurrence it became clear
this was the driver's own erroneous pattern for ending a turn while waiting
on a background task, not anything external — corrected by simply ending
turns with plain text and no tool call when waiting. Both stray agents
behaved harmlessly (recognized the empty input, asked for clarification, no
file/tool access), and neither was built on or trusted. Noted here for
completeness, not because it affected any result above — the two audits and
the design-reviewer consult that matter for EXP-003 are the deliberately
launched ones (agentIds ending `...abc21af7fe8910f6f` [failed retry],
`...a4691a8ff19e5e1ee`, `...a9d6258f0c353d89e`, `...a90834d93c372e801`).

**Open for the operator (not decided here):**
1. Book identity: trade the static tilt, strip it, or deliberately blend +
   report separately?
2. Ratify the canary-interpretation-for-residual-signals rule (with the
   two-condition adjudication test) as a standing, blocking-relevant
   decision — given it returns with force at Phase 5 for any residual-class
   signal in the eventual book?
3. Green-light drafting the EXP-004 (Tier-2 carry) spec with the 5 frozen
   requirements above, now that EXP-003 has closed out?

**Committed this round (pending):** `src/decomposition.py`,
`tests/fast/test_decomposition.py`,
`research/exp003_timing_decomposition/{run_exp003.py,results.json}`, 2 new
`research/experiments.jsonl` rows, 2 new `research/audit_log.jsonl` entries,
`research/decisions.md` (D31), this journal entry.

## 2026-07-11 — Operator round 2: book identity, canary rule, carry scope (D32-D34); EXP-004 spec drafted, corrected twice, frozen (D35)

Operator answered the three open items from the prior entry, all via
AskUserQuestion, logged as **D32** (broaden EXP-004's carry construction to
include FRED credit-spread data for LQD/HYG/JNK/EMB, not just the treasury
curve -- presented as a genuine fork per the org-level "present, don't pick
silently" instruction, since a treasury-only construction would mostly
re-test duration/term-premium rather than genuine cross-asset carry),
**D33** (ratify the two-condition canary-interpretation rule for
residual-class signals going forward, with blocking force at Phase 5), and
**D34** (blend the static-tilt and timing-residual book identity, report
both separately, defer the final trade/strip choice).

**EXP-004 spec drafting.** First draft (`specs/EXP-004-tier2-carry-ic-screen.md`)
attempted a TS-z-score construction to make yield-level and spread-level
carry comparable across legs. A design-reviewer consult (REFINE, blocking)
found this was a real, non-obvious error: per-asset TS z-scoring against an
expanding history IS `src/decomposition.py::expanding_static_timing`'s
residual operation, so the "raw" signal was already a timing-like object --
this would have made the static/timing decomposition diagnostic near-
circular, misdirected the correlation diagnostic, and mismatched the
pre-declared canary expectation. The same consult found the proposed
holdout-period FRED clear-storage rationale had INVERTED its own mirror
argument (train/val prices clear + holdout prices locked implies train/val
FRED clear + holdout FRED locked -- the draft concluded the opposite) and
silently contradicted D18 without citing it; flagged unaddressed
publication-lag/point-in-time handling; and left the AGG/BND duration-
mapping unpinned.

Because `specs/` files freeze on creation (the quarantine guard blocks any
further Edit/Write the moment a file exists, regardless of review status --
not just after an explicit freeze step), the first draft could not be
patched in place. Following the `REFEREE-SELFTEST-v2` precedent, wrote a
new versioned file, `specs/EXP-004-tier2-carry-ic-screen-v2.md`, marking v1
explicitly abandoned/superseded before any FRED data was pulled or any real
computation happened (squarely inside normal Explore-then-Plan iteration,
not post-hoc goalpost-moving -- nothing had run yet). v2: level-based excess
carry (Treasury CMT minus 3-month bill; OAS directly for credit names)
restoring a genuine raw-carry test; holdout FRED lockboxed via the existing
D12/D21 mechanism, correcting course back to D18's principle rather than
departing from it; a pinned as-of merge for publication lag (t-1 business
day, 5-day max staleness); a pinned AGG/BND/TIP duration-tenor mapping rule.

**TIP handling required resolving a tension the first consult's own two
recommendations left unstated:** it suggested both "maybe drop TIP to a
9-name sleeve" (part of the construction fix) and "keep min_names=10
unchanged" (to stay consistent with the frozen canary machinery) --
`validation/canaries.py` calls `cross_sectional_ic` internally without
passing `min_names`, always using the frozen module default of 10, so a
9-name sleeve would make the entire canary suite silently return
degenerate (falsely-passing) results, not visibly broken ones. Resolved by
keeping TIP in a 10-name sleeve via a nominal-equivalent construction
(TIPS real yield + matched breakeven inflation), flagged transparently as
an additional FRED series.

A second, narrowly-scoped verification consult (not a full re-review, just
checking the fixes) found 4 of 5 issues cleanly resolved, plus one further
correction: FRED's breakeven series is DEFINED as nominal CMT minus TIPS
real CMT at the same tenor, so "real yield + breakeven" is an exact
arithmetic identity for the nominal CMT -- no third FRED series is actually
needed, and the disclosed consequence (TIP's ~6.5-7y duration is close
enough to IEF's ~7.5-8y that both may map to the same nearest tenor,
creating a permanent rank tie on some/all dates -- power-reducing,
disclosed, not engineered around) is now pinned. Logged as a decisions.md
addendum (**D35**) rather than a v3 file, since it is a one-paragraph
clarification with zero trial cost and no data pulled yet.

**EXP-004 (v2 + the D35 addendum) is now the frozen pre-registration,
ready for FRED implementation next** -- matching the operator's own choice
("draft the spec now, ready to implement next"). No FRED data has been
pulled; no real computation has happened. Planned trial count 1
(`EXP-004-S5_carry`); budget would move from 8/250 to 9/250 once run.

**Committed this round:** `research/decisions.md` (D32-D35),
`specs/EXP-004-tier2-carry-ic-screen.md` (v1, abandoned),
`specs/EXP-004-tier2-carry-ic-screen-v2.md` (v2, active), this journal
entry. No code, no data pull, no ledger rows this round -- pure design/
pre-registration work, consistent with "draft the spec now" being the
authorized scope for this step.

**Next session:** implement the FRED data pull (new dependency, per D28/D32
-- point-in-time, full history, holdout rows lockboxed per D35), the S5
carry construction, and run EXP-004 against the frozen v2 spec.

## 2026-07-11 — FRED API access configured and verified (setup only, no data pulled)

Operator provided a FRED API key. Stored as `FRED_API_KEY` in
`/workspace/activate.sh` (outside the git repo, never committed -- the
operator ran the append command directly; two attempts landed in the file
with different values because the first `echo >>` produced no stdout and
was assumed to have failed, so it was re-run -- confirmed with the operator
which was correct, and the stale line removed by exact line number via
`sed`, never by content-matching, so the key itself was never re-printed
into a command or its output beyond the operator's own original message).

Verified live against the real API (not just presence-checked): a raw
REST call to `/fred/series/observations` for DGS10 returned real data
(16,833 observations, earliest 1962-01-02); `fredapi==0.5.2` installed via
`uv pip install --python /workspace/venv/bin/python fredapi` and
smoke-tested end-to-end (fetched DGS10 for a real date range). Pinned in
`requirements.lock.txt` (alphabetical position, no new transitive
dependencies -- `requests`/`urllib3` were already pinned at the exact
versions fredapi resolved to). `pytest tests/fast tests/data -q` ->
**84 passed** after the install, confirming nothing broke.

This is setup only: no FRED series data has been pulled or stored, no
EXP-004 code has been written, no ledger row logged (nothing to log --
installing a verified, already-approved dependency's client library is not
itself a trial). `research/RESUME_PROMPT_2026-07-11.md` updated twice
today to keep it accurate as state changed (git-tracked, so the diff is
the record: key configured -> verified+installed).

## 2026-07-11 — EXP-004 implementation blocked before any data pull: credit-leg OAS data structurally unavailable pre-holdout (D36)

Began EXP-004 implementation per the frozen v2 spec. Verified FRED series IDs
live via `fredapi` before writing any construction code, per the spec's own
"implementation-time verification, not guessed at design time" framing:

- **Treasury leg: clean.** Nominal CMT (DGS3MO/DGS6MO/DGS1/DGS2/DGS3/DGS5/
  DGS7/DGS10/DGS20/DGS30) all daily, long history (1962-1981 onward).
  TIPS real CMT (DFII5/7/10/20/30) daily from 2003. Fund effective durations
  looked up from current issuer fact sheets (iShares/BlackRock, Vanguard;
  WebSearch, dated sources): SHY ~1.9-2.0y, IEF 7.2y, TLT 15.20y (2026-07-08),
  AGG 5.78y (2026-03-31), BND ~5.7-5.8y, TIP 6.41y (2026-03-31). Applying the
  frozen nearest-tenor rule: SHY->2, IEF->7, TLT->20, AGG->5, BND->5, TIP->7.
- **Credit leg: BLOCKED.** Live-queried `BAMLC0A0CM` (IG OAS), `BAMLH0A0HYM2`
  (HY OAS), `BAMLEMCBPIOAS` (EM OAS) — all three (and every rating-bucket
  variant checked) first-observe **2023-07-11**, confirmed via a direct
  `fred.get_series()` pull, not just search metadata. This is FRED/ICE's
  well-documented 2022 licensing event: FRED had to drop historical vintages
  of ICE-sourced index data and now carries only a rolling ~3-year trailing
  window (2023-07-11 is exactly "today minus 3y" — a design-reviewer
  observation, not mine originally; the gap never closes by waiting).

**Consequence: the frozen v2 confirmatory test cannot run.** The holdout is
2022-01-01..2026-06-30 (D8); 2023-07-11 onward sits entirely inside it. The
credit leg (LQD/HYG/JNK/EMB) has zero FRED-sourced values on any train/val
date, so the `min_names=10` combined test — and even the `min_names=4`
credit-only leg-attribution diagnostic — cannot score a single pre-holdout
date. This is a data-infeasibility wall, not a weak or noisy result.

**Design-reviewer consult (Fable, D27 standing rule — major/unexpected
finding, before any operator presentation).** Verdict: ENDORSE the read that
this is a §7 pause, REFINE on the option set. Independently re-verified the
finding and additionally checked the one rescue I hadn't: ALFRED vintage
archives (`get_series_as_of_date`, `get_series_all_releases`) — confirmed
dead, no point-in-time path to pre-2022 OAS exists anywhere on FRED. Ruled
out a Moody's-proxy substitute (BAA10Y is daily since 1986 but only covers
the IG name; HYG/JNK/EMB stay dead, and dropping below 10 names silently
degenerates `validation/canaries.py`'s hardcoded `min_names=10` default —
the exact problem D35 kept TIP in-sleeve to avoid). Confirmed the
TIP/breakeven tenor question I'd flagged as a secondary snag is **already
closed by D35** (breakeven = nominal-minus-real means TIP needs no
breakeven series at all, just DGS7 minus the 3mo bill) — dropped that
non-fork from the presentation. Zero trials consumed; ledger untouched;
logged as **D36** (architect finding, pending operator direction).

**Two live options identified, presented to the operator (not decided
silently, per the org-level "present forks" instruction):**
1. Defer Tier-2 carry, log the blocker, return to Phase 3 with S1+S2
   (already graduated, D23) — zero new spec, zero trial cost.
2. Freeze a narrower v3 spec testing the 6-name TREASURY-ONLY leg (reverting
   D32's credit-leg broadening), with two defects pinned ex-ante if chosen:
   a <10-name sleeve breaks the frozen canary suite's hardcoded default, and
   the tenor-mapping rule collapses 6 names to only 4 distinct carry values
   every date (IEF/TIP tie at tenor=7 — the tie D35's addendum already
   predicted, now confirmed live).

No FRED data pulled, no S5_carry construction written, no ledger row logged.
Awaiting operator decision.

## 2026-07-11 — Operator round 3: keep the credit leg, build a yfinance proxy (D37-D38); EXP-004 v3 frozen

Operator's read on D36: carry is rated the strongest candidate signal in the
literature (D6), so losing the credit leg to a data-access problem is worse
than the cost of building a careful substitute. Chose neither literal D36
option — not a full defer, not treasury-only — but a third path: **D37**,
build a yfinance-derived proxy for the credit leg (LQD/HYG/JNK/EMB), keeping
all 10 names, with the explicit instruction to "do the construction properly
to avoid the noise."

**Construction designed:** each credit name's trailing-twelve-month (TTM)
distribution yield (sum of ex-dividend distributions over the trailing 366
days ÷ raw non-dividend-adjusted close) minus a duration-matched treasury
CMT yield (LQD 7.88y->7y tenor, HYG 2.91y->3y, JNK ~3.08y->3y, EMB
6.61y->7y — funds' durations looked up the same way as the treasury leg's).
TTM chosen deliberately as the external, industry-standard fund-yield
convention (Morningstar/ETF.com), not a window tuned against this signal's
IC. Treasury leg unchanged from v2.

**Design-reviewer consult (Fable, D27) on the full draft, before writing
anything to `specs/`.** Verdict: REFINE. Caught one real factual error in my
first draft — I'd claimed "verified: no splits" for the 4 credit ETFs; JNK
actually had a 1-for-3 reverse split on 2019-05-06 (harmless to the
construction once checked: yfinance back-adjusts both price and dividends
to the same basis, so the TTM ratio is split-invariant, but the false
"verified" claim itself was exactly the kind of defect the mission's
evidence discipline exists to catch). Also: re-sourced LQD's duration
directly from iShares (7.88y, I'd only had a sibling-fund proxy), confirmed
duration-matching the treasury subtraction (not just the 3mo bill) is the
right call to avoid re-absorbing term premium, found a real gap in holdout
storage (raw yfinance inputs weren't covered, only derived carry), corrected
the disclosed-limitation's episode attribution from an assumed "credit
stress periods" framing to the actually-measured cause (year-end special
distributions — JNK 2010-12-29, EMB 2018-12-18, both ~3-5x typical), and
added ex-date-count diagnostics as a data-quality detector. All incorporated.

**D38 (operator decision, AskUserQuestion):** for holdout-period storage of
the new yfinance-derived data, chose a SECOND independent lockbox
(`data/lockbox_carry/`, a new token `/workspace/OPERATOR_TOKEN_CARRY.txt`)
over design-reviewer's D18-pattern alternative (retain nothing, recompute at
Phase 6). Keeps holdout carry bit-reproducible now; costs a second token to
secure and a future logged `final_eval.py` change at Phase 6 (not needed
today).

**`specs/EXP-004-tier2-carry-ic-screen-v3.md` frozen** — supersedes v2
(infeasible, not wrong). Same hypothesis, same graduation rule, same
min_names=10, same planned trial count (1, `EXP-004-S5_carry`, budget
8/250->9/250). No FRED or yfinance data pulled yet; no ledger row logged.

**Next: actual implementation** — confirm remaining FRED series (already
mostly done pre-blocker), build the combined data pull (FRED treasury +
yfinance credit), construct S5_carry, run the confirmatory test + all
diagnostics, lockbox holdout rows into the new `data/lockbox_carry/`, audit,
log, commit.

## 2026-07-11 — EXP-004 executed against v3: a units bug, then a graduating result that is not what it appears to be

**Implementation.** `research/exp004_tier2_carry/run_exp004.py`: FRED treasury
leg (`fred_asof`, t-1 business-day lag / 5-business-day staleness cap, same
convention as v2) + yfinance credit leg (`ttm_yield_and_coverage`: trailing
366-day ex-dividend distribution sum, at cutoff `t-1bd`, divided by raw
non-dividend-adjusted close, floored at 8 distinct ex-dates in-window).
Smoke-tested both pieces on real data before the full run (fred_asof against
a hand-checked DGS7 window; JNK's TTM yield confirmed continuous across its
2019-05-06 1-for-3 reverse split, as the spec's split-invariance argument
predicted).

**First run — a real bug, caught by my own pre-audit, not by a subagent.**
Stated a 5-10 minute duration estimate (matching the spec) and ran the full
battery: **S5_carry FAILED**, mean_ic=-0.050, t_nw=-2.403 — significant in
the WRONG direction. But the leg-attribution diagnostic showed BOTH legs
individually PASSING on their own (treasury t=2.09, credit t=2.02) — a sign
flip on combination is exactly the kind of surprise the mission's "surprise
= suspicion" principle exists for, so I checked before reporting anything.
Root cause, found directly: a UNITS BUG. FRED Treasury CMT yields are
percentage POINTS (DGS7≈4.40, meaning 4.40%); my TTM yield was a raw decimal
FRACTION (≈0.05, meaning 5%). `credit_carry = ttm_yield - matched_cmt`
subtracted a ~2-7-point FRED number from a ~0.03-0.09 decimal, so
credit_carry collapsed to ≈`-matched_cmt(t)` — an accidental, sign-flipped
proxy for treasury yield LEVELS, carrying essentially none of the intended
credit information. Confirmed via a direct rank check: LQD averaged rank
1.00 (ALWAYS lowest of 10), TLT averaged rank 10.00 (ALWAYS highest) — a
complete, zero-overlap partition between the two legs, impossible for a
genuine carry signal and a dead giveaway of a unit-scale artifact.

**Fix:** multiply TTM yield by 100 before subtracting (`run_exp004.py:99`,
with an inline comment recording what happened and pointing here). Re-ran;
the rank distribution is now sane and interleaved across legs (no fixed
partition), and values are economically plausible (JNK/HYG highest carry,
SHY lowest, exactly as expected).

**Ledger handling of the buggy run (a judgment call, made deliberately, not
by default):** the buggy run's ledger row (`EXP-004-S5_carry`,
config_hash=`00d95153d4af`, mean_ic=-0.050) is NOT deleted — the mission's
"log every trial including discards" principle is explicit that a real
execution against real data counts, bug or not; deleting it would look like
quietly discounting an inconvenient trial from the DSR-relevant count. The
corrected run is logged as a SECOND, distinct row (config_hash=`866821022d35`
via a `construction_rev` marker added specifically so the ledger's own
dedup-by-id logic wouldn't silently treat the fix as "just a
reproducibility re-run"). **Budget correction: this experiment consumes 2
trials, not the 1 the frozen spec estimated (8/250 -> 10/250, not 9/250)** —
the spec's trial-count estimate could not have anticipated a mid-experiment
bug fix; disclosing the discrepancy here rather than silently using the
spec's stale number.

**Second (corrected) run — S5_carry PASSES graduation, but the passing
number is not what it looks like.** mean_ic=0.100, t_nw=5.746,
pct_years_positive=92.9% — by a wide margin the strongest result in the
project's history (EXP-001's S1/S2: t=3.92/3.16; EXP-003's S1_timing: a
marginal t=2.37). Per Sec3.3f, an unusually strong result is presumed a
defect until disproven, not celebrated — so before telling the operator
anything, I checked the static/timing decomposition already required by the
frozen spec: **static component t_nw=4.81 (essentially all of the raw
signal's strength, raw-vs-static correlation 0.88); timing component
t_nw=-1.08, NEGATIVE and not significant, 46% years positive (worse than a
coin flip).** The raw signal also FAILS the time_shift canary (base_t=5.75,
lagged_t=5.08 — barely decayed at all under a 26-week shift).

**Writer≠verifier audits (both fresh-context, given only the code/results/
spec, not this reasoning).** `reviewer`: **REQUEST-CHANGES** — independently
reproduced the confirmatory number bit-identically from a fresh FRED/
yfinance pull (confirms the number itself is real and correctly computed,
not a run-environment artifact); found 3 blocking records defects (detailed
below) and 3 non-blocking construction nits (TTM window anchored ~1-3
calendar days earlier than the spec's literal wording; the raw-close
denominator's `ffill(limit=3)` permits ~3 REBALANCES stale, not 3 days,
since it operates on the already-weekly-reindexed series, and isn't itself
pinned in the spec — both immaterial to the result, disclosed here rather
than fixed-and-rerun to avoid manufacturing a third trial for a cosmetic
deviation). `leak-hunter`: **CONCERNS-FOUND** — PIT/look-ahead clean
(empirical truncation attacks at 2015-12-31 and 2019-12-31 bit-identical,
max diff 0.0), quarantine clean, decomposition usage confirmed correct
(identity holds to 4.4e-16). The substantive finding, independently derived
(I did not tell leak-hunter my own read of the decomposition before it
ran): **rank-stability analysis on the 680 confirmatory dates shows
week-over-week rank autocorrelation of 0.989 — near-total persistence. A
LITERALLY CONSTANT signal (each name's full-sample mean carry, held fixed
on every date) scores t_nw=5.33, 85.7% years positive — 98% of the raw
signal's mean IC, correlation 0.966 with the real, time-varying signal's IC
series.** S5_carry contains essentially zero per-date discriminating
information: the t=5.75 is the t-stat of ONE constant, persistent bet (long
JNK/HYG/EMB/TLT — high-carry/higher-duration/credit names; short SHY/AGG/
BND — low-carry/safe names) held through the single 2009-2021 credit-and-
duration bull regime, with Newey-West(2) treating 680 weekly
quasi-independent-looking observations that are economically closer to one
regime-length observation. leak-hunter independently re-ran the canary
suite and confirms `all_passed=False` for the raw signal, which per Sec3.3d
means **S5_carry does not reach "validated" status despite clearing the
graduation math** — the pre-registered numeric rule and "validated" are not
the same thing, and this is the clearest case yet in the project of that
distinction actually mattering.

**This is the D29-named risk, arriving exactly as warned.** D29's rationale
cites "the unconditional risky-beats-safe ordering" as "the project's
single largest known risk" (from the 2026-07-10 Fable consult). S5_carry's
entire graduating edge is that risk, measured directly: a static tilt
toward duration/credit exposure that worked over one specific post-crisis,
low-rate, spread-compression regime — precisely the ordering the 2022-start
holdout is known to have inverted (2022 was the year long-duration bonds
AND credit both fell together while cash/short-duration held up best). This
is not a new discovery of the risk in the abstract — it is that risk
showing up, concretely, inside a pre-registered confirmatory test, mechanically
clearing the same numeric gate S1/S2 cleared for better reasons.

**Records defects found by the audits, being corrected now (not
construction bugs, no new trial required):**
1. The second (FRED-carry) lockbox was built during the BUGGY first run
   (`research/lockbox_access.log` timestamp matches the buggy run, not the
   corrected one) and never rebuilt — the corrected run hit `FileExistsError`
   (by the same one-shot design as the price-panel lockbox) and its
   corrected payload was never written. The ledger's second row cites that
   unwritten sha as if it were stored — misleading, now flagged. Needs the
   operator to delete the stale lockbox+token pair (same D17/D21-precedent
   mechanism) before a corrected lockbox can be built; not something I can
   do myself (quarantine-guard-blocked, by design).
2. This entry itself closes reviewer's "dangling citation" finding — the
   code comment referencing "the diagnostic that caught this" pointed to a
   journal entry that didn't exist yet at review time. It exists now.
3. Trial budget corrected above (10/250, not the spec's stale 9/250
   estimate).
4. `specs/EXP-004-tier2-carry-ic-screen-v3.md` was uncommitted at audit
   time (weaker freeze-timing evidence than EXP-003's git-anchored
   precedent, per leak-hunter) — will be committed together with this round.

**Audit verdicts logged** to `research/audit_log.jsonl` (both entries,
verbatim summaries above).

**Design-reviewer consult (Fable, D27) on characterization — REFINED two
things I had wrong in the paragraphs above, before this reached the
operator:**
1. **Not "falsified."** The frozen spec's falsification clause was not
   triggered — all three legs of the graduation rule passed. Declaring
   falsification post-hoc because the pass is hollow would be goalpost-
   moving in the honest direction, but still goalpost-moving, and would
   corrupt the ledger's meaning. The hypothesis as pre-registered was
   **confirmed**; what's in question is what that confirmation is worth.
2. **Not "fails to reach validated status" as though that's a stage it
   missed.** Canaries are diagnostic/non-blocking FOR GRADUATION (D30);
   "validated" is a strategy-level status (Phase 5/6) no signal reaches at
   Phase 2 — S1/S2 aren't "validated" either. Graduation and validation were
   never the same gate; this is just the first result where the gap between
   them is load-bearing. Correct framing: S5_carry graduates under the
   frozen rule (real, PIT-clean, independently reproduced number); the
   spec's own required diagnostics show the graduating edge is a static
   tilt, not carry-timing information (static t=4.81, timing t=-1.08
   negative, a literally constant per-name signal reproduces 98% of the
   IC); this is D29's named risk arriving inside a passing confirmatory
   test; its raw form could not survive Phase 5 while the canary trip
   stands.
3. **Regime correction:** 2008 (partial coverage — the sleeve isn't fully
   eligible until 2008-12-17) is actually the single best year in the
   window (IC +0.478), plausibly a late-2008 duration rally on thin early
   coverage — say "2008(partial)-2021," not "the 2009-2021 regime."
4. **A finding that outlives S5, worth surfacing on its own:** the
   constant-signal control (a literally time-invariant per-name ordering
   scoring t=5.33, 98% of the real signal's IC) demonstrates the Phase-2
   graduation rule (mean_ic/t_nw/pct_years_positive) has a measured blind
   spot for slow, persistent signals — it cannot distinguish genuine
   evolving relative-attractiveness information from a fixed bet that
   happened to pay over the sample. This is a candidate methodology
   refinement (e.g., report a constant-signal control alongside every
   future graduation screen) for a FUTURE operator decision — not something
   to apply unilaterally, and NOT retroactive to S1/S2 (outside D30's
   scope).
5. **D34 evidence, the number that actually matters for the book-identity
   fork:** S5_static correlates 0.55 with S1_static and 0.56 with S2_static
   — the new FRED/yfinance dependency bought a static tilt that is HALF
   SHARED with the tilt S1/S2 already carry. Some diversification, same
   risk family, not fully new breadth.
6. **Next step, design-reviewer's recommendation:** reject re-opening
   S5_carry's construction (no defect found by either audit; any change
   hits the spec's locked-dial list; a new trial would spend budget to
   relearn a known property of the asset class, not learn something new).
   S5_carry is closed as a timing signal. The narrow question left for the
   operator: does S5_static enter the Phase-3 candidate set as an
   explicitly-labeled static-tilt sleeve (0.55/0.56 correlation and the
   2022-inversion risk disclosed up front), or is it shelved as D34
   evidence only, with Phase 3 proceeding on S1+S2 alone?

Next: log this characterization as a decision row, then present to the
operator — three items (the corrected finding + D29 risk manifestation,
the D34 disposition question, and the stale-lockbox rebuild that needs
operator action) per the §7 pause bar design-reviewer confirmed is met.

**Committed** (4964acd): `run_exp004.py`, `results.json`,
`specs/EXP-004-tier2-carry-ic-screen-v3.md`, decisions D36-D39, this
journal entry, both audit_log.jsonl entries. Fast+data suite green (84) at
commit time. Lockbox rebuild deliberately NOT committed (still holds the
buggy payload; untracked in git either way).

**Operator round 4 — D40:** presented the D34 disposition question
(shelve S5_static as evidence-only vs. include it in the Phase-3 candidate
set, explicitly labeled). Operator chose **include**, explicitly labeled as
a static risk-premium tilt with the 0.55/0.56 S1/S2-static correlation and
the 2022-inversion risk (D29) carried into every downstream report on it —
not shelved. Rationale: correlation ~0.55/0.56 is real, partial
diversification (not a near-duplicate of S1/S2's own tilts), and Phase
3-5's own gates (PBO/CSCV/DSR/cost-regime-stability) are the right place to
judge a disclosed, honestly-labeled static tilt on its actual portfolio
merit, rather than pre-filtering it out at Phase 2 on the strength of the
D29 concern alone.

**Phase-3 candidate set is now: S1 (momentum), S2 (TS trend), S5_static
(carry, static-tilt-labeled).** Still requested, separately: operator
action to clear the stale `data/lockbox_carry/` + `OPERATOR_TOKEN_CARRY.txt`
pair so a corrected lockbox rebuild can happen — not yet done.
