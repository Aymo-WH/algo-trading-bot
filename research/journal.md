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
