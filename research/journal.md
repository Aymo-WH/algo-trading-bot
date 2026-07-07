# Gordian v2 — Research Journal (append-only)

## Status — 2026-07-07
- **Phase:** R (Research & Design) — in progress.
- **Passed/failed:** nothing run yet; no strategy metrics exist.
- **Current best validation metrics:** n/a (no alpha work permitted until Phase 0 referee is green).
- **Open risks:** effective breadth of an ETF cross-section may be far below nominal N (to be quantified in R3); survivorship in a hand-picked "alive today" ETF universe.
- **Next steps:** finish R1–R4, write `research/design.md`, pre-register, present for operator approval.
- **Pod cost note:** Phase R is cheap (reads + small yfinance downloads + 3 subagents).

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
