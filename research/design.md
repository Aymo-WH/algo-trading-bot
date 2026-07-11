# Gordian v2 — Design Proposal (Phase R deliverable)

**Date:** 2026-07-07 · **Status:** PROPOSED — awaiting operator approval
**Verdict on the §4 default: CONFIRM the chassis, REFINE the parts.**
Evidence basis: `research/data_recon/` (measured), `research/v1_forensics.md`,
`research/evidence_etf_cross_section.md`, `research/validation_methodology.md`.

---

## 1. Architecture verdict (R4)

The market-neutral cross-sectional default survives pressure-testing, with three
material corrections:

1. **Breadth honesty.** §4's "50–100× breadth" is optimistic. Measured on our own
   pre-holdout panel: raw effective N ≈ 3 (PC1 = 53–58% of variance); ≈ 16–17 after
   removing PC1; ≈ 9–10 cross-sectionally demeaned. Literature (ReSolve PCA work; Baz
   et al. 2015) credits 5–10 independent macro bets. **Design expectation: ~10–20
   effective bets**, i.e. breadth multiplier ~10× over v1, not 50–100×. Consequence:
   realistic net Sharpe for a multi-signal book is **0.3–0.8**; the contract's 0.5 is
   ambitious-but-defensible; >1.2 in backtest triggers a leakage audit (tightens
   mission §3.3f's 1.5 bound as an *audit* trigger, keeping 1.5 as the *presumed-bug*
   bound).
2. **Signal list re-ranked by evidence.** Short-term reversal and lead-lag are
   **dropped** (contradicted at index level after costs — stock-level microstructure
   effects). Time-series trend is **added** as a first-class signal (stronger than
   cross-sectional momentum on macro universes; Baz et al., Babu et al.). Carry — the
   strongest-evidence family — is **Tier 2** (needs non-price data: FRED yields,
   dividends, futures-slope proxies; added only after Tier-1 infrastructure is proven).
   v1's microstructure features are demoted to optional regime conditioners (Tier 3,
   low prior — "regime-conditioning" has weak out-of-sample evidence).
3. **ML demoted from oracle to earn-its-keep combiner.** The baseline is a transparent
   z-score composite. XGBoost is admitted only if it beats that baseline on CPCV by a
   pre-registered margin; PPO sizing only after that, same rule. This directly guards
   against the v1 failure mode.

**Alternatives considered and rejected (logged per §3.7):**
- **A1 — single-stock universe (true breadth ~100+):** rejected for this mission.
  Point-in-time index membership (survivorship-free) is not available via our data
  stack; the bias we'd introduce is worse than the ETF universe's residual closure
  bias, and data/compute scale ~10×. Revisit only with operator approval + a proper
  PIT membership source.
- **A2 — pure time-series trend book (managed-futures style):** rejected as the sole
  design — heavily mined, flat post-2010, and abandons the cross-sectional hedge
  against common-factor unpredictability. Adopted instead as one signal inside the
  neutral book.
- **A3 — monetize v1's vol/regime predictability via vol instruments:** rejected —
  VIX-complex ETPs have short history, extreme costs and path decay; monetizing a vol
  forecast without directional edge needs derivatives infrastructure beyond scope.

## 2. Universe (measured, not assumed)

From the 78-ticker audit (`research/data_recon/universe_summary.csv`): keep tickers
with ≥ $10M median daily dollar volume (3y) and real history → **69 names** across
sectors (20), broad US (5), styles (7), countries/regions (17), bonds (10),
commodities (5), REITs (3), FX (1: UUP), spanning 1993–2026.
**Excluded:** TUR, FXE, EWM, EZA, DBA, FXY, EWS (thin), RSX (dead — no data).
- **Point-in-time entry:** an ETF enters the panel at its first date with ≥ 252 trading
  days of history; the panel starts at the first weekly rebalance with **≥ 20 eligible
  names** (~2000). No return-based inclusion or removal — membership is fixed by this
  rule alone.
- **Survivorship (honest limitation):** yfinance cannot see dead ETFs (RSX probe:
  1 junk row). Our universe is category mega-ETFs whose closure rate is very low, but
  the residual bias is real, mainly in the country tail, and will be stated in every
  report. No claim of a survivorship-free backtest will be made.

## 3. Data & quarantine boundaries (pre-registered)

- Daily adjusted OHLCV via yfinance, panel-aligned on NYSE calendar, ~2000-01 →
  2026-06-30. Adjusted closes for returns; unadjusted volume for liquidity. Known
  caveat: retroactive dividend adjustment is the standard, accepted compromise.
- **Split:** train/validation = panel start → **2021-12-31**; **quarantined holdout =
  2022-01-01 → 2026-06-30** (~4.5y, ~17% of sample; contains the 2022 rate-shock bear,
  the 2023–24 rally, and 2025–26 — regime-diverse). Purge 5 trading days + embargo 10
  days at the boundary.
- Phase R breadth statistics already respected this boundary (computed ≤ 2021-12-31).
- **Lockbox (§3.3h):** the data factory writes holdout-period returns to an encrypted
  store; development code has no key. `final_eval.py` is the only reader, requires an
  operator-provided token, logs its invocation, and may run once.

## 4. Prediction target

**5-trading-day forward return, cross-sectionally demeaned** (primary) with
beta-residualized variant as a pre-registered robustness check. We predict *ranking*,
never absolute direction. Weekly rebalance on Wednesdays (avoids Monday/Friday
calendar artifacts); daily rebalance is out of scope unless a pre-registered test
justifies its extra cost.

## 5. Signal library — Tier 1 (price-based), each with rationale + falsification

Graduation rule (identical for every signal, pre-registered): on train/validation
walk-forward, **mean weekly rank IC ≥ 0.01 AND Newey-West t ≥ 2.0 AND positive-IC in
≥ 60% of calendar years**. A signal failing any leg is dropped — no post-hoc rescue by
re-parameterization without a new logged trial.

| ID | Signal | Economic rationale | Construction sketch |
|----|--------|--------------------|---------------------|
| S1 | Cross-sectional momentum (12-1, 6m, 3m blend) | Underreaction / slow-moving capital across asset classes; AMP 2013 | trailing return skip-month, z-scored per date |
| S2 | Time-series trend, neutralized | Hedger flows + gradual information diffusion; TS > XS on macro universes (Baz 2015) | sign/strength of 12m & 3m trend per asset, used as ranking input; book stays neutral |
| S3 | Low-beta / defensive (BAB) | Leverage constraints (Frazzini-Pedersen 2014) | rank by trailing 252d beta vs equal-weight market factor |
| S4 | Calendar seasonality | Persistent seasonal demand (Keloharju et al. 2016) | same-calendar-month mean relative return, ≥ 5y history required |

**Tier 2 (pre-registered as future work, needs new data):** S5 carry — bond carry from
FRED yield curves, equity carry from trailing dividend yield, commodity carry from
roll-proxy. Strongest published evidence (KMPV 2018); added only after Tier-1
infrastructure is green and with operator awareness of the new dependency.
**Tier 3 (low prior, optional):** v1 microstructure features (VPIN, Amihud, SADF) as
regime conditioners — only if Tiers 1–2 produce a live book to condition.

## 6. Model — combiner, not oracle

- **M0 baseline (mandatory first):** equal-weight z-score composite of surviving
  signals. Transparent, near-unoverfittable, and it is the bar every ML model must beat.
- **M1 (optional):** XGBoost cross-sectional ranker on pooled panel (purged CPCV
  training) — admitted only if CPCV-median net Sharpe ≥ M0 + 0.1 with PBO no worse.
- **M2 (optional, last):** PPO sizing overlay (v1 harness, rebuilt gym) — same
  admission rule vs M1. Fixed defects from forensics (short-capable action space, live
  risk-penalty terms, `np_random` seeding) are prerequisites.

## 7. Portfolio construction & risk

Dollar-neutral AND ex-ante beta-neutral (252d rolling betas); weights ∝ clipped
composite z (±2.5); gross 200% (100/100); position cap 10% of gross; category cap 40%;
**vol-targeted at 10% annualized** (leverage cap 3×); no-trade band (skip trades
< 0.5% NAV) for turnover control; drawdown brake: halve gross at 15% peak-to-trough on
validation equity. Momentum-crash defense (Daniel-Moskowitz): the beta-neutrality is
*dynamic* (betas re-estimated weekly), not just dollar-neutral.

## 8. Costs

5 bps/side baseline (contract), stress 10 and 20 bps; edge must survive 10 bps.
Evidence check: blended honest estimate for this universe is 5–10 bps/side, so the
10 bps stress is the realistic center, not the pessimistic tail — reported side by side.

## 9. Validation referee (Phase 0 spec — built before any alpha)

Per `research/validation_methodology.md`:
- **Engine:** vectorized panel backtester; every entry point auto-appends to
  `research/experiments.jsonl` (nothing runs uncounted).
- **CSCV/PBO:** S=16, C(16,8)=12,870, reusing the verified `pbo_validator.py` core;
  trial matrix = real config variants; runs on train/validation only.
- **DSR:** Bailey-LdP 2014 with skew/kurtosis correction and effective trial count
  N̂ = ρ̄ + (1−ρ̄)M; accept > 0.95. HLZ t ≥ 3.0 on final candidate IC as a parallel check.
- **CPCV:** N=8, k=2, purge 5d, embargo 10d; walk-forward (expanding, annual steps) as
  secondary. All scalers/PCA/statistics fit point-in-time per fold.
- **Canary suite (all must pass before any result is "validated"):** within-date label
  shuffle (primary), row-block time shuffle, feature time-shift (±26w), synthetic null
  panel with empirical covariance (pipeline must find nothing), planted-signal panel
  (pipeline must recover it). Wired as tests — a tripped canary blocks "validated" status.
- **Lockbox + token-gated `final_eval.py`** (§3.3h), one-shot, logged.
- **Runtime guardrails (`.claude/`):** PreToolUse hook blocking reads of the holdout
  store and writes to `validation/`, canaries, or frozen `specs/`; PostToolUse
  lint+fast-tests on strategy edits; read-only `leak-hunter` auditor subagent;
  `/preregister` and `/run-validation` skills. Unit tests prove purge/embargo and
  quarantine mechanics.

## 10. Trial budget, kill criteria, expectations (pre-registered)

- **Trial budget:** ≤ 250 logged configurations for Phases 2–5 (ablations and discards
  count). Exceeding it requires logged operator approval. DSR always uses the true count.
- **K1:** no Tier-1 signal graduates (§5 rule) → Tier-1 null; attempt Tier-2 carry; if
  that also fails → write the rigorous null (mission success path B).
- **K2:** combined book net Sharpe < 0.2 on walk-forward after the trial budget → null.
- **K3:** candidate PBO > 0.5 on CSCV → not eligible for holdout; iterate on
  train/validation or declare null. PBO ≤ 0.2 remains the hardening target.
- **K4:** any tripped canary halts new work until root-caused.
- **Audit triggers:** net Sharpe > 1.2 → mandatory leakage audit; > 1.5 → presumed bug
  (mission §3.3f); any single name/day dominating P&L → audit.
- **Expectation set ex-ante:** net Sharpe 0.3–0.8 if signals are real; meaningful
  probability of an honest null. Both are success.

## 11. Success contract

§2 of `FABLE_MISSION.md`, verbatim, numbers unchanged (operator confirmed 2026-07-07).
Final gate: the one-shot, token-gated holdout evaluation (Phase 6), run once,
reported honestly either way.
