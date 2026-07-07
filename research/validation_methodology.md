# Validation Referee Methodology Brief (Phase R, 2026-07-07)

Produced by a web-research subagent (R2a) from primary sources. This is the spec basis
for the Phase 0 referee. Setting: cross-sectional market-neutral, 40–70 ETFs, daily
data, weekly rebalance, T ≈ 3,800–5,500 obs.

## 1. CSCV / PBO — Bailey, Borwein, López de Prado, Zhu (2017), J. Computational Finance
Build M (T×N): one column of P&L per trial/config, rows synchronous. Partition rows into an even number S of disjoint, order-preserving blocks. Form all C(S, S/2) combinations; for each combination c: train J = union of S/2 blocks, test J̄ = complement. Rank all N configs by the metric in J; pick IS-best n*. ω̄c = OOS rank of n* / (N+1); logit λc = ln(ω̄c/(1−ω̄c)). **PBO = fraction of λc ≤ 0.**
Params: **S = 16** → C(16,8) = 12,870 combinations (paper text misprints "12,780"). Metric: Sharpe. For 15–20 yr daily, S=16 → ~1–1.25 yr blocks. Pitfalls: N too small → coarse ranks (want dozens+ of logged trials incl. discards); highly correlated configs → narrow logit dispersion, less informative. PBO < 0.2 accept threshold is convention, not primary-source (our contract: < 0.5 pass, ≤ 0.2 target).

## 2. Deflated Sharpe Ratio — Bailey & López de Prado (2014), JPM 40(5)
E[max SR under null], N independent trials (Gumbel/Euler–Mascheroni, γ ≈ 0.5772156649):
`SR₀ = sqrt(V[{SR̂ₙ}]) · [(1−γ)·Z⁻¹(1−1/N) + γ·Z⁻¹(1−1/(N·e))]`
`DSR = Z[(SR̂ − SR₀)·√(T−1) / sqrt(1 − γ̂₃·SR̂ + ((γ̂₄−1)/4)·SR̂²)]` with γ̂₃ = skew, γ̂₄ = raw kurtosis, per-period SR̂ (non-annualized). Accept if DSR > 0.95.
**Effective independent trials:** ρ̄ = mean off-diagonal correlation of trial return series; `N̂ = ρ̄ + (1−ρ̄)·M`. Caveat: if M > T the correlation matrix is ill-conditioned — reduce dimension first. V[{SR̂ₙ}] is the empirical variance of SRs across ALL logged trials — the ledger is a statistical input.

## 3. CPCV & purging/embargo — López de Prado (2018), AFML Ch. 7 & 12
**Purging:** drop training obs whose label interval [t, t+h] overlaps any test-label interval. **Embargo:** additionally drop training obs immediately *after* each test block (serial-correlation leakage); AFML heuristic ≈ 0.01·T; principled minimum = label horizon. **CPCV:** N sequential groups, all C(N,k) test combinations (canonical N=6, k=2 → 15 splits, φ=5 backtest paths) → distribution of OOS Sharpes vs walk-forward's single path.
**Our setting:** purge = 5 trading days (weekly labels), embargo ≥ 5 days (we adopt 10 = 2× horizon, conservative; sources disagree between 0.01T ≈ 40–55d and label-horizon rule — we log both options, default 10d). CPCV N=8, k=2 for ~5,500 obs. Walk-forward retained as secondary realism check.

## 4. Harvey, Liu, Zhu (2016), RFS 29(1)
New factors must clear **t > 3.0** given the mined factor population. Haircuts: Bonferroni, Holm (FWER), BHY (FDR under dependence). Practical: impose t ≥ 3.0 on the final candidate's IC/mean-return t-stat, plus BHY across the family of logged trial p-values. Note LdP criticizes fixed hurdles in favor of DSR — run both (they answer different questions).

## 5. MinBTL — Bailey et al. (2014), Notices of the AMS 61(5)
`MinBTL(years) < 2·ln(N) / E[max_N]²`. With 20+ yr the bound tolerates large N, but use N̂ (effective trials) and treat as sanity floor, not license.

## 6. Negative controls / leak canaries (practitioner synthesis; conventions, flagged as such)
- **Within-date label shuffle (primary for cross-sectional):** per rebalance date, permute forward returns across assets within that date. Preserves each date's cross-sectional distribution, date fixed effects, vol regime; destroys signal–asset alignment. Pass: |IC t| < 2, DSR non-significant.
- **Row-block time shuffle (secondary):** permute whole date-rows (blocks) across time — preserves cross-asset correlation exactly, destroys temporal predictability. Run both.
- **Feature time-shift:** extra lag (+26w) → IC must die; forward shift (peek) → IC must jump. Baseline ≈ peeked ⇒ leak.
- **Synthetic null panel:** random-walk returns with empirical covariance (or stationary bootstrap of date-rows); full pipeline incl. CSCV must report PBO ≈ 0.5 and null DSR. If it finds edge, the harness itself is broken.
- **Planted signal:** inject known alpha into the synthetic panel; the pipeline must recover it at the expected IC (tests power, not just size).

## 7. Cross-sectional IC testing (Grinold & Kahn; Newey–West 1987)
Per rebalance date: IC_t = Spearman(signal, forward 1-wk return) across names. Report mean IC and t = mean(IC)/SE_NW; NW lags ≥ label overlap (non-overlapping weekly: 1–2; overlapping daily-sampled: ≥5). Magnitudes: 0.02–0.05 mean rank IC realistic at weekly horizon; 0.05–0.10 strong; **> 0.15 = leakage red flag**. IC decay: horizons k = 1…8 weeks; half-life must exceed rebalance interval net of costs.

**Flagged disagreements:** (a) CSCV paper "12,780" vs true 12,870; (b) embargo 0.01T vs label-horizon; (c) HLZ fixed t-hurdle vs DSR; (d) PBO 0.2 / DSR 0.95 thresholds are conventions.

Primary sources: Bailey et al. PBO (J. Comp. Finance 2017 / SSRN 2326253); Bailey & LdP Deflated Sharpe (JPM 2014 / SSRN 2460551); Bailey et al. Pseudo-Mathematics (AMS Notices 2014); Harvey-Liu-Zhu (RFS 2016); LdP AFML (2018) Ch. 7, 12; Newey–West (1987); Grinold & Kahn APM.
