# Gordian v2 — Design Decisions (append-only)

| # | Date | Decision | Rationale | Status |
|---|------|----------|-----------|--------|
| D1 | 2026-07-07 | Work on branch `research/gordian-v2`; keep v1 code intact for reuse | v1 engineering (FFD, PIT-PCA, PPO harness, PBO gate) is salvage material per mission §0 | active |
| D2 | 2026-07-07 | Phase R statistics (correlation structure, effective breadth) computed on pre-holdout data only, even before the holdout boundary is formally frozen | Conservative reading of the Prime Directive: design choices must not be informed by any test-period structure | active |
| D3 | 2026-07-07 | CONFIRM market-neutral cross-sectional chassis; REFINE parts (see design.md §1) | Measured residual breadth 10–20 (not 50–100); evidence re-ranks signals; alternatives A1–A3 considered and rejected with rationale | proposed |
| D4 | 2026-07-07 | Drop short-term reversal + lead-lag from signal library | Contradicted at index level after costs (Dai/Novy-Marx; Da et al.) | proposed |
| D5 | 2026-07-07 | Add time-series trend as first-class signal inside the neutral book | TS > XS momentum on macro universes (Baz et al. 2015; Babu et al.) | proposed |
| D6 | 2026-07-07 | Carry deferred to Tier 2 (new data dependency: FRED/dividends) | Strongest evidence but requires non-price data; Tier-1 infra first | proposed |
| D7 | 2026-07-07 | ML demoted to earn-its-keep combiner (M0 linear baseline mandatory) | Direct guard against v1 failure mode (ML as oracle) | proposed |
| D8 | 2026-07-07 | Holdout = 2022-01-01 → 2026-06-30; train/val ends 2021-12-31 | ~17% of sample, regime-diverse (2022 bear, 2023–24 rally); matches breadth-stat cutoff already used | proposed |
| D9 | 2026-07-07 | Universe = 69 ETFs, $10M liquidity floor, PIT entry rule, no return-based membership | Measured in universe_summary.csv; survivorship limitation documented, not hidden | proposed |
