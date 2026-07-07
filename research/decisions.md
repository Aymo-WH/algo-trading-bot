# Gordian v2 — Design Decisions (append-only)

| # | Date | Decision | Rationale | Status |
|---|------|----------|-----------|--------|
| D1 | 2026-07-07 | Work on branch `research/gordian-v2`; keep v1 code intact for reuse | v1 engineering (FFD, PIT-PCA, PPO harness, PBO gate) is salvage material per mission §0 | active |
| D2 | 2026-07-07 | Phase R statistics (correlation structure, effective breadth) computed on pre-holdout data only, even before the holdout boundary is formally frozen | Conservative reading of the Prime Directive: design choices must not be informed by any test-period structure | active |
