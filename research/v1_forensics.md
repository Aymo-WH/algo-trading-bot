# Gordian v1 — Forensic Reusability Audit (Phase R, 2026-07-07)

Produced by a fresh-context read-only auditor subagent; verified against code, not
docstrings. Basis for the v2 reuse decisions in `research/design.md`.

## 1. PBO/CSCV (`src/core/pbo_validator.py`, `src/validate_pbo.py`)
**Core validator: correct.** Contiguous S=16 blocks via `np.array_split` (pbo_validator.py:44), full C(S,S/2)=12,870 combos (:48), IS-best by mean/std Sharpe proxy (:57-58), OOS rank of that same index `ranks[optimal_idx]/(N+1)` (:64-65), logit and PBO=`mean(logits<0)` (:68-73) — faithful Bailey/López de Prado. No off-by-one; column alignment preserved. No purging between blocks (paper doesn't require it, but 15-bar holding periods straddle block boundaries — mild dependence leak).
**Wrapper: weak.** (a) Trials are a nested confidence-threshold grid on ONE fixed model (validate_pbo.py:93) — highly correlated columns understate selection bias (honestly labeled a proxy, :19-25). (b) It runs on `data/test/` (:50) — the gate itself consumes the quarantined set; for v2, CSCV must run pre-test. (c) Single-ticker timeline, ragged columns truncated (:149-151).
**Verdict:** validator REUSE as-is; wrapper REBUILD (trial matrix = real hyperparameter/seed variants, run on train/val, per-date panel returns).

## 2. FFD (`src/data_factory.py:68-130`)
Fixed-width, causal: kernel `[w_0..w_K]` convolved so `x_t = Σ w_k·f[t-k]` — weights depend only on `d`, past values only. Verified the double-reversal + `np.convolve` flip cancels correctly. **One leak:** optimal `d` is chosen by ADF on the FULL series including test dates (:408-418). Reusable for a daily panel per-column with `d` selected on train-only data. **REUSE with fix.**

## 3. Point-in-time PCA (`src/data_factory.py:459-472`)
Genuinely train-only, not full-sample: `scaler.fit(df.loc[train_clean_idx, ...])` (:461), `pca.fit(scaled_train_tech)` (:470-471) where `train_clean_idx = index < split_date` (:457), transform applied to all rows with frozen weights (:472). It is a single static 80% split, NOT expanding/walk-forward — fine for one holdout, insufficient for rolling research. **REUSE pattern; upgrade to expanding refits for v2.**

## 4. Purge/embargo
- Train/test: 60 bars dropped from the START of test (data_factory.py:488-493). Correct side; the "maximum feature lookback" claim is wrong — SADF window=100 and dollar-bar threshold window=210 exceed 60 (harmless: feature lookback into train isn't leakage; TBM labels are computed per-CSV so they truncate rather than leak).
- XGB 3-way split: EMBARGO=15 (=label horizon) dropped from the END of train and val (train_agent.py:215-239) — correct side, correct size. **REUSE.**

## 5. Confirmed v1 defects
(a) **Long-only artifact:** action space is `Box(0.0, 1.0)` (trading_gym.py:135); `act = xgb_signal * bet_size` (:272) so a short signal (-1) can only SELL held shares (:292-306) — no short positions ever. telemetry.py:144-162 explicitly hard-maps short signals to 0.0. Confirmed.
(b) **Zeroed reward terms:** trading_gym.py:327-330 — `turnover_penalty_coef = 0.0`, `variance_penalty_coef = 0.0`, `cvar_penalty = 0.0`; reward is `daily_return*100` plus a +50% bonus on positive returns only (:332-334) — asymmetric, risk-seeking. Confirmed.
(c) **Train/serve mismatch (live_inference.py):** fatal. Live path fetches 1h TIME bars, never builds dollar bars (:69 vs data_factory:388); computes only the 4 microstructural features (:84) — none of Close_FFD/RET_*/VOL_20/ZSCORE_20 — so `scaler.transform` (:120) receives 4 features where 11 were fit (would raise ValueError today); feature ordering by column iteration (:97-98) not by `tech_cols`; PPO sizing absent (XGB-only, fixed 0.01 lot, :130); all orders proxied to BTC/USDT regardless of signal ticker (:146). **DISCARD.**

## 6. Data factory (`src/data_factory.py`)
Produces per-ticker CSVs: 730d of 1h bars → dynamic dollar bars (:133-204) → microstructure + momentum features → PCA_1..11 + Close/OHLCV/Optimal_PT/SL, split into `data/train|test`. Multi-ticker loop but each asset independent — no date alignment, no panel, per-ticker split dates differ (:452), and each run wipes prior data (:277-278). Look-ahead findings: FFD `d` selection (item 2); dollar-bar threshold `.bfill()` backfills future volume into the first ~209 bars (:158); scalers/labels/rolling windows otherwise clean (barrier optimizer is causal — fits on trailing window, assigns forward, optimize_barriers.py:118-146). Configs' `start_date`/`end_date`/`interval` keys are DEAD — `fetch_data` never reads them. Extending to a 50-ETF daily panel = REBUILD the orchestration (daily bars, common calendar, cross-sectional alignment) while reusing FFD/feature/PIT-fit functions.

## 7. Transaction costs
Per-side proportional fee on notional at trade time in the gym (trading_gym.py:279, :296), from config `transaction_fee_percent`: 0.0001 (1bp) in config_phase1.json, 0.001 (10bp) elsewhere. No spread, slippage, or market impact modeling; 1bp/side is optimistic even for liquid ETFs. Evaluation and PBO inherit it via the env. **FIX for v2:** add spread/slippage; costs must include shorting for market-neutral.

## 8. Seeding/reproducibility
`set_global_seed(42)` seeds random/numpy/torch (train_agent.py:54-61); PPO/vec_env/XGB all seed=42. **Defect:** the gym uses the module-level `random` (`random.choice`/`randint`, trading_gym.py:122, 172, 185) instead of Gymnasium's `self.np_random` — the `env.reset(seed=...)` API is a no-op for data/start selection, and forked SubprocVecEnv workers inherit IDENTICAL `random` state (same episode sequences across workers). Evaluation is deterministic (fixed start_steps, deterministic=True). **FIX:** route env randomness through `self.np_random`.

## 9. Other load-bearing pieces
- **Thread-cap pattern** (train_agent.py:10-12, env vars set before numpy/torch import) and `usable_cpu_count()` via `os.sched_getaffinity` (:63-75) — REUSE verbatim for any container training.
- Numba SADF (data_factory.py:17-65) and njit TBM labeler (train_agent.py:31-52) — fast, causal, reusable.
- `pca_feature_columns` numeric-sort util (core/utils.py:35-51) — small but prevents a real train/serve column-order bug.
- telemetry.py / meta_agent.py / evaluate_agents.py are v1 architecture-specific (single-asset, long-only, ROI-vs-B&H framing); evaluate_agents also evaluates directly on `data/test`.

## Verdict table

| Component | Verdict | Reason |
|---|---|---|
| core/pbo_validator.py | REUSE | Faithful CSCV; correct combos, ranks, logits |
| validate_pbo.py wrapper | REBUILD | Nested-threshold proxy trials; runs on quarantined test set |
| FFD (data_factory) | FIX | Causal fixed-width, but `d` selected on full sample incl. test |
| PIT PCA/scaler fit | REUSE | Truly train-only fit; upgrade to expanding windows |
| Purge/embargo (XGB split) | REUSE | Correct side, size = label horizon |
| Dollar bars | DISCARD | Irrelevant for daily panel; `.bfill()` look-ahead at start |
| trading_gym.py | REBUILD | Long-only by construction; dead penalty terms; broken seeding |
| Reward function | REBUILD | Zeroed risk terms + asymmetric +50% profit bonus |
| train_agent.py XGB path | REUSE | Sound purged temporal split, regularization, calibration |
| Thread-cap / CPU-affinity | REUSE | Correct container pattern, verbatim |
| live_inference.py | DISCARD | Feature-count mismatch (crashes), time-vs-dollar bars, BTC proxy orders |
| telemetry.py, meta_agent.py | DISCARD | v1 long-only architecture specific |
| evaluate_agents.py | REBUILD | Reasonable mechanics but single-asset, touches test set freely |
| Cost model | FIX | Per-side bps only; no spread/slippage/shorting costs |
| config/*.json | FIX | `start_date`/`end_date`/`interval` keys silently ignored |
