# Critical Project Review — FX Technicals Dashboard

**Scope:** full codebase review (all `core/`, `pages/`, `visualization/`, `utils/`, `data/`, config, docs) plus an end-to-end run of the analysis pipeline on the bundled data.
**Audience:** a trader of FX spot and FX options who wants this to be practically usable for real decisions.

---

## 1. Verdict

This is a well-organized prototype with genuinely good instincts — the ADX "style selector" framing, the liquidity-consumption model for S/R, the IV-vs-RV premium concept, and the multi-timeframe structure are all the right ideas. But **in its current state it cannot be trusted for live trading decisions**, for three structural reasons:

1. **The options/vol layer runs on fabricated data.** The implied vol file is a seeded random walk (`data/generate_sample_iv.py`, `np.random.seed(42)`), the shipped CSV covers only Jan–May **2023**, and the app forward-fills it onto current prices and issues recommendations from it. Verified end-to-end: the pipeline produced *"SELL VOL BIAS: IV trading above RV — selling strategies may be favorable"* by comparing May-2023 synthetic IV against realized vol computed from Jan-2025 prices. No banner anywhere marks this as sample data.
2. **The data layer fails silently into wrong data.** When yfinance fails, AUDUSD and USDCNH silently display **EURUSD's** CSV (`core/data_fetcher.py:16-17`); the CSVs end 2025-01-27 with no staleness warning; if hourly data is unavailable the "4h" timeframe silently becomes a copy of daily (`data_fetcher.py:138`), double-counting daily in the MTF alignment score.
3. **No signal has ever been validated.** There is no backtest in the repo, yet the UI presents parameters as *"optimal from backtest"* (`config.py:179`, `pages/1_Settings.py`). There are zero tests of any kind. The dashboard is an opinion generator whose opinions have unknown (unknowable, currently) edge.

None of this is fatal — the fix path is clear and most of the analytical scaffolding is reusable. The roadmap in §6 orders the work by what makes it *practically implementable* fastest.

---

## 2. What's genuinely good

- **Clean separation of concerns**: `core/` (analytics) vs `visualization/` vs `pages/` is the right shape; most analytics are plain pandas functions that could be lifted into a headless pipeline or backtester unchanged.
- **The ADX regime framework** (`core/regime.py`) — treating ADX slope as more informative than level, and mapping regime → S/R behavior ("wall" vs "liquidity target") — is a professional framing most retail dashboards lack.
- **The S/R liquidity model** (`core/sr_analysis.py`) — touch quality, role flips, liquidity depletion phases, approach compression — is conceptually strong and the scoring is explicit and tunable.
- **Configurable strategy settings** with presets (`config.py`) is exactly the structure a backtester needs — it just doesn't have one yet.
- **The IV/RV premium concept** and term-structure display are the right primitives for an options trader — they just need real inputs.

---

## 3. Critical issues (trust & correctness)

These are the items that would actively burn a trader. Evidence cited as `file:line`; items marked ✅ were reproduced by running the code.

### 3.1 Data integrity

| # | Issue | Evidence |
|---|-------|----------|
| D1 ✅ | **Synthetic IV presented as real.** Seeded random walk; shipped CSV has 95 dates ending **2023-05-31**. Forward-filled onto live prices (`merge_iv_with_price_data`), then drives BUY/SELL VOL recommendations, IV-RV spread charts, term-structure "(INVERTED)" flags. No sample-data banner anywhere. | `data/generate_sample_iv.py:17`, `core/implied_vol.py:204-214`, `charts.py:1043-1216` |
| D2 ✅ | **Silent wrong-pair fallback.** AUDUSD=X and USDCNH=X map to `eurusd_daily.csv`. With yfinance down (or rate-limited), the trader sees EURUSD prices labeled AUDUSD. | `core/data_fetcher.py:11-18` |
| D3 ✅ | **No data provenance or staleness signal.** Live vs CSV fallback is invisible; bundled CSVs end 2025-01-27; exceptions during fetch are swallowed (`except: pass`). | `data_fetcher.py:88-92` |
| D4 ✅ | **Silent 4h→daily substitution** double-counts daily in MTF alignment when hourly fetch fails. | `data_fetcher.py:138` |
| D5 | **yfinance as sole live source** is inadequate for FX: indicative/delayed pricing, no bid/ask, **zero volume** — which means `detect_volume_weighted_levels` always returns `[]` (`support_resistance.py:141-143`). The Volume Profile feature advertised in the README is structurally dead code. | `README.md:39`, `support_resistance.py:141` |
| D6 ✅ | **NaN→0.0 coercion poisons signals on short histories.** With <200 bars (e.g., the 148-row CSVs, or ~30 weekly bars), SMA200/EMA200 become 0.0 via `_safe_float`, so "price above X" comparisons run against zero. Weekly ADX is computed on ~30 bars and treated as meaningful. | `indicators.py:121-125` |

### 3.2 Analytical correctness

| # | Issue | Evidence |
|---|-------|----------|
| A1 ✅ | **Trailing-ATR "exit strategy" is conflated with a pre-entry stop.** `apply_exit_strategy` replaces the S/R-invalidation stop with `entry − 3.75×ATR` on a *not-yet-entered* setup. Verified: risk 68.8 → 136.4 pips, R:R 0.62 → 0.31, setup invalidated. Trailing stops are position-management state (the repo even has `TrailingStopState` for this — unused by the app); they don't belong in setup R:R. | `trade_setup.py:417-495` |
| A2 | **"Net Bias" sums incommensurable signals.** RSI-oversold (mean reversion) and MACD-cross (momentum) count as equal votes, while the ADX section of the same page explains that RSI means opposite things by regime. The headline metric contradicts the app's own framework. | `main.py:459-466`, `signals.py` |
| A3 | **Market Health Index is semantically wrong.** `signal_agreement` is direction-agnostic (uniformly bearish market → 10/10 "health"); high ADX always scores as "healthy"; high vol and squeezes are penalized — precisely the conditions an options trader wants. Arbitrary 0.30/0.25/0.25/0.20 weights. | `summary_aggregator.py:373-430` |
| A4 | **"Vol at TA Signal Events" chart is a placeholder presented as analysis** — it plots *today's* signals at *today's date* ("simulated historical events" per its own comment). There is no signal history anywhere. | `main.py:709-720` |
| A5 | **Untouched round numbers become S/R zones.** 11 round-number levels are always injected with age 0; DBSCAN `min_samples=1` promotes each to a standalone zone whether or not price ever reacted there. | `support_resistance.py:211-238, 266` |
| A6 | **ATR "annualized vol" overstates.** `(ATR/price)×√252×100` treats average daily *range* as daily σ — systematically ~1.5–1.7× close-close RV. Displayed alongside true RV without a caveat. | `volatility.py:131-137` |
| A7 | **Fabricated neutrality on missing data.** IV percentile and RV percentile default to 50.0 with no "insufficient data" flag. | `implied_vol.py:262-263`, `volatility.py:144` |
| A8 | **Weekly bars are Sunday-ending** (`resample('W')`), off the FX convention of the Friday 5pm NY close; 4h resample has no session-aligned offset, so bars won't match a broker's 4h candles. | `data_fetcher.py:96,128,160` |
| A9 | **Partial-bar arithmetic.** 1D/1W/1M % changes use the last daily bar, which intraday is an incomplete session. | `summary_aggregator.py:141-146` |
| A10 | **Pip formatting broken on the Summary page** — all prices formatted `{:.5f}`, so USDJPY shows `149.50000`. Same hardcoded `:.5f` in chart annotations. The S/R page handles this correctly — inconsistent. | `3_Summary.py:231,496`, `charts.py:121,139,791,809` |
| A11 | **IV column-name schism**: `create_iv_chart` expects `"IV 1M"`, all other chart functions expect `"iv_1M"`; mismatches silently render blank panels. | `charts.py:963` vs `1046,1264,1381,1483` |

### 3.3 Product coherence

| # | Issue | Evidence |
|---|-------|----------|
| P1 | **The Settings page does nothing on the Summary page.** `generate_market_summary()` is called with no settings → defaults; the user's strategy config silently doesn't apply to the multi-pair view. | `3_Summary.py:50`, `summary_aggregator.py:490-491` |
| P2 | **Settings don't persist** — session state only; lost on refresh. Saving also silently resets any field not exposed in the form to its dataclass default. | `1_Settings.py:270-301` |
| P3 | **Threshold drift**: "near S/R" is 50 pips on the page, 30 pips in alerts; "valid setup" R:R is 1.5 in two places and 2.0 in a third. | `3_Summary.py:397,469`, `summary_aggregator.py:460-466,525` |
| P4 | **Two conflicting directional verdicts** shown side by side: grid "Bias" from signal counts vs setup bias from regime/MTF — a pair can be NEUTRAL in the grid and LONG on the opportunity board, unexplained. | `summary_aggregator.py:276-281` vs `311-318` |
| P5 | **"Optimal from backtest" claims with no backtest in the repo** (trailing multiplier 3.75, window 6). Unverifiable provenance presented as validation. | `config.py:179`, `1_Settings.py:142` |

### 3.4 Engineering

- **Zero tests, no CI, unpinned deps.** ✅ `ta==0.11.0` no longer builds on modern setuptools (`AttributeError: install_layout`) — a fresh `pip install -r requirements.txt` fails today. The 8 indicators used are all ~5 lines of pandas each; hand-rolling them removes the fragile dependency entirely.
- **Serial per-pair pipeline** on the Summary page — full fetch + indicators + S/R + vol per pair in a loop (`summary_aggregator.py:496-499`); `ThreadPoolExecutor` imported but never used.
- **~600 lines of copy-paste** between `create_main_chart` and `create_main_chart_with_vol` (`charts.py:14-345` vs `671-931`) — bugs like the `:.5f` pip formatting must be fixed twice.
- **Two parallel S/R systems** (`support_resistance.py` vs `sr_analysis.py`) with two scoring configs (`CONFLUENCE_WEIGHTS` vs `SR_SCORING_CONFIG`) and duplicated timeframe weights (`sr_analysis.py:179-184` vs `config.py:80-81`).
- **Orphaned code**: the whole regime-change machinery (`RegimeChangeState`, `detect_regime_history`, etc., `regime.py:715-933`) is wired into `main.py` but the planned regime-change/S/R-break alerts in `SUMMARY_PAGE_PLAN.md` §10 were never built.
- `apply_filters` passes user search text to `str.contains(regex=True)` — `"("` throws. (`summary_aggregator.py:619`)

---

## 4. What an FX **spot** trader is missing

In rough order of practical impact:

1. **Position sizing.** The bridge from "setup" to "order": account currency, risk %, pip value with proper conversion (USD-quote vs USD-base vs cross), → units/lots. Risk in pips exists; risk in money does not.
2. **A backtest.** The `StrategySettings` structure is already backtest-shaped. Without an equity curve, hit rate, and per-regime breakdown, every signal displayed is an untested hypothesis.
3. **Alerts with delivery.** A dashboard you must stare at isn't a tool for a 24h market. Level-approach, signal-fired, squeeze-triggered alerts pushed via Telegram/email/webhook from a scheduled headless job (the analytics core is already UI-independent enough to support this).
4. **Signal history on the chart** — where past signals fired and what happened next (this also replaces the placeholder "Vol at TA events" chart with the real thing).
5. **Economic calendar integration.** NFP/CPI/central-bank days dominate FX; signals and alerts should at minimum be flagged when a top-tier event is inside the trade horizon.
6. **Session awareness** — Tokyo/London/NY session markers, Friday-close weekly bars, broker-aligned 4h bars, and awareness that the daily "1D change" bar may be in progress.
7. **Carry** — overnight swap/forward points on any held setup (first-order for USDJPY-type pairs).
8. **A trade journal** — log the setup taken (settings snapshot, levels, reasoning) and track outcomes; turns the tool into a feedback loop.

## 5. What an FX **options** trader is missing

The current "options" support is one synthetic ATM-IV line per tenor. The minimum viable options layer for FX:

1. **Real vol data with smile**: ATM, 25Δ/10Δ **risk reversals and butterflies** per tenor. RR/BF are how FX vol is quoted and where the positioning information lives; ATM-only misses skew entirely.
2. **Vol cones**: realized vol over 1W/1M/3M/6M windows with historical min/25/50/75/max percentile cones, overlaid with current IV per tenor. Natural extension of the RV math already present, and the single most useful IV-vs-RV visualization.
3. **Breakeven analysis**: IV-implied daily breakeven (`IV/√252 × spot`) vs. actual daily moves and ATR — a running "did owning gamma pay this month?" scoreboard. Cheap to build from existing data.
4. **Garman-Kohlhagen pricer + greeks** (~40 lines): given strike/tenor/notional, show delta, gamma, vega, theta and a spot-ladder P&L. Needs deposit rates or forward points per pair.
5. **Forward points / carry** — needed both for the pricer and for reasoning about risk-reversal positioning.
6. **Event-vol extraction**: with a calendar in place, back out the implied event move from the term structure (e.g., 1W vs 1M around a central-bank date) — replaces the current generic "inverted term structure suggests event risk" caution.
7. **FX-native strategy language.** "Covered calls, cash-secured puts" (`volatility.py:404`) is equity vocabulary; recommendations should name FX structures with concrete strikes/tenors and the IV used, e.g. "1M 25Δ strangle sale at 7.2 vol vs 5.9 RV".

**Where to get real data** (in ascending order of cost/effort):
- **Spot/candles**: OANDA v20 REST (free practice account, real bid/ask, well-supported Python client) or Dukascopy historical; both strictly dominate yfinance for FX.
- **Options/IV**: hardest part for free. Realistic options: (a) broker with FX options — IBKR's API exposes CME FX options chains with IVs/greeks; (b) CME end-of-day settlement data for FX options; (c) a manual/CSV **adapter interface** so a trader with Bloomberg/Refinitiv/broker access can drop in a daily vol-surface export — the `load_iv_data` seam already exists, it just needs a documented schema including RR/BF columns and a provenance/staleness banner.

---

## 6. Recommended roadmap

Ordered so each phase makes the tool honestly usable at that level before adding surface area.

### Phase 0 — Stop the bleeding (days)
1. **Banner every synthetic/stale/fallback data path.** "DEMO DATA (synthetic, 2023)" on all IV panels; "CSV fallback, last bar 2025-01-27" on prices; hard-fail rather than substitute EURUSD for AUDUSD.
2. Remove the trailing-ATR mutation of pre-entry setup stops (keep it as a "management preview" display only).
3. Fix `_safe_float` NaN→0 (propagate `None`, gate each signal on data sufficiency; show "insufficient history" instead of fake 50th percentiles).
4. Fix pip-decimal formatting on the Summary page and chart annotations; unify the 30/50-pip and 1.5/2.0 R:R thresholds into config.
5. Wire `StrategySettings` into the Summary page; persist settings to a JSON file.
6. Pin dependencies; replace `ta` with ~80 lines of pandas (it currently doesn't even install); add a smoke test that imports every module and runs the pipeline on a bundled fixture.

### Phase 1 — Trustworthy data layer (1–2 weeks)
7. **OANDA (or equivalent) adapter** behind a `DataSource` interface; yfinance demoted to fallback with an explicit provenance badge and timestamp shown in the UI.
8. **Local store** (SQLite or parquet) with a scheduled ingest job — enables 5+ years of history (the 252-day percentiles currently run on barely 2 years), removes fetch latency from page loads, and gives the backtester a stable substrate.
9. Friday-close weekly bars (`W-FRI`), session-aligned 4h bars, exclude the in-progress daily bar from "1D change".
10. Documented IV import schema (date, tenor, ATM, 25ΔRR, 25ΔBF, 10ΔRR, 10ΔBF) + staleness banner; delete the pretense of live IV until real data flows.

### Phase 2 — Validation (1–2 weeks, highest value-per-effort in the repo)
11. **Vectorized backtester** over the existing `StrategySettings` grid: equity curve, hit rate, avg R, max DD, and per-ADX-regime breakdown. Replace every "optimal from backtest" claim with a linked, reproducible result.
12. **Signal history**: persist fired signals per bar; plot them on the chart; make the "vol at TA events" chart real (event-study: RV/IV behavior around past signals).
13. Unit tests for pip math, position sizing, stop logic, S/R clustering (golden files).

### Phase 3 — Spot-trader completeness (parallelizable)
14. Position sizing panel (account ccy, risk % → units, money-at-risk on every setup card).
15. Headless alert engine (reuse `core/` pipeline in a cron job; Telegram/email webhooks; alert state persisted, deduped, with ack).
16. Economic calendar (e.g., a weekly-refreshed ICS/API import) gating signals and stamping charts.
17. Trade journal (SQLite; snapshot settings + setup + outcome).

### Phase 4 — Options-trader layer
18. Vol cones (RV percentile bands vs IV by tenor).
19. RR/BF smile display + history once real IV flows.
20. GK pricer + greeks + spot-ladder P&L for a hypothesized position.
21. Breakeven/gamma scoreboard; event-vol extraction between tenors.
22. Rewrite strategy suggestions to name concrete FX structures with strikes/tenors/vols, gated on data provenance (never on demo data).

### Phase 5 — Consolidation
23. Merge the two S/R systems into one scored implementation; single chart builder (kill the ~600-line duplication); one `AnalysisResult` pipeline per pair, cached by `(symbol, data_version, settings_hash)`, parallelized across pairs.

---

## 7. Quick reference: top 10 fixes by impact/effort

| Rank | Fix | Impact | Effort |
|------|-----|--------|--------|
| 1 | Synthetic-IV banners + kill silent pair fallback | Prevents catastrophically wrong conclusions | Hours |
| 2 | Real spot data adapter (OANDA) | Every number improves | Days |
| 3 | Backtester over existing settings | Converts opinions into measured edge | ~1 week |
| 4 | Un-conflate trailing stop from setup R:R | Setup validity currently nonsensical | Hours |
| 5 | NaN→0 fix + data-sufficiency gates | Removes a whole class of silent lies | Hours |
| 6 | Position sizing | Setup → order bridge | 1–2 days |
| 7 | Alert engine with delivery | Dashboard → tool for a 24h market | Days |
| 8 | Vol cones from existing RV math | Best options visual for least work | 1 day |
| 9 | Pin deps, drop `ta`, add smoke tests | Project currently doesn't install fresh | Hours |
| 10 | Persist settings + wire into Summary | Settings page currently half-illusory | Hours |
