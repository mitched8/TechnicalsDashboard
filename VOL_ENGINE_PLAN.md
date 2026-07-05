# Vol Regime Engine — Implementation Plan

**Goal:** a validated, data-source-agnostic analysis engine that answers the FX options
trader's core question — *what regime are we in, is a regime shift underway, and what
does that imply for realized and implied vol over each option tenor?* — and proves its
own claims with event studies rather than lore.

**Portability constraint (per project owner):** development happens on a laptop with
poor data access; the engine moves to a work PC where real price + IV data is available.
Therefore:

- All analytics live in a standalone package `vol_engine/` — pure pandas/numpy,
  **zero Streamlit imports, zero `ta` imports** (the `ta` package no longer builds on
  modern toolchains). Moving to the work PC = copy the package + feed it OHLC.
- Input contract is a plain DataFrame: `DatetimeIndex`, columns `Open, High, Low, Close`
  (Volume optional/ignored). One loader (`data_io.py`) reads CSVs in that schema.
- IV is an *optional* input everywhere. Every IV-dependent output accepts a proxy or
  degrades gracefully. When real IV arrives at work, it plugs into the same seams.
- Synthetic data is used **only** for engine validation (planted regimes with known
  ground truth) and an explicitly-labeled demo mode. It is never silently substituted.

---

## 1. Conceptual frame

Regimes are modeled as **vol states**, not trend states, because an option monetizes the
distribution of returns, not its sign:

```
RANGE ──(width/ATR/RV compression)──> COMPRESSION ──(channel break)──> BREAK_ATTEMPT
  ^                                                                        │
  │<────────────── FAILED_BREAK (close back inside) ──────────────────────┤
  │                                                                        v
  └───────────(efficiency ratio decays)────────────── TREND <── BREAK_CONFIRMED
```

Each state carries an empirical claim about forward realized vol (RV bleeds in ranges,
is bimodal out of compression, jumps at breaks, peaks early then grinds down in mature
trends, spikes-then-crushes on failed breaks). **The engine's job is to measure those
claims per pair from history** — conditional forward-RV distributions, resolution
times, and expansion probabilities — so the playbook is calibrated, not asserted.

## 2. Modules

| Module | Contents |
|---|---|
| `vol_engine/data_io.py` | OHLC CSV loader/validator; input schema doc |
| `vol_engine/rv_estimators.py` | Log returns; close-to-close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang RV (rolling, annualized %); per-bar variance proxies; multi-window RV panel; RV term-structure ratios |
| `vol_engine/cones.py` | Vol cones: percentile bands of RV by window length + current RV markers; rolling RV percentile |
| `vol_engine/regime.py` | Causal feature set (Donchian channel + width percentile, ATR + ATR ratio, Kaufman efficiency ratio, NR7/inside-day counts, expansion bars, RV backwardation, composite compression score) and the state machine (RANGE / COMPRESSION / BREAK_ATTEMPT / TREND) with break **acceptance** and **failure** logic; emits a typed event list (compression start/end, break attempt/confirmed/failed, trend start/end) |
| `vol_engine/event_study.py` | The validation harness: conditional forward-RV distributions at multiple horizons vs unconditional baseline, Monte-Carlo p-values, expansion probability, event-time average paths (e.g. RV around breaks), time-to-resolution stats |
| `vol_engine/levels.py` | Causal S/R level tracker (swing levels confirmed with lag, touch counts, age, liquidity phase: fresh/tested/depleted) and break-event tagging — the causal analogue of the dashboard's liquidity model, enabling event studies conditioned on the broken level's test history |
| `vol_engine/har.py` | HAR-RV forecaster (daily/weekly/monthly RV components, OLS on log-RV), walk-forward out-of-sample evaluation vs naive trailing-RV baseline, and regime-augmented variant reporting incremental skill |
| `vol_engine/hmm.py` | Lightweight 2-state Gaussian HMM (numpy EM + smoothing) on returns as a statistical cross-check of the rule-based states |
| `vol_engine/breakeven.py` | Straddle-breakeven scoreboard: IV-implied daily breakeven vs realized \|moves\|, rolling gamma-P&L proxy and hit rate; accepts real IV series or a labeled trailing-RV proxy |
| `vol_engine/breadth.py` | Cross-pair breadth: fraction of pairs in compression / expanding / trending; average width percentile |
| `vol_engine/playbook.py` | Declarative state → stance mapping (own gamma / sell premium / directional / fade), FX-native structures, tenor guidance derived from measured time-to-resolution, caveats |
| `vol_engine/synthetic.py` | Regime-switching series generator with **planted ground-truth labels** — for tests and the clearly-labeled demo mode only |
| `tests/` | pytest suite (see §4) |
| `pages/4_Vol_Regime.py` + `visualization/vol_regime_charts.py` | Dashboard surface: state timeline on price, cone, RV panel, event-study tables/paths, HAR forecast, playbook card, HMM strip. Data-provenance banner always visible; insufficient-history warnings instead of fabricated defaults |
| `run_vol_analysis.py` | Headless CLI: CSV in → JSON/CSV report out (for the work PC, no UI needed) |

## 3. Method specifications

**RV estimators** (rolling window *n*, annualization 252, output in vol points %):
- Parkinson: σ² = (1/(4·ln2))·mean[(ln H/L)²]
- Garman-Klass: σ² = mean[0.5·(ln H/L)² − (2ln2−1)·(ln C/O)²]
- Rogers-Satchell: σ² = mean[ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)]
- Yang-Zhang: σ²_o + k·σ²_c + (1−k)·σ²_RS with k = 0.34/(1.34 + (n+1)/(n−1))
- Default estimator: Garman-Klass (efficient, robust when overnight gaps are small, as in FX).

**Causality rule:** every *signal* feature at bar *t* uses data ≤ *t* only (Donchian
levels use the prior N bars; percentiles are rolling-historical). *Forward* RV used by
the event-study harness intentionally looks ahead — it is measurement, not signal.
A dedicated test asserts prefix-stability of states (appending future data never
changes past states).

**State machine defaults** (all in a `RegimeConfig` dataclass): Donchian N=20,
confirmation window 5 bars, ≥2 closes beyond the broken level to confirm, close back
through the level ⇒ failed; compression = in-range ∧ width percentile < 25 ∧ composite
score ≥ 0.6; trend exits when 10-bar efficiency ratio stays < 0.25 for 5 bars.

**Event study:** for each event type, forward RV at horizons {5, 10, 21, 42} bars,
reported as median/quartiles vs the unconditional distribution, expansion probability
P(fwd RV > trailing RV), and a Monte-Carlo p-value (median of N random same-size date
samples). Events de-clustered by a minimum separation. Sample-size warnings below
n = 15 events.

**HAR-RV:** ŷ = β₀ + β_d·RV_d + β_w·RV̄_5 + β_m·RV̄_21 on log RV, refit walk-forward;
report OOS MAE and R² vs the trailing-RV random walk. Regime-augmented variant adds
compression score + trend dummy; its *incremental* OOS skill is the test of whether the
regime layer adds forecast power beyond vanilla HAR.

**Tenor mapping:** compression time-to-resolution (median bars to first expansion bar or
break) converts states into tenor guidance (e.g. resolution ≈ 9 bars ⇒ favor 2–3 week
optionality, not 3M).

## 4. Validation (test suite)

1. **Estimator correctness:** on simulated GBM with known σ, every estimator converges
   to σ within tolerance; range-based estimators show lower sampling variance than
   close-to-close on the same paths.
2. **Regime detection:** on synthetic series with planted compression → breakout and
   planted failed breaks, the state machine finds them (and doesn't on pure noise).
3. **No lookahead:** states computed on a prefix equal the same rows computed on the
   full series.
4. **Event-study power & calibration:** a planted "signal precedes vol doubling" is
   reported with expansion probability ≈ 1 and small p-value; a random signal reports
   ratio ≈ 1 and large p-value.
5. **HAR:** beats the naive baseline OOS on synthetic vol-clustered data; no NaN leakage.
6. **HMM:** recovers planted two-regime structure with high state accuracy.

## 5. Build order

1. `rv_estimators` + `cones` (everything downstream consumes these)
2. `regime` features + state machine + events
3. `event_study` harness (the centerpiece)
4. `har`, `hmm`
5. `breakeven`, `breadth`, `playbook`, `synthetic`, `data_io`, CLI
6. Tests green
7. Dashboard page + charts, wired to existing data fetch with provenance labels
8. Commit/push

## 6. Work-PC migration path

1. Copy `vol_engine/` (+ `tests/`) — no dashboard needed to run analyses (`run_vol_analysis.py`).
2. Export 10–20 years of daily OHLC per pair to the documented CSV schema; rerun the
   event studies — thresholds in `RegimeConfig` can then be tuned per pair against
   *measured* conditional distributions.
3. Feed real ATM IV (and later RR/BF) into `breakeven` and alongside HAR forecasts:
   conditional forward RV − entry IV per state = the regime-conditional vol risk
   premium, which is the tradeable output.
