"""Vol Regime page — regime states, event-study evidence, vol forecasts, playbook.

All analytics come from the standalone `vol_engine` package (no Streamlit inside it);
this page is only a viewing surface. Data provenance is always displayed, and every
statistic degrades to an explicit warning — never a fabricated default — when history
is insufficient.
"""

import numpy as np
import pandas as pd
import streamlit as st

from vol_engine.data_io import validate_ohlc, DataContractError
from vol_engine.synthetic import generate_ohlc
from vol_engine.regime import classify_states, RegimeConfig, EventKind, VolState
from vol_engine.cones import vol_cone
from vol_engine.event_study import (
    event_study, event_time_path, time_to_resolution, trailing_rv,
)
from vol_engine.har import har_forecast_report
from vol_engine.event_study import forward_rv
from vol_engine.hmm import fit_hmm_2state
from vol_engine.levels import tag_break_events, break_event_masks
from vol_engine.breadth import compression_breadth
from vol_engine.playbook import get_playbook
from vol_engine.rv_estimators import log_returns
from visualization.vol_regime_charts import (
    state_chart, cone_chart, rv_panel_chart, compression_chart,
    event_path_chart, har_chart, hmm_chart, breadth_chart, STATE_LABEL,
)

st.set_page_config(page_title="Vol Regime", page_icon="🌊", layout="wide")

st.title("🌊 Vol Regime Engine")
st.caption(
    "Regime states for options positioning — with event-study evidence instead of lore. "
    "Analytics run in the standalone `vol_engine` package (portable to the work PC)."
)

MIN_BARS_STATES = 160        # donchian warmup + percentile min_periods
MIN_BARS_STUDIES = 750       # below this, event studies are indicative at best


# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def load_project_data(symbol: str) -> pd.DataFrame:
    from core.data_fetcher import fetch_currency_data
    raw = fetch_currency_data(symbol, period="max", interval="1d")
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = validate_ohlc(raw, min_bars=1)
    df.attrs["provenance"] = (
        f"Project data source (yfinance or CSV fallback) — "
        f"{df.index.min().date()} to {df.index.max().date()}, {len(df)} bars"
    )
    return df


@st.cache_data(show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    df, _ = generate_ohlc()
    return df


@st.cache_data(show_spinner=False)
def run_engine(df_key: str, df: pd.DataFrame, donchian_n: int):
    cfg = RegimeConfig(donchian_n=donchian_n)
    result = classify_states(df, cfg)
    return cfg, result


with st.sidebar:
    st.header("Data")
    source = st.radio(
        "Source",
        ["Synthetic demo (labeled)", "Project data (yfinance/CSV)"],
        help=(
            "The demo series has planted regimes and exists to show the engine's "
            "mechanics. Switch to project data — or export CSVs on the work PC — "
            "for real analysis."
        ),
    )
    if source.startswith("Project"):
        symbol = st.selectbox(
            "Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCNH=X"]
        )
    st.divider()
    st.header("Regime config")
    donchian_n = st.slider("Channel lookback (bars)", 10, 60, 20, 5)


if source.startswith("Synthetic"):
    data = load_demo_data()
    data_key = "demo"
else:
    with st.spinner("Loading data..."):
        data = load_project_data(symbol)
    data_key = symbol

if data.empty:
    st.error("No data available from the project source. Use the synthetic demo, "
             "or run on a machine with data access.")
    st.stop()

# Provenance banner — always visible
if data.attrs.get("synthetic"):
    st.warning(f"**SYNTHETIC DATA** — {data.attrs.get('provenance')}")
else:
    st.info(f"Data: {data.attrs.get('provenance', 'unknown source')}")

if len(data) < MIN_BARS_STATES:
    st.error(
        f"Only {len(data)} bars — the regime engine needs at least {MIN_BARS_STATES} "
        "for channel and percentile warmup. No analysis shown (rather than a wrong one)."
    )
    st.stop()

cfg, result = run_engine(data_key, data, donchian_n)
feats = result.features
states = result.states
events = result.events_frame()


# ---------------------------------------------------------------------------
# Current state
# ---------------------------------------------------------------------------

current_state = states.iloc[-1]
st.header(f"Current state: {STATE_LABEL.get(current_state, current_state)}")

c1, c2, c3, c4, c5 = st.columns(5)


def metric_or_na(col, label, value, fmt="{:.2f}", help_text=None):
    with col:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            st.metric(label, "n/a", help=help_text or "Insufficient history")
        else:
            st.metric(label, fmt.format(value), help=help_text)


metric_or_na(c1, "Compression score", float(feats["compression_score"].iloc[-1]),
             help_text="0-1 composite: channel width, ATR ratio, RV backwardation, quiet bars")
metric_or_na(c2, "Channel width pctile", float(feats["width_pctile"].iloc[-1]), "{:.0f}",
             help_text="Donchian width vs trailing year — low = compressed")
metric_or_na(c3, "Efficiency ratio", float(feats["er"].iloc[-1]),
             help_text="|net move| / path length over 10 bars — high = clean trend")
metric_or_na(c4, "ATR 5/20 ratio", float(feats["atr_ratio"].iloc[-1]),
             help_text="< 1 = short-term ranges contracting")
metric_or_na(c5, "RV 5/21 ratio", float(feats["rv_5_over_21"].iloc[-1]),
             help_text="< 1 = realized vol in backwardation (compressing)")

# Recent break context
resolved = [e for e in result.events
            if e.kind in (EventKind.BREAK_CONFIRMED, EventKind.BREAK_FAILED)]
recent_failed = bool(resolved) and resolved[-1].kind == EventKind.BREAK_FAILED
if resolved:
    last = resolved[-1]
    st.caption(
        f"Last resolved break: **{last.kind.value}** ({last.direction}) on "
        f"{last.date.date()} at {last.level:.5f}"
    )

st.plotly_chart(
    state_chart(data, states, feats, events, title="Price, channel and regime states"),
    width='stretch',
)


# ---------------------------------------------------------------------------
# Vol structure
# ---------------------------------------------------------------------------

st.header("Vol structure")
col_a, col_b = st.columns(2)
with col_a:
    cone = vol_cone(data)
    if cone.empty:
        st.warning("Not enough history for a vol cone.")
    else:
        st.plotly_chart(cone_chart(cone), width='stretch')
        st.caption(
            "Overlay real IV per tenor here on the work PC "
            "(`cone_chart(cone, iv_points={days: iv_pct})`)."
        )
with col_b:
    st.plotly_chart(rv_panel_chart(feats), width='stretch')

st.plotly_chart(
    compression_chart(feats, cfg.compression_score_min), width='stretch'
)


# ---------------------------------------------------------------------------
# Event-study evidence
# ---------------------------------------------------------------------------

st.header("Event-study evidence")
st.caption(
    "Conditional forward realized vol after each signal vs the unconditional baseline. "
    "ratio > 1 with a small p-value = the signal predicted expansion; "
    "P(expand) = share of events where forward RV exceeded trailing RV."
)

if len(data) < MIN_BARS_STUDIES:
    st.warning(
        f"Only {len(data)} bars — event studies below are INDICATIVE ONLY. "
        f"For reliable conditional distributions use {MIN_BARS_STUDIES}+ bars "
        "(10-20y of daily data on the work PC)."
    )

tab_comp, tab_break = st.tabs(["Compression start", "Break confirmed"])

for tab, kind, label in [
    (tab_comp, EventKind.COMPRESSION_START, "compression_start"),
    (tab_break, EventKind.BREAK_CONFIRMED, "break_confirmed"),
]:
    with tab:
        mask = result.event_mask(kind)
        report = event_study(data, mask, label)
        if report.n_events_used == 0:
            st.info("No events of this type in the loaded history.")
            continue
        for w in report.warnings:
            st.warning(w)
        st.dataframe(report.to_frame(), width='stretch', hide_index=True)

        idx = np.flatnonzero(mask.values)
        rv5 = trailing_rv(data, 5)
        st.plotly_chart(
            event_path_chart(
                event_time_path(rv5, idx, pre=20, post=30),
                title=f"RV (5d) path around {label} events (n={len(idx)})",
            ),
            width='stretch',
        )

        if kind == EventKind.COMPRESSION_START:
            res_mask = feats["expansion_bar"] | result.event_mask(EventKind.BREAK_CONFIRMED)
            stats = time_to_resolution(idx, res_mask)
            if "median_bars" in stats:
                st.markdown(
                    f"**Time to resolution:** median **{stats['median_bars']:.0f} bars** "
                    f"(IQR {stats['q25_bars']:.0f}-{stats['q75_bars']:.0f}), "
                    f"{stats['n_censored']:.0f} unresolved of {stats['n']:.0f}. "
                    "This maps the signal to an option tenor: own expiries that "
                    "bracket the resolution window."
                )

        if kind == EventKind.BREAK_CONFIRMED:
            st.subheader("Conditioned on S/R level history")
            st.caption(
                "Breaks split by the broken level's causally-known test history "
                "(swing-based level tracker — the liquidity-consumption idea: "
                "well-tested levels hold trapped positioning, so their breaks "
                "should release more vol than drifts through untouched prices)."
            )
            tagged = tag_break_events(events, data, feats)
            at_tested, elsewhere = break_event_masks(tagged, data.index)
            split_cols = st.columns(2)
            for col, sub_mask, sub_label in [
                (split_cols[0], at_tested, "At tested level (>= 2 prior touches)"),
                (split_cols[1], elsewhere, "Fresh / no tracked level"),
            ]:
                with col:
                    st.markdown(f"**{sub_label}** — n={int(sub_mask.sum())}")
                    if sub_mask.sum() == 0:
                        st.info("No events in this group.")
                        continue
                    sub_report = event_study(data, sub_mask, sub_label)
                    for w in sub_report.warnings:
                        st.warning(w)
                    st.dataframe(sub_report.to_frame(), width='stretch',
                                 hide_index=True)
            brk = tagged[tagged["kind"] == "break_confirmed"]
            if not brk.empty:
                phases = brk["sr_phase"].value_counts().to_dict()
                st.caption(f"Broken-level phases: {phases}")


# ---------------------------------------------------------------------------
# Cross-pair breadth
# ---------------------------------------------------------------------------

st.header("Cross-pair breadth")
st.caption(
    "A break confirmed while most of the complex is also expanding is more likely "
    "genuine than a lone-pair move; broad compression means the whole board is "
    "coiling. Fraction of pairs in each state:"
)


@st.cache_data(show_spinner=False)
def demo_breadth_frame(donchian: int) -> pd.DataFrame:
    results = {}
    for i, seed in enumerate((42, 7, 99, 123)):
        d, _ = generate_ohlc(seed=seed)
        results[f"DEMO_{i + 1}"] = classify_states(d, RegimeConfig(donchian_n=donchian))
    return compression_breadth(results)


@st.cache_data(ttl=600, show_spinner=False)
def project_breadth_frame(donchian: int):
    pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCNH=X"]
    results, skipped = {}, []
    for sym in pairs:
        d = load_project_data(sym)
        if len(d) < MIN_BARS_STATES:
            skipped.append(f"{sym} ({len(d)} bars)")
            continue
        results[sym.replace("=X", "")] = classify_states(
            d, RegimeConfig(donchian_n=donchian)
        )
    frame = compression_breadth(results) if results else pd.DataFrame()
    return frame, skipped


if source.startswith("Synthetic"):
    st.warning("Breadth below is computed across 4 SYNTHETIC series — mechanics demo only.")
    breadth_frame = demo_breadth_frame(donchian_n)
else:
    with st.spinner("Computing breadth across pairs..."):
        breadth_frame, skipped = project_breadth_frame(donchian_n)
    if skipped:
        st.warning("Skipped (insufficient history): " + ", ".join(skipped))

if breadth_frame.empty:
    st.info("No pairs with sufficient history for breadth. On the work PC, point the "
            "engine at long per-pair CSVs (`run_vol_analysis.py --csv-dir`).")
else:
    latest = breadth_frame.dropna(subset=["fraction_compressed"]).iloc[-1]
    b1, b2, b3 = st.columns(3)
    b1.metric("Compressed now", f"{latest['fraction_compressed']:.0%}",
              help="Fraction of pairs currently in the compression state")
    b2.metric("Trending now", f"{latest['fraction_trending']:.0%}")
    b3.metric("Pairs covered", f"{int(latest['n_pairs'])}")
    st.plotly_chart(breadth_chart(breadth_frame), width='stretch')


# ---------------------------------------------------------------------------
# Playbook
# ---------------------------------------------------------------------------

st.header("Playbook for the current state")
comp_idx = np.flatnonzero(result.event_mask(EventKind.COMPRESSION_START).values)
res_stats = None
if len(comp_idx):
    res_mask = feats["expansion_bar"] | result.event_mask(EventKind.BREAK_CONFIRMED)
    res_stats = time_to_resolution(comp_idx, res_mask)

entry = get_playbook(current_state, res_stats, recent_failed)
st.subheader(entry.stance)
st.write(entry.rationale)
pcol1, pcol2 = st.columns(2)
with pcol1:
    st.markdown("**Structures**")
    for s in entry.structures:
        st.write(f"- {s}")
    if entry.tenor_guidance:
        st.markdown(f"**Tenor:** {entry.tenor_guidance}")
with pcol2:
    st.markdown("**Caveats**")
    for c in entry.caveats:
        st.write(f"- {c}")


# ---------------------------------------------------------------------------
# Forecast layer
# ---------------------------------------------------------------------------

st.header("Vol forecast (HAR-RV)")
with st.spinner("Walk-forward HAR evaluation..."):
    extra = pd.DataFrame({
        "compression_score": feats["compression_score"],
        "trend_dummy": (states == VolState.TREND.value).astype(float),
    })
    har = har_forecast_report(data, horizon=21, extra_features=extra)

if har.warnings:
    for w in har.warnings:
        st.warning(w)
else:
    m1, m2, m3, m4 = st.columns(4)
    latest_fc = har.forecast.dropna()
    m1.metric("Forecast 21d vol", f"{latest_fc.iloc[-1]:.2f}%" if len(latest_fc) else "n/a",
              help="Walk-forward HAR forecast of annualized RV over the next 21 bars")
    m2.metric("OOS MAE: HAR", f"{har.oos_mae_har:.2f}")
    m3.metric("OOS MAE: naive", f"{har.oos_mae_naive:.2f}",
              help="Naive = trailing 21d RV (random-walk forecast)")
    m4.metric("OOS R²: HAR", f"{har.oos_r2_har:.2f}")

    if har.oos_mae_augmented is not None:
        delta = har.oos_mae_har - har.oos_mae_augmented
        verdict = "adds" if delta > 0 else "does not add"
        st.caption(
            f"Regime-augmented HAR OOS MAE {har.oos_mae_augmented:.2f} vs vanilla "
            f"{har.oos_mae_har:.2f} — on this data the regime layer **{verdict}** "
            "forecast power beyond HAR. This is the honest test to rerun on real data."
        )

    realized_fwd = forward_rv(data, 21)
    st.plotly_chart(har_chart(har.forecast, realized_fwd), width='stretch')
    st.caption(
        "On the work PC: quoted 1M implied vol vs this forecast (adjusted by the "
        "state-conditional distributions above) = the vol risk premium you are "
        "being paid or paying."
    )

with st.expander("HMM cross-check (statistical regime model)"):
    hmm = fit_hmm_2state(log_returns(data["Close"]))
    if hmm is None:
        st.warning("Series too short for the HMM.")
    else:
        st.write(
            f"2-state Gaussian HMM: low-vol sigma "
            f"{hmm.sigmas[0] * np.sqrt(252) * 100:.1f}%, high-vol sigma "
            f"{hmm.sigmas[1] * np.sqrt(252) * 100:.1f}% (annualized). "
            f"P(high-vol now) = {hmm.smoothed_p_high.iloc[-1]:.2f}."
        )
        st.plotly_chart(hmm_chart(hmm.smoothed_p_high), width='stretch')
        st.caption(
            "Agreement between this agnostic model and the rule-based states is a "
            "sanity check; disagreement flags periods worth inspecting."
        )
