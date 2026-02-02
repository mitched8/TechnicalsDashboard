"""S/R Analysis Page - Professional Support/Resistance Analysis.

This page provides comprehensive S/R analysis implementing:
- Liquidity consumption model (discovery vs depletion phases)
- Multi-timeframe hierarchy scoring
- Reaction quality analysis
- Compression pattern detection
- Touch pattern analysis with diminishing bounce warnings
- Comprehensive confluence scoring
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import CURRENCY_PAIRS, SR_LOOKBACK_OPTIONS
from core.data_fetcher import fetch_multi_timeframe_data
from core.indicators import add_indicators
from core.sr_analysis import (
    analyze_multi_timeframe_sr, detect_enhanced_levels,
    get_zone_trading_notes, EnhancedZone, LiquidityPhase,
    ApproachPattern, ZoneType, TIMEFRAME_WEIGHTS, SCORING_CONFIG
)


st.set_page_config(
    page_title="S/R Analysis",
    page_icon="",
    layout="wide"
)

st.title("Support & Resistance Analysis")
st.markdown("""
Professional S/R analysis with liquidity consumption model, multi-timeframe hierarchy,
reaction quality scoring, and compression pattern detection.
""")

# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================

st.sidebar.header("S/R Configuration")

# Currency selection
symbol = st.sidebar.selectbox(
    "Currency Pair",
    options=list(CURRENCY_PAIRS.keys()),
    format_func=lambda x: x.replace("=X", "")
)

pair_config = CURRENCY_PAIRS[symbol]

# Lookback periods
st.sidebar.subheader("Lookback Periods")
weekly_lookback = st.sidebar.slider("Weekly Lookback (weeks)", 26, 104, 52, 2)
daily_lookback = st.sidebar.slider("Daily Lookback (days)", 60, 504, 252, 10)
h4_lookback = st.sidebar.slider("4H Lookback (bars)", 100, 500, 250, 25)

lookback_config = {
    "weekly": weekly_lookback,
    "daily": daily_lookback,
    "4h": h4_lookback,
}

# Filter settings
st.sidebar.subheader("Filters")
min_score = st.sidebar.slider("Minimum Score", 0.0, 8.0, 2.0, 0.5)
show_timeframes = st.sidebar.multiselect(
    "Show Timeframes",
    options=["weekly", "daily", "4h"],
    default=["weekly", "daily", "4h"]
)

# Display settings
st.sidebar.subheader("Display")
show_touch_details = st.sidebar.checkbox("Show Touch Details", value=True)
show_score_breakdown = st.sidebar.checkbox("Show Score Breakdown", value=True)
highlight_warnings = st.sidebar.checkbox("Highlight Warnings", value=True)

# ==============================================================================
# DATA LOADING
# ==============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_and_analyze(symbol: str, lookback_config: dict, pip_step: float):
    """Load data and perform S/R analysis."""
    data_dict = fetch_multi_timeframe_data(symbol, daily_period="2y")

    # Add indicators to daily data for additional S/R detection
    if not data_dict["daily"].empty:
        data_dict["daily"] = add_indicators(data_dict["daily"])

    # Perform analysis
    result = analyze_multi_timeframe_sr(data_dict, pip_step, lookback_config)

    return data_dict, result


with st.spinner("Loading data and analyzing S/R levels..."):
    data_dict, sr_result = load_and_analyze(
        symbol, lookback_config, pair_config.round_number_step
    )

if sr_result.current_price == 0:
    st.error("No data available for analysis")
    st.stop()


# ==============================================================================
# OVERVIEW METRICS
# ==============================================================================

st.header("Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Price",
        f"{sr_result.current_price:.5f}" if pair_config.pip_decimal == 4 else f"{sr_result.current_price:.3f}"
    )

with col2:
    if sr_result.nearest_support:
        distance = (sr_result.current_price - sr_result.nearest_support.center) / sr_result.current_price * 100
        st.metric(
            "Nearest Support",
            f"{sr_result.nearest_support.center:.5f}" if pair_config.pip_decimal == 4 else f"{sr_result.nearest_support.center:.3f}",
            f"{distance:.2f}% away"
        )
    else:
        st.metric("Nearest Support", "None detected")

with col3:
    if sr_result.nearest_resistance:
        distance = (sr_result.nearest_resistance.center - sr_result.current_price) / sr_result.current_price * 100
        st.metric(
            "Nearest Resistance",
            f"{sr_result.nearest_resistance.center:.5f}" if pair_config.pip_decimal == 4 else f"{sr_result.nearest_resistance.center:.3f}",
            f"{distance:.2f}% away"
        )
    else:
        st.metric("Nearest Resistance", "None detected")

with col4:
    total_levels = len(sr_result.supports) + len(sr_result.resistances)
    st.metric("Total Levels", total_levels)

# Alert banners
if sr_result.compression_into_resistance:
    st.warning("COMPRESSION ALERT: Higher lows forming into nearest resistance - elevated break probability")

if sr_result.compression_into_support:
    st.warning("COMPRESSION ALERT: Lower highs forming into nearest support - elevated break probability")


# ==============================================================================
# PRICE CHART WITH S/R ZONES
# ==============================================================================

st.header("Price Chart with S/R Zones")

# Chart timeframe selection
chart_tf = st.radio(
    "Chart Timeframe",
    options=["Daily", "Weekly", "4H"],
    horizontal=True
)

tf_map = {"Daily": "daily", "Weekly": "weekly", "4H": "4h"}
chart_data = data_dict.get(tf_map[chart_tf], pd.DataFrame())

if not chart_data.empty:
    # Create candlestick chart with S/R zones
    fig = go.Figure()

    # Add candlesticks
    fig.add_trace(go.Candlestick(
        x=chart_data.index,
        open=chart_data['Open'],
        high=chart_data['High'],
        low=chart_data['Low'],
        close=chart_data['Close'],
        name="Price",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))

    # Collect all zones to display
    all_zones = []
    if "weekly" in show_timeframes:
        all_zones.extend(sr_result.weekly_levels)
    if "daily" in show_timeframes:
        all_zones.extend(sr_result.daily_levels)
    if "4h" in show_timeframes:
        all_zones.extend(sr_result.h4_levels)

    # Filter by score
    filtered_zones = [z for z in all_zones if z.total_score >= min_score]

    # Add S/R zones
    for zone in filtered_zones:
        # Color based on type and warning state
        if zone.liquidity_phase in [LiquidityPhase.DEPLETION, LiquidityPhase.EXHAUSTED]:
            # Warning color for high break risk
            color = "rgba(255, 165, 0, 0.2)" if highlight_warnings else (
                "rgba(76, 175, 80, 0.2)" if zone.zone_type == ZoneType.SUPPORT else "rgba(244, 67, 54, 0.2)"
            )
            line_color = "orange" if highlight_warnings else (
                "green" if zone.zone_type == ZoneType.SUPPORT else "red"
            )
        elif zone.approach_pattern == ApproachPattern.COMPRESSION:
            color = "rgba(255, 193, 7, 0.3)" if highlight_warnings else (
                "rgba(76, 175, 80, 0.2)" if zone.zone_type == ZoneType.SUPPORT else "rgba(244, 67, 54, 0.2)"
            )
            line_color = "#FFC107" if highlight_warnings else (
                "green" if zone.zone_type == ZoneType.SUPPORT else "red"
            )
        else:
            color = "rgba(76, 175, 80, 0.2)" if zone.zone_type == ZoneType.SUPPORT else "rgba(244, 67, 54, 0.2)"
            line_color = "green" if zone.zone_type == ZoneType.SUPPORT else "red"

        # Line style based on timeframe
        dash = "solid" if zone.source_timeframe == "weekly" else (
            "dash" if zone.source_timeframe == "daily" else "dot"
        )

        # Add zone as filled area
        fig.add_hrect(
            y0=zone.lower_bound,
            y1=zone.upper_bound,
            fillcolor=color,
            line=dict(color=line_color, width=1, dash=dash),
            annotation_text=f"{zone.source_timeframe[0].upper()} | {zone.total_score:.1f}",
            annotation_position="right",
        )

    # Update layout
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        yaxis_title="Price",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=30, b=30),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Chart legend
    st.caption("""
    **Zone Lines:** Solid = Weekly | Dashed = Daily | Dotted = 4H |
    **Colors:** Green = Support | Red = Resistance | Orange/Yellow = Warning (high break risk or compression)
    """)

else:
    st.warning(f"No {chart_tf} data available")


# ==============================================================================
# DETAILED LEVEL ANALYSIS
# ==============================================================================

st.header("Level Analysis")

tab1, tab2, tab3 = st.tabs(["Resistance Levels", "Support Levels", "By Timeframe"])

def display_zone_details(zone: EnhancedZone, pip_decimal: int):
    """Display detailed information about a zone."""

    # Header with score badge
    score_color = {
        "Prime": "green",
        "Strong": "blue",
        "Moderate": "orange",
        "Weak": "gray"
    }.get(zone.score_category, "gray")

    col1, col2 = st.columns([3, 1])
    with col1:
        price_fmt = f"{zone.center:.5f}" if pip_decimal == 4 else f"{zone.center:.3f}"
        st.markdown(f"### {price_fmt}")
    with col2:
        st.markdown(f"**Score:** :{score_color}[{zone.total_score:.1f}] ({zone.score_category})")

    # Key info
    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.markdown(f"**Timeframe:** {zone.source_timeframe.upper()}")
        st.markdown(f"**Total Touches:** {zone.touch_count}")
        # Show clustering info
        if zone.total_clusters > 0:
            cluster_info = f"({zone.recent_cluster_size} recent"
            if zone.has_replenished:
                cluster_info += f", {zone.total_clusters} clusters)"
            else:
                cluster_info += ")"
            st.caption(cluster_info)

    with info_col2:
        phase_emoji = {
            LiquidityPhase.DISCOVERY: "",
            LiquidityPhase.ESTABLISHED: "",
            LiquidityPhase.DEPLETION: "",
            LiquidityPhase.EXHAUSTED: "",
        }.get(zone.liquidity_phase, "")
        st.markdown(f"**Liquidity Phase:** {phase_emoji} {zone.liquidity_phase.value.title()}")
        st.markdown(f"_{zone.liquidity_description}_")

    with info_col3:
        badges = []
        if zone.is_round_number:
            badges.append("Round Number")
        if zone.is_role_flip:
            badges.append("Role Flip")
        if zone.approach_pattern == ApproachPattern.COMPRESSION:
            badges.append("COMPRESSION")
        if zone.has_replenished:
            badges.append("Replenished")
        st.markdown(f"**Flags:** {', '.join(badges) if badges else 'None'}")

    # Score breakdown
    if show_score_breakdown:
        with st.expander("Score Breakdown"):
            for component, score in zone.score_breakdown.items():
                bar_width = max(0, min(100, (score + 2) / 6 * 100))  # Normalize to 0-100
                color = "green" if score > 0 else "red" if score < 0 else "gray"
                st.markdown(f"**{component}:** {score:+.1f}")
                st.progress(bar_width / 100)

    # Trading notes
    notes = get_zone_trading_notes(zone, sr_result.current_price)
    if notes:
        with st.expander("Trading Notes", expanded=highlight_warnings):
            for note in notes:
                if "WARNING" in note or "HIGH" in note or "COMPRESSION" in note:
                    st.warning(note)
                elif "Caution" in note:
                    st.info(note)
                else:
                    st.write(f"- {note}")

    # Touch details
    if show_touch_details and zone.touches:
        with st.expander(f"Touch History ({len(zone.touches)} touches)"):
            touch_data = []
            for touch in zone.touches[-10:]:  # Last 10 touches
                touch_data.append({
                    "Date": touch.timestamp.strftime("%Y-%m-%d"),
                    "Reaction": touch.reaction_quality.value.replace("_", " ").title(),
                    "Held": "Yes" if touch.held else "No",
                    "Bounce Size": f"{touch.bounce_size_pct:.1f}x ATR",
                    "Wick %": f"{touch.wick_size_pct*100:.0f}%"
                })
            if touch_data:
                st.dataframe(pd.DataFrame(touch_data), hide_index=True, use_container_width=True)

            # Diminishing bounce warning
            if zone.diminishing_bounces:
                st.error("Diminishing bounces detected - bounces getting smaller with each test")

    st.divider()


with tab1:
    st.subheader("Resistance Levels (Above Price)")
    resistances = [z for z in sr_result.resistances if z.total_score >= min_score]
    resistances = [z for z in resistances if z.source_timeframe in show_timeframes]

    if resistances:
        for zone in sorted(resistances, key=lambda z: z.total_score, reverse=True):
            display_zone_details(zone, pair_config.pip_decimal)
    else:
        st.info("No resistance levels match the current filters")

with tab2:
    st.subheader("Support Levels (Below Price)")
    supports = [z for z in sr_result.supports if z.total_score >= min_score]
    supports = [z for z in supports if z.source_timeframe in show_timeframes]

    if supports:
        for zone in sorted(supports, key=lambda z: z.total_score, reverse=True):
            display_zone_details(zone, pair_config.pip_decimal)
    else:
        st.info("No support levels match the current filters")

with tab3:
    st.subheader("Levels by Timeframe")

    for tf in ["weekly", "daily", "4h"]:
        if tf not in show_timeframes:
            continue

        tf_zones = {
            "weekly": sr_result.weekly_levels,
            "daily": sr_result.daily_levels,
            "4h": sr_result.h4_levels,
        }[tf]

        filtered = [z for z in tf_zones if z.total_score >= min_score]

        st.markdown(f"### {tf.upper()} Timeframe ({len(filtered)} levels)")

        if filtered:
            # Summary table
            table_data = []
            for zone in sorted(filtered, key=lambda z: z.total_score, reverse=True):
                price_fmt = f"{zone.center:.5f}" if pair_config.pip_decimal == 4 else f"{zone.center:.3f}"
                zone_type_str = "Support" if zone.zone_type == ZoneType.SUPPORT else "Resistance"
                table_data.append({
                    "Level": price_fmt,
                    "Type": zone_type_str,
                    "Score": f"{zone.total_score:.1f}",
                    "Category": zone.score_category,
                    "Liquidity": zone.liquidity_phase.value.title(),
                    "Touches": zone.touch_count,
                    "Flags": ", ".join([
                        f for f in [
                            "Round" if zone.is_round_number else "",
                            "Flip" if zone.is_role_flip else "",
                            "Compress" if zone.approach_pattern == ApproachPattern.COMPRESSION else ""
                        ] if f
                    ]) or "-"
                })

            st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)
        else:
            st.info(f"No {tf} levels match the current filters")

        st.divider()


# ==============================================================================
# LIQUIDITY CONSUMPTION CHART
# ==============================================================================

st.header("Liquidity Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Liquidity Phase Distribution")

    # Count zones by phase
    all_zones = sr_result.supports + sr_result.resistances
    phase_counts = {
        "Discovery": len([z for z in all_zones if z.liquidity_phase == LiquidityPhase.DISCOVERY]),
        "Established": len([z for z in all_zones if z.liquidity_phase == LiquidityPhase.ESTABLISHED]),
        "Depletion": len([z for z in all_zones if z.liquidity_phase == LiquidityPhase.DEPLETION]),
        "Exhausted": len([z for z in all_zones if z.liquidity_phase == LiquidityPhase.EXHAUSTED]),
    }

    fig_pie = go.Figure(data=[go.Pie(
        labels=list(phase_counts.keys()),
        values=list(phase_counts.values()),
        marker_colors=['#4CAF50', '#2196F3', '#FF9800', '#F44336'],
        hole=0.4
    )])
    fig_pie.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Score Distribution")

    # Histogram of scores
    scores = [z.total_score for z in all_zones]
    if scores:
        fig_hist = go.Figure(data=[go.Histogram(
            x=scores,
            nbinsx=10,
            marker_color='#2196F3'
        )])
        fig_hist.update_layout(
            height=300,
            xaxis_title="Score",
            yaxis_title="Count",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ==============================================================================
# EDUCATIONAL CONTENT
# ==============================================================================

with st.expander("Understanding S/R Analysis"):
    st.markdown("""
    ### Liquidity Consumption Model

    S/R levels are not magic lines - they represent zones of order clustering where traders
    have placed stops, limits, and pending orders.

    **Key Insight: Liquidity Replenishes Over Time**

    A level tested 5 times over 2 years is fundamentally different from one tested 5 times
    in 2 weeks. The former allows liquidity to replenish between tests (new orders placed),
    while the latter is actively consuming the available liquidity.

    The system tracks "touch clusters" - groups of tests that occur close together in time.
    When there's a significant gap between tests (~25+ bars on daily), liquidity has time
    to replenish and the level can regain strength.

    **Discovery Phase (1-3 recent tests):** Level is being discovered by the market. Each quality
    test that holds increases institutional confidence. These are the freshest, most tradeable levels.

    **Established Phase:** Level is proven and known. May have multiple test clusters over time
    with replenishment between them. Look for "Replenished" flag - these levels are strong because
    liquidity has been restored after previous tests.

    **Depletion Phase (6+ rapid tests):** Liquidity is being consumed with each touch. Key warning
    signs:
    - Many tests in a short time period (same cluster)
    - Diminishing bounces (each reaction smaller than the last)
    - Tighter consolidation near the level

    **Exhausted Phase:** Level is likely to break. Rapid repeated tests with diminishing bounces
    indicate the orders that made it hold are largely filled.

    ### Compression Patterns

    When price makes higher lows into resistance (or lower highs into support), it signals
    building pressure. This "compression" often precedes breaks because:
    - Stop orders cluster just beyond the level
    - Buyers become more aggressive (willing to pay higher prices)
    - Sellers defending the level face increasing pressure

    ### Timeframe Hierarchy

    Higher timeframe levels carry more significance:
    - **Weekly:** Institutional flow, major regime changes
    - **Daily:** Professional traders, swing trades
    - **4H:** Intraday positioning

    A daily level confluent with a weekly level is significantly stronger than a daily level alone.

    ### Scoring System

    The total score combines:
    - **Timeframe:** Weekly (3), Daily (2), 4H (1)
    - **Structure:** Swing high/low (2), Role flip (2), Volume cluster (1)
    - **Reactions:** Strong rejection (+2), Mild rejection (+1), Acceptance (-1)
    - **Recency:** Recent test (+1), Medium (+0.5), Old (0)
    - **Touch Pattern:** Discovery (+1.5), Established (+1), Depletion (-0.5), Exhausted (-1.5)
    - **Approach:** Compression (-1), Impulse (+0.5)
    - **Confluence:** Round number (+1), Multi-source (+0.5 each), MTF (+1.5)
    """)
