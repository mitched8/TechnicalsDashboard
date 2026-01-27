"""FX Trading Dashboard - Professional Technical Analysis Tool."""

import streamlit as st
import pandas as pd

from config import CURRENCY_PAIRS
from core.data_fetcher import fetch_multi_timeframe_data, validate_data
from core.indicators import calculate_all_indicators
from core.support_resistance import analyze_support_resistance
from core.regime import detect_regime, analyze_mtf_context, TrendDirection
from core.trade_setup import generate_trade_setup, Bias
from visualization.charts import create_main_chart, create_mtf_summary_chart
from utils.forex_utils import (
    get_pip_info, format_price, format_pip_distance,
    format_percentage_move, format_level_display
)

st.set_page_config(
    page_title="FX Trading Dashboard",
    page_icon="📈",
    layout="wide"
)


def main():
    st.title("FX Trading Dashboard")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        symbol = st.selectbox(
            "Currency Pair",
            options=list(CURRENCY_PAIRS.keys()),
            format_func=lambda x: x.replace("=X", "")
        )

        st.divider()
        show_raw_data = st.checkbox("Show Raw Data", value=False)

    # Get pip configuration
    pip_decimal, round_step = get_pip_info(symbol)

    # Fetch multi-timeframe data
    with st.spinner("Fetching market data..."):
        mtf_data = fetch_multi_timeframe_data(symbol)

    if not validate_data(mtf_data.get('daily', pd.DataFrame())):
        st.error("Failed to fetch data. Please try again.")
        return

    # Calculate indicators for each timeframe
    for tf in ['weekly', 'daily', '4h']:
        if not mtf_data[tf].empty:
            mtf_data[tf], _ = calculate_all_indicators(mtf_data[tf])

    daily_data = mtf_data['daily']
    current_price = float(daily_data['Close'].iloc[-1])

    # Analyze S/R
    sr_result = analyze_support_resistance(
        daily_data,
        pip_step=round_step
    )

    # Detect regimes
    daily_regime = detect_regime(daily_data)
    mtf_context = analyze_mtf_context(
        mtf_data['weekly'],
        mtf_data['daily'],
        mtf_data['4h']
    )

    # Generate trade setup
    atr = float(daily_data['atr'].iloc[-1]) if 'atr' in daily_data.columns else 0.001
    trade_setup = generate_trade_setup(
        sr_result, daily_regime, mtf_context,
        atr=atr, pip_decimal=pip_decimal
    )

    # ========== DISPLAY ==========

    # Row 1: Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label=symbol.replace("=X", ""),
            value=format_price(current_price, pip_decimal)
        )

    with col2:
        trend_emoji = daily_regime.get_trend_emoji()
        st.metric(
            label="Regime",
            value=f"{trend_emoji} {daily_regime.description}"
        )

    with col3:
        if mtf_context.alignment == "aligned_bullish":
            alignment_display = "🟢 Bullish"
        elif mtf_context.alignment == "aligned_bearish":
            alignment_display = "🔴 Bearish"
        else:
            alignment_display = "🟡 Mixed"
        st.metric(
            label="MTF Alignment",
            value=alignment_display
        )

    with col4:
        if trade_setup.bias == Bias.LONG:
            bias_display = "📈 LONG"
        elif trade_setup.bias == Bias.SHORT:
            bias_display = "📉 SHORT"
        else:
            bias_display = "⏸️ NEUTRAL"
        st.metric(
            label="Bias",
            value=bias_display
        )

    st.divider()

    # Row 2: Key Price Levels
    st.subheader("Key Price Levels")
    level_col1, level_col2 = st.columns(2)

    with level_col1:
        st.markdown("**Resistance Levels**")
        if sr_result.resistances:
            for i, zone in enumerate(sr_result.resistances[:3]):
                level_str = format_level_display(current_price, zone.center, pip_decimal)
                sources = ", ".join(zone.sources)
                st.markdown(f"**R{i+1}:** {level_str}")
                st.caption(f"Sources: {sources} | Strength: {zone.strength:.1f}")
        else:
            st.info("No resistance levels detected")

    with level_col2:
        st.markdown("**Support Levels**")
        if sr_result.supports:
            for i, zone in enumerate(sr_result.supports[:3]):
                level_str = format_level_display(current_price, zone.center, pip_decimal)
                sources = ", ".join(zone.sources)
                st.markdown(f"**S{i+1}:** {level_str}")
                st.caption(f"Sources: {sources} | Strength: {zone.strength:.1f}")
        else:
            st.info("No support levels detected")

    st.divider()

    # Row 3: Trade Setup Card
    if trade_setup.bias != Bias.NEUTRAL:
        st.subheader("Trade Setup")

        setup_col1, setup_col2, setup_col3 = st.columns(3)

        with setup_col1:
            st.markdown("**Entry & Risk**")
            st.write(f"Entry Zone: {format_price(trade_setup.entry_zone_low, pip_decimal)} - {format_price(trade_setup.entry_zone_high, pip_decimal)}")
            st.write(f"Stop Loss: {format_price(trade_setup.stop_loss, pip_decimal)} ({trade_setup.risk_pips:.0f} pips)")
            st.write(f"Invalidation: {format_price(trade_setup.invalidation, pip_decimal)}")

        with setup_col2:
            st.markdown("**Targets**")
            st.write(f"T1: {format_price(trade_setup.target_1, pip_decimal)} ({trade_setup.reward_1_pips:.0f} pips)")
            if trade_setup.target_2:
                st.write(f"T2: {format_price(trade_setup.target_2, pip_decimal)} ({trade_setup.reward_2_pips:.0f} pips)")
            if trade_setup.target_3:
                st.write(f"T3: {format_price(trade_setup.target_3, pip_decimal)} ({trade_setup.reward_3_pips:.0f} pips)")

        with setup_col3:
            st.markdown("**Risk/Reward**")
            rr_color = "green" if trade_setup.risk_reward_1 >= 1.5 else "orange" if trade_setup.risk_reward_1 >= 1.0 else "red"
            st.markdown(f"R:R to T1: **:{rr_color}[{trade_setup.risk_reward_1:.2f}]**")
            if trade_setup.risk_reward_2:
                st.write(f"R:R to T2: {trade_setup.risk_reward_2:.2f}")
            if trade_setup.risk_reward_3:
                st.write(f"R:R to T3: {trade_setup.risk_reward_3:.2f}")
            st.write(f"Confluence: {trade_setup.confluence_score:.1f}/10")

        with st.expander("Setup Reasoning"):
            for reason in trade_setup.reasoning:
                st.write(f"• {reason}")
    else:
        st.info("No actionable trade setup at current levels. " +
                (trade_setup.reasoning[0] if trade_setup.reasoning else ""))

    st.divider()

    # Row 4: Main Chart
    st.subheader("Daily Chart with S/R Zones")
    main_chart = create_main_chart(
        daily_data.tail(120),
        sr_result,
        trade_setup if trade_setup.bias != Bias.NEUTRAL else None,
        title=f"{symbol.replace('=X', '')} Daily"
    )
    st.plotly_chart(main_chart, use_container_width=True)

    # Row 5: MTF Overview
    st.subheader("Multi-Timeframe Context")
    mtf_chart = create_mtf_summary_chart(
        mtf_data['weekly'],
        mtf_data['daily'],
        mtf_data['4h'],
        symbol
    )
    st.plotly_chart(mtf_chart, use_container_width=True)

    # MTF Regime Details
    mtf_col1, mtf_col2, mtf_col3 = st.columns(3)
    with mtf_col1:
        w_emoji = mtf_context.weekly_regime.get_trend_emoji()
        st.markdown(f"**Weekly:** {w_emoji} {mtf_context.weekly_regime.description}")
    with mtf_col2:
        d_emoji = mtf_context.daily_regime.get_trend_emoji()
        st.markdown(f"**Daily:** {d_emoji} {mtf_context.daily_regime.description}")
    with mtf_col3:
        h4_emoji = mtf_context.four_hour_regime.get_trend_emoji()
        st.markdown(f"**4-Hour:** {h4_emoji} {mtf_context.four_hour_regime.description}")

    # Raw Data (optional)
    if show_raw_data:
        st.divider()
        st.subheader("Raw Data")
        st.dataframe(daily_data.tail(50))


if __name__ == "__main__":
    main()
