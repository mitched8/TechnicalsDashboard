"""FX Trading Dashboard - Professional Technical Analysis Tool."""

import streamlit as st
import pandas as pd

from config import CURRENCY_PAIRS, StrategySettings, get_default_settings
from core.data_fetcher import fetch_multi_timeframe_data, validate_data
from core.indicators import calculate_all_indicators, check_ema_200_trend_confirmed
from core.support_resistance import analyze_support_resistance
from core.regime import (
    detect_regime, analyze_mtf_context, TrendDirection,
    create_regime_state, should_filter_signal, count_recent_regime_changes
)
from core.signals import (
    generate_signals_with_settings, generate_ema_200_signal,
    summarize_signals, Signal, SignalStrength
)
from core.trade_setup import generate_trade_setup, apply_exit_strategy, Bias
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


def init_session_state():
    """Initialize session state with default settings if not present."""
    if 'strategy_settings' not in st.session_state:
        st.session_state.strategy_settings = get_default_settings()


def get_settings() -> StrategySettings:
    """Get current strategy settings from session state."""
    init_session_state()
    return st.session_state.strategy_settings


def display_active_settings_summary(settings: StrategySettings):
    """Display a compact summary of active settings."""
    active = []

    # Trend detection
    if settings.trend_detection.use_ema_200_filter:
        active.append("EMA 200")
    if settings.trend_detection.use_window_confirmation:
        active.append(f"Window({settings.trend_detection.window_size})")
    if settings.trend_detection.use_sma_alignment:
        active.append("SMA")

    # Entry signals
    if settings.entry_signals.use_pullback_resumption:
        active.append("Pullback")
    if settings.entry_signals.use_histogram_patterns:
        active.append("Histogram")

    # Exit strategy
    if settings.exit_strategy.exit_strategy == "trailing_atr":
        active.append(f"Trail ATR({settings.exit_strategy.trailing_atr_multiplier})")
    elif settings.exit_strategy.exit_strategy == "swing_based":
        active.append("Swing Stop")

    # Filters
    if settings.signal_filters.use_regime_change_filter:
        active.append("Regime Filter")
    if settings.signal_filters.require_mtf_alignment:
        active.append("MTF Required")

    if active:
        st.caption(f"Active: {', '.join(active)}")
    else:
        st.caption("Using default settings")


def main():
    init_session_state()
    settings = get_settings()

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

        # Strategy settings summary
        st.subheader("Strategy")
        display_active_settings_summary(settings)
        st.page_link("pages/1_Settings.py", label="Configure Strategy", icon="Settings")

        st.divider()
        show_raw_data = st.checkbox("Show Raw Data", value=False)
        show_signals = st.checkbox("Show Signal Details", value=False)

    # Get pip configuration
    pip_decimal, round_step = get_pip_info(symbol)

    # Fetch multi-timeframe data
    with st.spinner("Fetching market data..."):
        mtf_data = fetch_multi_timeframe_data(symbol)

    if not validate_data(mtf_data.get('daily', pd.DataFrame())):
        st.error("Failed to fetch data. Please try again.")
        return

    # Calculate indicators for each timeframe
    indicator_values = {}
    for tf in ['weekly', 'daily', '4h']:
        if not mtf_data[tf].empty:
            mtf_data[tf], indicator_values[tf] = calculate_all_indicators(mtf_data[tf])

    daily_data = mtf_data['daily']
    daily_indicators = indicator_values.get('daily')
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

    # ========== NEW: Generate signals based on settings ==========
    indicator_dict = {
        'rsi': daily_indicators.rsi if daily_indicators else 50,
        'macd': daily_indicators.macd if daily_indicators else 0,
        'macd_signal': daily_indicators.macd_signal if daily_indicators else 0,
        'macd_histogram': daily_indicators.macd_histogram if daily_indicators else 0,
        'sma_20': daily_indicators.sma_20 if daily_indicators else current_price,
        'sma_50': daily_indicators.sma_50 if daily_indicators else current_price,
        'sma_200': daily_indicators.sma_200 if daily_indicators else current_price,
        'ema_200': daily_indicators.ema_200 if daily_indicators else current_price,
        'stoch_k': daily_indicators.stoch_k if daily_indicators else 50,
        'stoch_d': daily_indicators.stoch_d if daily_indicators else 50,
    }

    # Generate signals with settings
    signals = generate_signals_with_settings(
        df=daily_data,
        close=current_price,
        indicators=indicator_dict,
        settings=settings.entry_signals
    )

    # Add EMA 200 signal if enabled
    if settings.trend_detection.use_ema_200_filter and daily_indicators:
        ema_signal = generate_ema_200_signal(
            close=current_price,
            ema_200=daily_indicators.ema_200,
            ema_200_trend=daily_indicators.ema_200_trend,
            bars_confirmed=max(daily_indicators.ema_200_bars_above, daily_indicators.ema_200_bars_below),
            window_required=settings.trend_detection.window_size
        )
        if ema_signal:
            signals.append(ema_signal)

    # Apply regime-change filter if enabled
    filtered_signals = []
    regime_state = create_regime_state(daily_regime.trend)

    if settings.signal_filters.use_regime_change_filter:
        # Check for recent regime changes
        regime_changes = count_recent_regime_changes(daily_data, lookback=20)
        if regime_changes > 0:
            regime_state.regime_changed = True

        for signal in signals:
            if not should_filter_signal(signal.direction, regime_state, settings.signal_filters):
                filtered_signals.append(signal)
            else:
                # Track filtered signal for display
                signal.description += " [FILTERED - regime change]"
                filtered_signals.append(signal)
    else:
        filtered_signals = signals

    # Summarize signals
    signal_summary = summarize_signals(filtered_signals)

    # ========== Generate trade setup ==========
    atr = float(daily_data['atr'].iloc[-1]) if 'atr' in daily_data.columns else 0.001
    trade_setup = generate_trade_setup(
        sr_result, daily_regime, mtf_context,
        atr=atr, pip_decimal=pip_decimal
    )

    # Apply exit strategy modifications
    trade_setup = apply_exit_strategy(
        setup=trade_setup,
        df=daily_data,
        atr=atr,
        exit_settings=settings.exit_strategy,
        pip_decimal=pip_decimal
    )

    # Check EMA 200 filter for trade validity
    ema_200_warning = None
    if settings.trend_detection.use_ema_200_filter and daily_indicators:
        if trade_setup.bias == Bias.LONG and daily_indicators.ema_200_trend != "bullish":
            ema_200_warning = "EMA 200 filter: Price not confirmed above EMA 200"
        elif trade_setup.bias == Bias.SHORT and daily_indicators.ema_200_trend != "bearish":
            ema_200_warning = "EMA 200 filter: Price not confirmed below EMA 200"

    # Check MTF alignment filter
    mtf_warning = None
    if settings.signal_filters.require_mtf_alignment:
        if mtf_context.alignment == "mixed":
            mtf_warning = "MTF filter: Timeframes not aligned"

    # ========== DISPLAY ==========

    # Row 1: Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label=symbol.replace("=X", ""),
            value=format_price(current_price, pip_decimal)
        )
        # Show EMA 200 position if enabled
        if settings.trend_detection.use_ema_200_filter and daily_indicators:
            ema_pos = "above" if current_price > daily_indicators.ema_200 else "below"
            st.caption(f"EMA 200: {ema_pos} ({daily_indicators.ema_200_trend})")

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

    # Show filter warnings
    if ema_200_warning or mtf_warning:
        with st.container():
            if ema_200_warning:
                st.warning(ema_200_warning)
            if mtf_warning:
                st.warning(mtf_warning)

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

    # Signal Details (optional)
    if show_signals:
        st.divider()
        st.subheader("Signal Analysis")

        sig_col1, sig_col2, sig_col3 = st.columns(3)

        with sig_col1:
            st.markdown(f"**Bullish Signals:** {signal_summary['bullish']}")
        with sig_col2:
            st.markdown(f"**Bearish Signals:** {signal_summary['bearish']}")
        with sig_col3:
            st.markdown(f"**Neutral:** {signal_summary['neutral']}")

        with st.expander("Signal Details", expanded=True):
            for signal in filtered_signals:
                if signal.direction == "bullish":
                    icon = "🟢"
                elif signal.direction == "bearish":
                    icon = "🔴"
                else:
                    icon = "⚪"

                strength_label = signal.strength.value.upper()
                filtered_marker = " ⚠️" if "[FILTERED" in signal.description else ""

                st.markdown(
                    f"{icon} **{signal.name}** ({strength_label}){filtered_marker}: "
                    f"{signal.description}"
                )

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
