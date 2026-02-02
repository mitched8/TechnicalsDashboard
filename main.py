"""FX Trading Dashboard - Professional Technical Analysis Tool."""

import streamlit as st
import pandas as pd

from config import CURRENCY_PAIRS, StrategySettings, get_default_settings, SR_LOOKBACK_OPTIONS
from core.data_fetcher import fetch_multi_timeframe_data, validate_data
from core.indicators import calculate_all_indicators, check_ema_200_trend_confirmed
from core.support_resistance import analyze_support_resistance
from core.regime import (
    detect_regime, analyze_mtf_context, TrendDirection,
    create_regime_state, should_filter_signal, count_recent_regime_changes,
    get_previous_regime_before_change,
    # ADX Regime Analysis
    analyze_adx, get_rsi_interpretation, ADXRegime, ADXSlope
)
from core.signals import (
    generate_signals_with_settings, generate_ema_200_signal,
    summarize_signals, Signal, SignalStrength
)
from core.trade_setup import generate_trade_setup, apply_exit_strategy, Bias
from core.volatility import (
    analyze_volatility, calculate_vol_time_series,
    VolForecast, VolRegime
)
from core.implied_vol import (
    get_latest_iv, get_all_tenors_df, merge_iv_with_price_data,
    analyze_iv_vs_rv, get_iv_time_series, IV_TENORS
)
from core.signal_explanations import (
    get_signal_explanation, get_all_explanations, SignalExplanation
)
from visualization.charts import (
    create_main_chart, create_mtf_summary_chart,
    create_main_chart_with_vol, create_vol_chart, create_vol_gauge,
    create_iv_chart, create_iv_rv_comparison_chart, create_term_structure_chart,
    create_spot_iv_rv_aligned_chart, create_vol_with_ta_events_chart
)
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

    if settings.trend_detection.use_ema_200_filter:
        active.append("EMA 200")
    if settings.trend_detection.use_window_confirmation:
        active.append(f"Window({settings.trend_detection.window_size})")
    if settings.trend_detection.use_sma_alignment:
        active.append("SMA")
    if settings.entry_signals.use_pullback_resumption:
        active.append("Pullback")
    if settings.entry_signals.use_histogram_patterns:
        active.append("Histogram")
    if settings.exit_strategy.exit_strategy == "trailing_atr":
        active.append(f"Trail ATR({settings.exit_strategy.trailing_atr_multiplier})")
    elif settings.exit_strategy.exit_strategy == "swing_based":
        active.append("Swing Stop")
    if settings.signal_filters.use_regime_change_filter:
        active.append("Regime Filter")
    if settings.signal_filters.require_mtf_alignment:
        active.append("MTF Required")

    if active:
        st.caption(f"Active: {', '.join(active)}")
    else:
        st.caption("Using default settings")


def display_signal_explanation(explanation: SignalExplanation):
    """Display a formatted signal explanation."""
    st.markdown(f"### {explanation.name}")
    st.markdown(f"**Category:** {explanation.category.title()}")

    st.markdown("**What It Is:**")
    st.write(explanation.what_it_is)

    st.markdown("**Calculation:**")
    st.code(explanation.calculation, language=None)

    st.markdown("**Professional Use:**")
    st.write(explanation.professional_use)

    st.markdown("**Signal Interpretation:**")
    st.info(explanation.signal_interpretation)


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

        st.subheader("Strategy")
        display_active_settings_summary(settings)
        st.page_link("pages/1_Settings.py", label="Configure Strategy", icon="⚙️")

        st.divider()

        st.subheader("Display")
        show_vol_analysis = st.checkbox("Show Vol Analysis", value=True)
        show_trade_setup = st.checkbox("Show Trade Setup", value=True)
        show_raw_data = st.checkbox("Show Raw Data", value=False)

        st.divider()

        # S/R Configuration
        st.subheader("S/R Levels")
        sr_lookback_label = st.selectbox(
            "S/R Lookback Period",
            options=list(SR_LOOKBACK_OPTIONS.keys()),
            index=2,  # Default to 1 Year
            help="Longer periods find more significant historical levels"
        )
        sr_lookback = SR_LOOKBACK_OPTIONS[sr_lookback_label]

        sr_min_strength = st.slider(
            "Min Strength",
            min_value=0.0, max_value=5.0, value=1.5, step=0.5,
            help="Filter out weak levels (higher = stronger levels only)"
        )

        sr_min_age = st.slider(
            "Min Age (days)",
            min_value=0, max_value=60, value=0, step=5,
            help="Filter out recent levels (higher = older levels only)"
        )

        if show_vol_analysis:
            st.divider()
            st.subheader("Volatility")
            selected_tenor = st.selectbox(
                "IV Tenor",
                options=IV_TENORS,
                index=1,
                help="Select implied vol tenor for comparison"
            )
        else:
            selected_tenor = '1M'

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

    # Analyze S/R with configurable lookback
    sr_result = analyze_support_resistance(
        daily_data,
        pip_step=round_step,
        sr_lookback=sr_lookback,
        min_strength=sr_min_strength,
        min_age_bars=sr_min_age
    )

    # Detect regimes
    daily_regime = detect_regime(daily_data)
    mtf_context = analyze_mtf_context(
        mtf_data['weekly'],
        mtf_data['daily'],
        mtf_data['4h']
    )

    # ADX Regime Analysis (professional trading framework)
    adx_analysis = analyze_adx(daily_data)

    # Generate signals
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
        'adx': daily_indicators.adx if daily_indicators else 25,
        'plus_di': daily_indicators.plus_di if daily_indicators else 50,
        'minus_di': daily_indicators.minus_di if daily_indicators else 50,
    }

    signals = generate_signals_with_settings(
        df=daily_data,
        close=current_price,
        indicators=indicator_dict,
        settings=settings.entry_signals
    )

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

    # Apply regime-change filter
    filtered_signals = []
    regime_state = create_regime_state(daily_regime.trend)

    if settings.signal_filters.use_regime_change_filter:
        regime_changes = count_recent_regime_changes(daily_data, lookback=20)
        if regime_changes > 0:
            regime_state.regime_changed = True
            previous_regime = get_previous_regime_before_change(daily_data, lookback=20)
            if previous_regime:
                regime_state.previous_regime = previous_regime

        for signal in signals:
            if not should_filter_signal(signal.direction, regime_state, settings.signal_filters):
                filtered_signals.append(signal)
            else:
                signal.description += " [FILTERED - regime change]"
                filtered_signals.append(signal)
    else:
        filtered_signals = signals

    signal_summary = summarize_signals(filtered_signals)

    # Volatility Analysis
    vol_analysis = analyze_volatility(daily_data, indicator_dict)
    daily_data_with_vol = calculate_vol_time_series(daily_data)

    # Implied Vol Data
    iv_data = get_latest_iv(symbol)
    iv_time_series_df = get_all_tenors_df(symbol)
    iv_analysis = None

    if iv_data and vol_analysis:
        daily_data_with_vol = merge_iv_with_price_data(daily_data_with_vol, symbol)
        iv_history = get_iv_time_series(symbol, selected_tenor)
        iv_analysis = analyze_iv_vs_rv(
            iv_data=iv_data,
            rv_1m=vol_analysis.metrics.rv_1m,
            selected_tenor=selected_tenor,
            iv_history=iv_history
        )

    # Generate trade setup
    atr = float(daily_data['atr'].iloc[-1]) if 'atr' in daily_data.columns else 0.001
    trade_setup = generate_trade_setup(
        sr_result, daily_regime, mtf_context,
        atr=atr, pip_decimal=pip_decimal
    )
    trade_setup = apply_exit_strategy(
        setup=trade_setup,
        df=daily_data,
        atr=atr,
        exit_settings=settings.exit_strategy,
        pip_decimal=pip_decimal
    )

    # ========== DISPLAY ==========

    # Row 1: Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label=symbol.replace("=X", ""),
            value=format_price(current_price, pip_decimal)
        )
        if settings.trend_detection.use_ema_200_filter and daily_indicators:
            ema_pos = "above" if current_price > daily_indicators.ema_200 else "below"
            st.caption(f"EMA 200: {ema_pos} ({daily_indicators.ema_200_trend})")

    with col2:
        trend_emoji = daily_regime.get_trend_emoji()
        st.metric(label="Regime", value=f"{trend_emoji} {daily_regime.description}")

    with col3:
        if mtf_context.alignment == "aligned_bullish":
            alignment_display = "🟢 Bullish"
        elif mtf_context.alignment == "aligned_bearish":
            alignment_display = "🔴 Bearish"
        else:
            alignment_display = "🟡 Mixed"
        st.metric(label="MTF Alignment", value=alignment_display)

    with col4:
        if trade_setup.bias == Bias.LONG:
            bias_display = "📈 LONG"
        elif trade_setup.bias == Bias.SHORT:
            bias_display = "📉 SHORT"
        else:
            bias_display = "⏸️ NEUTRAL"
        st.metric(label="Bias", value=bias_display)

    st.divider()

    # ========== ADX REGIME ANALYSIS (Professional Trading Framework) ==========
    st.header("ADX Trend Regime Analysis")
    st.caption("ADX is a 'style selector' - tells you HOW to trade, not WHAT to trade")

    # ADX Metrics Row
    adx_col1, adx_col2, adx_col3, adx_col4 = st.columns(4)

    with adx_col1:
        # ADX Value with slope indicator
        slope_arrow = adx_analysis.slope_emoji
        st.metric(
            label="ADX Value",
            value=f"{adx_analysis.adx:.1f} {slope_arrow}",
            delta=f"{adx_analysis.slope_value:+.1f}/bar" if adx_analysis.slope_value != 0 else "flat",
            delta_color="normal" if adx_analysis.slope == ADXSlope.RISING else "inverse" if adx_analysis.slope == ADXSlope.FALLING else "off"
        )

    with adx_col2:
        # Regime classification
        regime_labels = {
            ADXRegime.RANGE_MEAN_REVERSION: "Range (Mean Revert)",
            ADXRegime.TRANSITION_BREAKOUT: "Transition (Breakout)",
            ADXRegime.TREND_CONTINUATION: "Trend (Continuation)",
        }
        regime_colors = {
            ADXRegime.RANGE_MEAN_REVERSION: "blue",
            ADXRegime.TRANSITION_BREAKOUT: "orange",
            ADXRegime.TREND_CONTINUATION: "green",
        }
        regime_label = regime_labels.get(adx_analysis.regime, "Unknown")
        regime_color = regime_colors.get(adx_analysis.regime, "gray")
        st.metric(label="Trading Regime", value=f"{adx_analysis.regime_emoji} {regime_label}")

    with adx_col3:
        # Direction from DI
        di_diff = adx_analysis.plus_di - adx_analysis.minus_di
        if adx_analysis.di_direction == "bullish":
            di_display = f"+DI > -DI ({di_diff:+.1f})"
            di_emoji = "📈"
        elif adx_analysis.di_direction == "bearish":
            di_display = f"-DI > +DI ({di_diff:+.1f})"
            di_emoji = "📉"
        else:
            di_display = "Neutral"
            di_emoji = "↔"
        st.metric(label="Directional Index", value=f"{di_emoji} {di_display}")

    with adx_col4:
        # S/R Behavior prediction
        sr_emoji = {"wall": "🧱", "liquidity_target": "🎯", "transitioning": "⚡"}.get(adx_analysis.sr_behavior, "❓")
        sr_labels = {"wall": "Wall (Hold)", "liquidity_target": "Break Target", "transitioning": "Uncertain"}.get(adx_analysis.sr_behavior, "Unknown")
        st.metric(label="S/R Behavior", value=f"{sr_emoji} {sr_labels}")

    # Trading Style Recommendation
    st.subheader(f"Trading Style: {adx_analysis.trading_style}")

    style_col1, style_col2 = st.columns([2, 1])

    with style_col1:
        st.markdown("**Current conditions favor:**")
        for detail in adx_analysis.style_details[:4]:  # Show first 4 recommendations
            if "WARNING" in detail or "Don't" in detail:
                st.warning(f"• {detail}")
            else:
                st.write(f"• {detail}")

    with style_col2:
        # RSI interpretation based on ADX regime
        rsi_value = daily_indicators.rsi if daily_indicators else 50
        rsi_interp, rsi_detail = get_rsi_interpretation(rsi_value, adx_analysis)
        st.markdown("**RSI in this regime:**")
        st.write(f"RSI: {rsi_value:.1f}")

        rsi_color = {"bullish": "green", "bearish": "red", "neutral": "gray", "warning": "orange", "caution": "yellow"}.get(rsi_interp, "gray")
        st.markdown(f":{rsi_color}[{rsi_detail}]")

    # Compression breakout alert
    if adx_analysis.is_compression_breakout:
        st.info(f"COMPRESSION BREAKOUT: ADX rising from recent low of {adx_analysis.recent_low:.1f} - trend regime 'turning on'")

    # S/R behavior explanation
    with st.expander("Understanding S/R Behavior in Current Regime"):
        st.markdown(f"**{adx_analysis.sr_behavior_reason}**")
        st.markdown("""
        Key insight from professional trading:
        - **Low ADX (Range)**: S/R levels act as walls - fades and bounces are reliable
        - **Rising ADX (Transition)**: Watch for breakout acceptance before committing
        - **High ADX (Trend)**: S/R levels are liquidity targets - breaks more likely

        This answers: *"Is this level likely to reject... or get eaten through?"*
        """)

    st.divider()

    # ========== PRIMARY SECTION: TECHNICAL ANALYSIS SIGNALS ==========
    st.header("Technical Analysis Signals")

    # Signal Summary Row
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        bullish_count = signal_summary['bullish']
        st.metric("Bullish Signals", bullish_count,
                  delta="Strong" if bullish_count >= 3 else None,
                  delta_color="normal" if bullish_count >= 3 else "off")

    with summary_col2:
        bearish_count = signal_summary['bearish']
        st.metric("Bearish Signals", bearish_count,
                  delta="Strong" if bearish_count >= 3 else None,
                  delta_color="inverse" if bearish_count >= 3 else "off")

    with summary_col3:
        st.metric("Neutral", signal_summary['neutral'])

    with summary_col4:
        net_signal = bullish_count - bearish_count
        if net_signal > 0:
            st.metric("Net Bias", f"+{net_signal} Bullish", delta_color="normal")
        elif net_signal < 0:
            st.metric("Net Bias", f"{net_signal} Bearish", delta_color="inverse")
        else:
            st.metric("Net Bias", "Neutral")

    # Active Signals Display
    st.subheader("Active Signals")

    if not filtered_signals:
        st.info("No active signals at current levels.")
    else:
        # Group signals by direction
        bullish_signals = [s for s in filtered_signals if s.direction == "bullish"]
        bearish_signals = [s for s in filtered_signals if s.direction == "bearish"]
        neutral_signals = [s for s in filtered_signals if s.direction == "neutral"]

        sig_col1, sig_col2 = st.columns(2)

        with sig_col1:
            st.markdown("**Bullish Signals**")
            if bullish_signals:
                for signal in bullish_signals:
                    strength_badge = f"[{signal.strength.value.upper()}]"
                    filtered_marker = " ⚠️" if "[FILTERED" in signal.description else ""
                    st.markdown(f"🟢 **{signal.name}** {strength_badge}{filtered_marker}")
                    st.caption(signal.description)
            else:
                st.caption("No bullish signals")

        with sig_col2:
            st.markdown("**Bearish Signals**")
            if bearish_signals:
                for signal in bearish_signals:
                    strength_badge = f"[{signal.strength.value.upper()}]"
                    filtered_marker = " ⚠️" if "[FILTERED" in signal.description else ""
                    st.markdown(f"🔴 **{signal.name}** {strength_badge}{filtered_marker}")
                    st.caption(signal.description)
            else:
                st.caption("No bearish signals")

        if neutral_signals:
            st.markdown("**Neutral/Confirming Signals**")
            for signal in neutral_signals:
                st.markdown(f"⚪ **{signal.name}**: {signal.description}")

    # TA Signal Explanations (Collapsible)
    with st.expander("📚 Technical Indicator Reference Guide", expanded=False):
        st.markdown("""
        This section explains each technical analysis indicator used in this dashboard,
        including how it's calculated and how professional traders interpret the signals.
        """)

        explanations = get_all_explanations()
        tabs = st.tabs(["Momentum", "Trend", "Volatility", "S/R"])

        with tabs[0]:  # Momentum
            for name, exp in explanations.items():
                if exp.category == "momentum":
                    with st.expander(exp.name):
                        display_signal_explanation(exp)

        with tabs[1]:  # Trend
            for name, exp in explanations.items():
                if exp.category == "trend":
                    with st.expander(exp.name):
                        display_signal_explanation(exp)

        with tabs[2]:  # Volatility
            for name, exp in explanations.items():
                if exp.category == "volatility":
                    with st.expander(exp.name):
                        display_signal_explanation(exp)

        with tabs[3]:  # Support/Resistance
            for name, exp in explanations.items():
                if exp.category == "support_resistance":
                    with st.expander(exp.name):
                        display_signal_explanation(exp)

    st.divider()

    # ========== PRICE CHART WITH TA ==========
    st.subheader("Price Chart with Technical Analysis")

    main_chart = create_main_chart(
        daily_data.tail(120),
        sr_result,
        trade_setup if trade_setup.bias != Bias.NEUTRAL else None,
        title=f"{symbol.replace('=X', '')} Daily"
    )
    st.plotly_chart(main_chart, use_container_width=True)

    # Support/Resistance Levels
    st.subheader("Support & Resistance Levels")
    st.caption(f"Lookback: {sr_lookback_label} | Min Strength: {sr_min_strength} | Min Age: {sr_min_age}d")

    level_col1, level_col2 = st.columns(2)

    with level_col1:
        st.markdown("**Resistance Levels**")
        if sr_result.resistances:
            for i, zone in enumerate(sr_result.resistances[:5]):
                level_str = format_level_display(current_price, zone.center, pip_decimal)
                sources = ", ".join(zone.sources)

                # Age indicator
                age_emoji = "🔴" if zone.age_bars < 20 else "🟡" if zone.age_bars < 60 else "🟢"

                st.markdown(f"**R{i+1}:** {level_str} {age_emoji}")
                st.caption(
                    f"Strength: {zone.strength_category} ({zone.strength:.1f}) | "
                    f"Age: {zone.age_category} ({zone.age_bars}d) | "
                    f"Touches: {zone.touches}"
                )
        else:
            st.info("No resistance levels detected")

    with level_col2:
        st.markdown("**Support Levels**")
        if sr_result.supports:
            for i, zone in enumerate(sr_result.supports[:5]):
                level_str = format_level_display(current_price, zone.center, pip_decimal)
                sources = ", ".join(zone.sources)

                # Age indicator
                age_emoji = "🔴" if zone.age_bars < 20 else "🟡" if zone.age_bars < 60 else "🟢"

                st.markdown(f"**S{i+1}:** {level_str} {age_emoji}")
                st.caption(
                    f"Strength: {zone.strength_category} ({zone.strength:.1f}) | "
                    f"Age: {zone.age_category} ({zone.age_bars}d) | "
                    f"Touches: {zone.touches}"
                )
        else:
            st.info("No support levels detected")

    # Legend for age indicators
    with st.expander("S/R Level Legend"):
        st.markdown("""
        **Age Indicators:**
        - 🔴 Recent (< 20 days) - May be noise
        - 🟡 Medium (20-60 days) - Developing level
        - 🟢 Established (60+ days) - More significant

        **Strength Categories:**
        - Very Strong (5.0+): Multiple confluent sources
        - Strong (3.5-5.0): Good confluence
        - Moderate (2.0-3.5): Some confluence
        - Weak (< 2.0): Single source or few touches

        **Sources:** swing_high/low, volume, bb_upper/lower, round numbers
        """)

    # ========== TRADE SETUP (if enabled) ==========
    if show_trade_setup:
        st.divider()
        st.subheader("Trade Setup")

        if trade_setup.bias != Bias.NEUTRAL:
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
                st.write(f"Confluence: {trade_setup.confluence_score:.1f}/10")

            with st.expander("Setup Reasoning"):
                for reason in trade_setup.reasoning:
                    st.write(f"• {reason}")
        else:
            st.info("No actionable trade setup at current levels. " +
                    (trade_setup.reasoning[0] if trade_setup.reasoning else ""))

    # ========== VOLATILITY ANALYSIS (Linked to TA) ==========
    if show_vol_analysis:
        st.divider()
        st.header("Volatility Analysis")
        st.caption("How TA signals relate to implied and realized volatility movements")

        # Vol Metrics Summary
        vol_col1, vol_col2, vol_col3 = st.columns(3)

        with vol_col1:
            st.markdown("**Realized Vol**")
            st.metric("1M RV", f"{vol_analysis.metrics.rv_1m:.2f}%",
                      delta=f"{vol_analysis.metrics.vol_change_5d:+.1f}% (5d)")
            st.caption(f"Percentile: {vol_analysis.metrics.rv_percentile:.0f}th")

        with vol_col2:
            st.markdown("**Implied Vol**")
            if iv_analysis:
                st.metric(f"IV {selected_tenor}", f"{iv_analysis.current_iv:.2f}%")
                spread = iv_analysis.iv_rv_spread
                spread_label = "Expensive" if spread > 2 else "Cheap" if spread < -2 else "Fair"
                st.caption(f"IV-RV Spread: {spread:+.2f}% ({spread_label})")
            else:
                st.caption("IV data not available")

        with vol_col3:
            st.markdown("**Vol Forecast**")
            forecast_label = vol_analysis.forecast.value.upper()
            if vol_analysis.forecast == VolForecast.EXPANDING:
                st.success(f"📈 {forecast_label}")
            elif vol_analysis.forecast == VolForecast.COMPRESSING:
                st.error(f"📉 {forecast_label}")
            else:
                st.info(f"➡️ {forecast_label}")
            st.caption(vol_analysis.strategy_suggestion)

        # Chart 1: Spot + IV + RV Aligned
        st.subheader("Spot Price with Volatility")
        st.caption("Spot price movement with implied and realized volatility aligned on time axis")

        aligned_chart = create_spot_iv_rv_aligned_chart(
            daily_data_with_vol.tail(180),
            iv_tenor=selected_tenor,
            title=f"{symbol.replace('=X', '')} - Spot with Vol Overlay"
        )
        st.plotly_chart(aligned_chart, use_container_width=True)

        # Chart 2: IV/RV with TA Event Markers
        st.subheader("Volatility at TA Signal Events")
        st.caption("How volatility behaved at technical signal points (support/resistance, MACD crossovers)")

        # Create TA events list from current signals (simulated historical events)
        ta_events = []
        current_date = daily_data.index[-1]

        # Add current signals as events
        for signal in filtered_signals:
            if "[FILTERED" not in signal.description:
                ta_events.append({
                    'date': current_date,
                    'type': signal.name,
                    'direction': signal.direction,
                    'description': f"{signal.name}: {signal.description[:50]}..."
                })

        vol_events_chart = create_vol_with_ta_events_chart(
            daily_data_with_vol.tail(180),
            ta_events=ta_events,
            iv_tenor=selected_tenor,
            title=f"{symbol.replace('=X', '')} - Vol with TA Events"
        )
        st.plotly_chart(vol_events_chart, use_container_width=True)

        # Term Structure (if IV data available)
        if iv_data:
            with st.expander("IV Term Structure"):
                term_chart = create_term_structure_chart(
                    iv_data.get_term_structure(),
                    current_rv=vol_analysis.metrics.rv_1m,
                    title=f"{symbol.replace('=X', '')} Vol Term Structure"
                )
                st.plotly_chart(term_chart, use_container_width=True)

    # ========== MTF CONTEXT ==========
    st.divider()
    st.subheader("Multi-Timeframe Context")

    mtf_chart = create_mtf_summary_chart(
        mtf_data['weekly'],
        mtf_data['daily'],
        mtf_data['4h'],
        symbol
    )
    st.plotly_chart(mtf_chart, use_container_width=True)

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

    # Raw Data
    if show_raw_data:
        st.divider()
        st.subheader("Raw Data")
        st.dataframe(daily_data.tail(50))


if __name__ == "__main__":
    main()
