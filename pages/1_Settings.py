"""Strategy Settings Page - Configure trading strategy parameters."""

import streamlit as st
from config import (
    StrategySettings,
    TrendDetectionSettings,
    EntrySignalSettings,
    ExitStrategySettings,
    SignalFilterSettings,
    get_default_settings,
    get_macd_ema_trend_preset,
    get_conservative_preset
)

st.set_page_config(
    page_title="Strategy Settings",
    page_icon="Settings",
    layout="wide"
)


def init_session_state():
    """Initialize session state with default settings if not present."""
    if 'strategy_settings' not in st.session_state:
        st.session_state.strategy_settings = get_default_settings()


def save_settings(settings: StrategySettings):
    """Save settings to session state."""
    st.session_state.strategy_settings = settings


def main():
    init_session_state()

    st.title("Strategy Settings")
    st.markdown("Configure which trading strategy methods are applied on the main dashboard.")

    # Preset buttons
    st.subheader("Quick Presets")
    preset_col1, preset_col2, preset_col3 = st.columns(3)

    with preset_col1:
        if st.button("Default (Original)", use_container_width=True):
            save_settings(get_default_settings())
            st.rerun()

    with preset_col2:
        if st.button("MACD EMA Trend", use_container_width=True,
                     help="Settings matching the MACD_EMA_Trend strategy"):
            save_settings(get_macd_ema_trend_preset())
            st.rerun()

    with preset_col3:
        if st.button("Conservative (All)", use_container_width=True,
                     help="Enable all methods for maximum confluence"):
            save_settings(get_conservative_preset())
            st.rerun()

    st.divider()

    # Get current settings
    settings = st.session_state.strategy_settings

    # Create two columns for the main settings groups
    left_col, right_col = st.columns(2)

    # ==================== LEFT COLUMN ====================
    with left_col:
        # ========== TREND DETECTION ==========
        st.subheader("1. Trend Detection")
        st.caption("Methods used to identify the overall market trend direction")

        with st.container(border=True):
            # SMA Alignment
            use_sma = st.checkbox(
                "SMA Alignment",
                value=settings.trend_detection.use_sma_alignment,
                help="Use SMA 20/50/200 crossovers for trend detection"
            )
            if use_sma:
                st.caption("Bullish: Price > SMA20 > SMA50 | Bearish: Price < SMA20 < SMA50")

            st.markdown("---")

            # EMA 200 Filter
            use_ema_200 = st.checkbox(
                "EMA 200 Filter",
                value=settings.trend_detection.use_ema_200_filter,
                help="Use EMA 200 as major trend filter - only trade in direction of EMA 200"
            )
            if use_ema_200:
                st.caption("Only LONG when price > EMA 200 | Only SHORT when price < EMA 200")

            st.markdown("---")

            # Window Confirmation
            use_window = st.checkbox(
                "Window Confirmation",
                value=settings.trend_detection.use_window_confirmation,
                help="Require N consecutive bars on same side of EMA for trend confirmation"
            )
            window_size = settings.trend_detection.window_size
            if use_window:
                window_size = st.slider(
                    "Window Size (bars)",
                    min_value=3, max_value=12,
                    value=settings.trend_detection.window_size,
                    help="Number of consecutive bars required on same side of EMA 200"
                )

        # ========== EXIT STRATEGY ==========
        st.subheader("3. Exit Strategy")
        st.caption("Methods for managing stop loss and profit taking")

        with st.container(border=True):
            exit_strategy = st.radio(
                "Exit Method",
                options=["fixed_targets", "trailing_atr", "swing_based"],
                index=["fixed_targets", "trailing_atr", "swing_based"].index(
                    settings.exit_strategy.exit_strategy
                ),
                format_func=lambda x: {
                    "fixed_targets": "Fixed Targets (T1/T2/T3)",
                    "trailing_atr": "Trailing ATR Stop",
                    "swing_based": "Swing-Based Stop"
                }[x],
                help="Select primary exit strategy"
            )

            trailing_atr_mult = settings.exit_strategy.trailing_atr_multiplier
            swing_lookback = settings.exit_strategy.swing_lookback

            if exit_strategy == "fixed_targets":
                st.caption("Use predefined target levels based on S/R zones")

            elif exit_strategy == "trailing_atr":
                trailing_atr_mult = st.slider(
                    "ATR Multiplier",
                    min_value=1.0, max_value=6.0, step=0.25,
                    value=settings.exit_strategy.trailing_atr_multiplier,
                    help="Trailing stop distance = ATR x multiplier (3.75 optimal from backtest)"
                )
                st.caption("Stop trails behind price and only ratchets tighter (never loosens)")

            elif exit_strategy == "swing_based":
                swing_lookback = st.slider(
                    "Swing Lookback (bars)",
                    min_value=5, max_value=15,
                    value=settings.exit_strategy.swing_lookback,
                    help="Bars to look back for swing high/low"
                )

            # Hybrid option available for both trailing_atr and swing_based
            use_hybrid = st.checkbox(
                "Hybrid: max(swing, ATR x mult)",
                value=settings.exit_strategy.use_swing_atr_hybrid,
                help="Use the greater of swing level or ATR-based stop"
            ) if exit_strategy in ["trailing_atr", "swing_based"] else settings.exit_strategy.use_swing_atr_hybrid

    # ==================== RIGHT COLUMN ====================
    with right_col:
        # ========== ENTRY SIGNALS ==========
        st.subheader("2. Entry Signals")
        st.caption("Methods that generate buy/sell signals")

        with st.container(border=True):
            st.markdown("**MACD Signals**")

            # Standard MACD
            use_std_macd = st.checkbox(
                "Standard MACD Crossover",
                value=settings.entry_signals.use_standard_macd,
                help="Traditional MACD/Signal line crossover"
            )
            if use_std_macd:
                st.caption("Bullish: MACD crosses above Signal | Bearish: MACD crosses below Signal")

            # Pullback Resumption
            use_pullback = st.checkbox(
                "Pullback Resumption",
                value=settings.entry_signals.use_pullback_resumption,
                help="MACD cross while both lines below/above zero (pullback recovery)"
            )
            if use_pullback:
                st.caption("Long: MACD cross up while both < 0 | Short: MACD cross down while both > 0")

            # Histogram Patterns
            use_histogram = st.checkbox(
                "Histogram Patterns",
                value=settings.entry_signals.use_histogram_patterns,
                help="Detect histogram shrinking/expansion patterns"
            )
            histogram_lookback = settings.entry_signals.histogram_lookback
            if use_histogram:
                histogram_lookback = st.slider(
                    "Histogram Lookback (bars)",
                    min_value=5, max_value=12,
                    value=settings.entry_signals.histogram_lookback
                )

            st.markdown("---")
            st.markdown("**Momentum Signals**")

            # RSI
            use_rsi = st.checkbox(
                "RSI Extremes",
                value=settings.entry_signals.use_rsi_signals,
                help="RSI oversold/overbought signals"
            )
            rsi_oversold = settings.entry_signals.rsi_oversold
            rsi_overbought = settings.entry_signals.rsi_overbought
            if use_rsi:
                rsi_col1, rsi_col2 = st.columns(2)
                with rsi_col1:
                    rsi_oversold = st.number_input("Oversold", value=settings.entry_signals.rsi_oversold,
                                                   min_value=10, max_value=40)
                with rsi_col2:
                    rsi_overbought = st.number_input("Overbought", value=settings.entry_signals.rsi_overbought,
                                                     min_value=60, max_value=90)

            # Stochastic
            use_stoch = st.checkbox(
                "Stochastic Crossover",
                value=settings.entry_signals.use_stochastic_signals,
                help="Stochastic %K/%D crossovers at extremes"
            )

        # ========== SIGNAL FILTERS ==========
        st.subheader("4. Signal Filters")
        st.caption("Additional filters to reduce false signals")

        with st.container(border=True):
            # MTF Alignment
            require_mtf = st.checkbox(
                "Require MTF Alignment",
                value=settings.signal_filters.require_mtf_alignment,
                help="Only take signals when multiple timeframes agree"
            )
            min_tf = settings.signal_filters.min_aligned_timeframes
            if require_mtf:
                min_tf = st.slider(
                    "Min Aligned Timeframes",
                    min_value=2, max_value=3,
                    value=settings.signal_filters.min_aligned_timeframes
                )

            st.markdown("---")

            # Regime Change Filter
            use_regime_filter = st.checkbox(
                "Regime-Change Filter",
                value=settings.signal_filters.use_regime_change_filter,
                help="Ignore first opposite signal after regime change to avoid whipsaws"
            )
            ignore_signals = settings.signal_filters.regime_change_ignore_signals
            if use_regime_filter:
                ignore_signals = st.slider(
                    "Signals to Ignore",
                    min_value=1, max_value=3,
                    value=settings.signal_filters.regime_change_ignore_signals,
                    help="Number of opposite signals to ignore after regime change"
                )

    st.divider()

    # ========== SAVE BUTTON ==========
    if st.button("Save Settings", type="primary", use_container_width=True):
        # Build new settings from form values
        new_settings = StrategySettings(
            trend_detection=TrendDetectionSettings(
                use_sma_alignment=use_sma,
                use_ema_200_filter=use_ema_200,
                use_window_confirmation=use_window,
                window_size=window_size
            ),
            entry_signals=EntrySignalSettings(
                use_standard_macd=use_std_macd,
                use_pullback_resumption=use_pullback,
                use_histogram_patterns=use_histogram,
                histogram_lookback=histogram_lookback,
                use_rsi_signals=use_rsi,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
                use_stochastic_signals=use_stoch
            ),
            exit_strategy=ExitStrategySettings(
                exit_strategy=exit_strategy,
                use_fixed_targets=(exit_strategy == "fixed_targets"),
                trailing_atr_multiplier=trailing_atr_mult,
                swing_lookback=swing_lookback,
                use_swing_atr_hybrid=use_hybrid
            ),
            signal_filters=SignalFilterSettings(
                require_mtf_alignment=require_mtf,
                min_aligned_timeframes=min_tf,
                use_regime_change_filter=use_regime_filter,
                regime_change_ignore_signals=ignore_signals
            )
        )
        save_settings(new_settings)
        st.success("Settings saved! Return to the main dashboard to see changes.")

    # ========== CURRENT SETTINGS SUMMARY ==========
    with st.expander("View Current Settings (Debug)"):
        st.json({
            "trend_detection": {
                "use_sma_alignment": settings.trend_detection.use_sma_alignment,
                "use_ema_200_filter": settings.trend_detection.use_ema_200_filter,
                "use_window_confirmation": settings.trend_detection.use_window_confirmation,
                "window_size": settings.trend_detection.window_size
            },
            "entry_signals": {
                "use_standard_macd": settings.entry_signals.use_standard_macd,
                "use_pullback_resumption": settings.entry_signals.use_pullback_resumption,
                "use_histogram_patterns": settings.entry_signals.use_histogram_patterns,
                "use_rsi_signals": settings.entry_signals.use_rsi_signals,
                "use_stochastic_signals": settings.entry_signals.use_stochastic_signals
            },
            "exit_strategy": {
                "method": settings.exit_strategy.exit_strategy,
                "trailing_atr_multiplier": settings.exit_strategy.trailing_atr_multiplier,
                "swing_lookback": settings.exit_strategy.swing_lookback,
                "use_swing_atr_hybrid": settings.exit_strategy.use_swing_atr_hybrid
            },
            "signal_filters": {
                "require_mtf_alignment": settings.signal_filters.require_mtf_alignment,
                "use_regime_change_filter": settings.signal_filters.use_regime_change_filter
            }
        })


if __name__ == "__main__":
    main()
