"""Configuration constants for FX Trading Dashboard."""

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class CurrencyPairConfig:
    """Configuration for a currency pair."""
    symbol: str
    pip_decimal: int  # 4 for most pairs, 2 for JPY pairs
    round_number_step: float  # 0.01 for most, 1.0 for JPY

CURRENCY_PAIRS: Dict[str, CurrencyPairConfig] = {
    "EURUSD=X": CurrencyPairConfig("EURUSD=X", 4, 0.01),
    "GBPUSD=X": CurrencyPairConfig("GBPUSD=X", 4, 0.01),
    "USDJPY=X": CurrencyPairConfig("USDJPY=X", 2, 1.0),
    "AUDUSD=X": CurrencyPairConfig("AUDUSD=X", 4, 0.01),
    "USDCNH=X": CurrencyPairConfig("USDCNH=X", 4, 0.01),
}

# Timeframe configurations
TIMEFRAMES = {
    "weekly": {"period": "1y", "interval": "1wk"},
    "daily": {"period": "6mo", "interval": "1d"},
    "4h": {"period": "60d", "interval": "1h"},  # Fetch 1h, resample to 4h
}

# Support/Resistance Detection Parameters
SR_CONFIG = {
    "swing_window": 5,              # Bars on each side to confirm swing
    "volume_lookback": 100,         # Bars for volume profile analysis
    "zone_merge_threshold": 0.002,  # 0.2% to merge nearby levels
    "min_touches": 2,               # Minimum touches to confirm level
    "max_levels": 5,                # Max levels to display per direction
    "bollinger_touch_threshold": 0.0005,  # How close to band = "touch"
}

# S/R Lookback Period Options (in trading days)
SR_LOOKBACK_OPTIONS = {
    "3 Months": 65,
    "6 Months": 130,
    "1 Year": 252,
    "2 Years": 504,
    "All Available": None,
}

# Regime Detection Parameters
REGIME_CONFIG = {
    "adx_trending_threshold": 25,   # ADX above this = trending
    "atr_high_percentile": 75,      # ATR percentile for high volatility
    "atr_low_percentile": 25,       # ATR percentile for low volatility
    "atr_lookback": 100,            # Bars for ATR percentile calculation
}

# Confluence Weights for S/R Strength (basic analysis)
CONFLUENCE_WEIGHTS = {
    "swing_level": 2.0,
    "volume_level": 1.5,
    "bollinger_touch": 1.0,
    "round_number": 1.0,
    "mtf_alignment": 2.0,
    "regime_favorable": 1.5,
}

# Professional S/R Scoring Configuration (for SR Analysis page)
SR_SCORING_CONFIG = {
    # Timeframe significance (0-3 points)
    "tf_weekly": 3.0,
    "tf_daily": 2.0,
    "tf_4h": 1.0,
    "tf_1h": 0.5,

    # Structure type (0-2 points each)
    "swing_high_low": 2.0,
    "role_flip": 2.0,       # Support became resistance or vice versa
    "volume_cluster": 1.0,
    "bollinger_touch": 0.5,

    # Reaction quality (based on recent touches)
    "strong_rejection": 2.0,   # Long wick, closed away from level
    "mild_rejection": 1.0,     # Some wick, closed near level
    "acceptance": -1.0,        # Closed through level

    # Recency (how recently the level was tested)
    "recent_test": 1.0,        # Tested within 20 bars
    "medium_test": 0.5,        # Tested 20-60 bars ago
    "old_test": 0.0,           # Not tested recently

    # Touch pattern / liquidity phase (can be negative)
    "discovery_phase": 1.5,    # 1-3 quality tests - level gaining strength
    "established_phase": 1.0,  # Proven but not over-tested
    "depletion_phase": -0.5,   # Many tests, liquidity being consumed
    "exhausted_phase": -1.5,   # Very high break risk

    # Approach pattern
    "compression": -1.0,       # Higher lows into resistance = break risk
    "impulse": 0.5,            # Strong move toward level

    # Confluence bonuses
    "round_number": 1.0,       # Psychological level
    "multi_source": 0.5,       # Per additional detection source
    "mtf_confluence": 1.5,     # Level exists on multiple timeframes
}

# Trade Setup Parameters
TRADE_CONFIG = {
    "atr_buffer_multiplier": 0.5,   # Half ATR for entry zone buffer
    "min_rr_ratio": 1.5,            # Minimum R:R to show setup
}


# ============================================================================
# STRATEGY SETTINGS - User-configurable trading strategy parameters
# ============================================================================

@dataclass
class TrendDetectionSettings:
    """Settings for trend detection methods."""
    # SMA-based trend detection (original)
    use_sma_alignment: bool = True
    sma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])

    # EMA 200 filter (from MACD_EMA_Trend strategy)
    use_ema_200_filter: bool = False
    ema_period: int = 200

    # Window-based confirmation (require N bars on same side of EMA)
    use_window_confirmation: bool = False
    window_size: int = 6  # Number of consecutive bars required


@dataclass
class EntrySignalSettings:
    """Settings for entry signal generation methods."""
    # Standard MACD crossover (original)
    use_standard_macd: bool = True

    # Pullback resumption - MACD cross while both lines below/above zero (new)
    use_pullback_resumption: bool = False

    # MACD histogram pattern detection (new)
    use_histogram_patterns: bool = False
    histogram_lookback: int = 7  # Bars to analyze for histogram patterns
    histogram_threshold: float = 4e-6  # Minimum histogram change threshold

    # RSI signals (original)
    use_rsi_signals: bool = True
    rsi_oversold: int = 30
    rsi_overbought: int = 70

    # Stochastic signals (original)
    use_stochastic_signals: bool = True
    stoch_oversold: int = 20
    stoch_overbought: int = 80


@dataclass
class ExitStrategySettings:
    """Settings for exit strategy / stop loss methods."""
    # Exit strategy type: "fixed_targets", "trailing_atr", "swing_based"
    exit_strategy: str = "fixed_targets"

    # Fixed targets settings (original)
    use_fixed_targets: bool = True

    # Trailing ATR stop settings (from MACD_EMA_Trend strategy)
    trailing_atr_multiplier: float = 3.75  # Optimal from backtest
    trailing_atr_ratchet_only: bool = True  # Only tighten, never loosen

    # Swing-based stop settings (from MACD_EMA_Trend strategy)
    swing_lookback: int = 8  # Bars to look for swing high/low
    use_swing_atr_hybrid: bool = True  # max(swing, ATR * multiplier)


@dataclass
class SignalFilterSettings:
    """Settings for signal filtering methods."""
    # Multi-timeframe alignment filter (original)
    require_mtf_alignment: bool = False
    min_aligned_timeframes: int = 2  # Minimum TFs that must agree

    # Regime-change filter (from MACD_EMA_Trend strategy)
    use_regime_change_filter: bool = False
    regime_change_ignore_signals: int = 1  # Ignore first N opposite signals


@dataclass
class StrategySettings:
    """Combined strategy settings container."""
    trend_detection: TrendDetectionSettings = field(default_factory=TrendDetectionSettings)
    entry_signals: EntrySignalSettings = field(default_factory=EntrySignalSettings)
    exit_strategy: ExitStrategySettings = field(default_factory=ExitStrategySettings)
    signal_filters: SignalFilterSettings = field(default_factory=SignalFilterSettings)


def get_default_settings() -> StrategySettings:
    """Get default strategy settings."""
    return StrategySettings()


def get_macd_ema_trend_preset() -> StrategySettings:
    """Get preset matching the MACD_EMA_Trend strategy."""
    return StrategySettings(
        trend_detection=TrendDetectionSettings(
            use_sma_alignment=False,
            use_ema_200_filter=True,
            use_window_confirmation=True,
            window_size=6
        ),
        entry_signals=EntrySignalSettings(
            use_standard_macd=False,
            use_pullback_resumption=True,
            use_histogram_patterns=True,
            use_rsi_signals=False,
            use_stochastic_signals=False
        ),
        exit_strategy=ExitStrategySettings(
            exit_strategy="trailing_atr",
            use_fixed_targets=False,
            trailing_atr_multiplier=3.75
        ),
        signal_filters=SignalFilterSettings(
            require_mtf_alignment=False,
            use_regime_change_filter=True
        )
    )


def get_conservative_preset() -> StrategySettings:
    """Get conservative preset with all filters enabled."""
    return StrategySettings(
        trend_detection=TrendDetectionSettings(
            use_sma_alignment=True,
            use_ema_200_filter=True,
            use_window_confirmation=True
        ),
        entry_signals=EntrySignalSettings(
            use_standard_macd=True,
            use_pullback_resumption=True,
            use_histogram_patterns=True,
            use_rsi_signals=True,
            use_stochastic_signals=True
        ),
        exit_strategy=ExitStrategySettings(
            exit_strategy="trailing_atr",
            use_fixed_targets=True
        ),
        signal_filters=SignalFilterSettings(
            require_mtf_alignment=True,
            use_regime_change_filter=True
        )
    )
