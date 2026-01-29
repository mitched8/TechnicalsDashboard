"""Market regime detection for FX Trading Dashboard."""

import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from config import REGIME_CONFIG, SignalFilterSettings


class TrendDirection(Enum):
    UP = "trending_up"
    DOWN = "trending_down"
    RANGING = "ranging"


class VolatilityState(Enum):
    HIGH = "high"
    LOW = "low"
    NORMAL = "normal"


@dataclass
class MarketRegime:
    """Market regime classification."""
    trend: TrendDirection
    trend_strength: float  # ADX value
    volatility: VolatilityState
    volatility_percentile: float
    atr_value: float
    description: str  # Human-readable summary

    def get_trend_emoji(self) -> str:
        """Get emoji for trend direction."""
        if self.trend == TrendDirection.UP:
            return "↑"
        elif self.trend == TrendDirection.DOWN:
            return "↓"
        else:
            return "→"


@dataclass
class MultiTimeframeContext:
    """Multi-timeframe regime analysis."""
    weekly_regime: MarketRegime
    daily_regime: MarketRegime
    four_hour_regime: MarketRegime
    alignment: str  # "aligned_bullish", "aligned_bearish", "mixed"
    alignment_score: float  # 0-1, higher = more aligned
    dominant_regime: MarketRegime  # Regime with highest ADX


def detect_regime(
    data: pd.DataFrame,
    config: dict = None
) -> MarketRegime:
    """
    Classify current market regime based on ADX and ATR.

    Trend Classification:
    - ADX > threshold AND +DI > -DI: Trending Up
    - ADX > threshold AND -DI > +DI: Trending Down
    - ADX <= threshold: Ranging

    Volatility Classification:
    - ATR in top 25%: High Volatility
    - ATR in bottom 25%: Low Volatility
    - Otherwise: Normal

    Args:
        data: DataFrame with indicators calculated
        config: Optional config dict, uses REGIME_CONFIG if None

    Returns:
        MarketRegime object
    """
    if config is None:
        config = REGIME_CONFIG

    latest = data.iloc[-1]

    # Get indicator values with fallbacks
    adx = _safe_float(latest.get('adx'), 20)
    plus_di = _safe_float(latest.get('plus_di'), 50)
    minus_di = _safe_float(latest.get('minus_di'), 50)
    atr = _safe_float(latest.get('atr'), 0)

    # Trend detection
    adx_threshold = config.get('adx_trending_threshold', 25)
    if adx > adx_threshold:
        if plus_di > minus_di:
            trend = TrendDirection.UP
        else:
            trend = TrendDirection.DOWN
    else:
        trend = TrendDirection.RANGING

    # Volatility detection
    atr_lookback = config.get('atr_lookback', 100)
    atr_history = data['atr'].tail(atr_lookback).dropna()

    if len(atr_history) > 0:
        atr_pct = (atr_history < atr).sum() / len(atr_history) * 100
    else:
        atr_pct = 50

    high_pct = config.get('atr_high_percentile', 75)
    low_pct = config.get('atr_low_percentile', 25)

    if atr_pct >= high_pct:
        volatility = VolatilityState.HIGH
    elif atr_pct <= low_pct:
        volatility = VolatilityState.LOW
    else:
        volatility = VolatilityState.NORMAL

    # Generate description
    trend_labels = {
        TrendDirection.UP: "Bullish",
        TrendDirection.DOWN: "Bearish",
        TrendDirection.RANGING: "Ranging",
    }

    vol_labels = {
        VolatilityState.HIGH: "High Vol",
        VolatilityState.LOW: "Low Vol",
        VolatilityState.NORMAL: "Normal Vol",
    }

    description = f"{trend_labels[trend]} | {vol_labels[volatility]} (ADX: {adx:.1f})"

    return MarketRegime(
        trend=trend,
        trend_strength=adx,
        volatility=volatility,
        volatility_percentile=atr_pct,
        atr_value=atr,
        description=description,
    )


def analyze_mtf_context(
    weekly_data: pd.DataFrame,
    daily_data: pd.DataFrame,
    four_hour_data: pd.DataFrame
) -> MultiTimeframeContext:
    """
    Analyze multi-timeframe alignment.

    Defers to the strongest signal (highest ADX) when timeframes conflict.

    Args:
        weekly_data: Weekly OHLCV with indicators
        daily_data: Daily OHLCV with indicators
        four_hour_data: 4-hour OHLCV with indicators

    Returns:
        MultiTimeframeContext object
    """
    weekly = detect_regime(weekly_data)
    daily = detect_regime(daily_data)
    four_hour = detect_regime(four_hour_data)

    regimes = [weekly, daily, four_hour]
    trends = [r.trend for r in regimes]

    # Count trend directions
    bullish_count = sum(1 for t in trends if t == TrendDirection.UP)
    bearish_count = sum(1 for t in trends if t == TrendDirection.DOWN)

    # Determine alignment
    if bullish_count >= 2:
        alignment = "aligned_bullish"
        alignment_score = bullish_count / 3
    elif bearish_count >= 2:
        alignment = "aligned_bearish"
        alignment_score = bearish_count / 3
    else:
        alignment = "mixed"
        alignment_score = 0.33

    # Find dominant regime (highest ADX = strongest signal)
    dominant = max(regimes, key=lambda r: r.trend_strength)

    return MultiTimeframeContext(
        weekly_regime=weekly,
        daily_regime=daily,
        four_hour_regime=four_hour,
        alignment=alignment,
        alignment_score=alignment_score,
        dominant_regime=dominant,
    )


def get_bias_from_context(context: MultiTimeframeContext) -> str:
    """
    Determine trading bias from MTF context.

    Uses the dominant regime (highest ADX) when timeframes conflict.

    Args:
        context: MultiTimeframeContext object

    Returns:
        "long", "short", or "neutral"
    """
    if context.alignment in ["aligned_bullish"]:
        return "long"
    elif context.alignment in ["aligned_bearish"]:
        return "short"
    else:
        # Mixed - defer to strongest signal
        dominant = context.dominant_regime
        if dominant.trend == TrendDirection.UP:
            return "long"
        elif dominant.trend == TrendDirection.DOWN:
            return "short"
        else:
            return "neutral"


def _safe_float(value, default: float = 0.0) -> float:
    """Convert value to float, handling NaN and None."""
    if value is None or pd.isna(value):
        return default
    return float(value)


# ============================================================================
# REGIME CHANGE FILTER FROM MACD_EMA_TREND STRATEGY
# ============================================================================

@dataclass
class RegimeChangeState:
    """
    State for tracking regime changes and filtering signals.

    From MACD_EMA_Trend strategy: After a regime change, the first opposite
    signal(s) are often false signals (whipsaws). This filter zeros out
    the first N opposite signals after a regime change.
    """
    current_regime: TrendDirection
    previous_regime: Optional[TrendDirection] = None
    regime_changed: bool = False
    bars_since_change: int = 0
    opposite_signals_since_change: int = 0
    signals_to_ignore: int = 1  # Number of opposite signals to ignore

    def update(self, new_regime: TrendDirection) -> 'RegimeChangeState':
        """Update state with new regime detection."""
        if self.current_regime != new_regime:
            # Regime changed
            self.previous_regime = self.current_regime
            self.current_regime = new_regime
            self.regime_changed = True
            self.bars_since_change = 0
            self.opposite_signals_since_change = 0
        else:
            self.bars_since_change += 1

        return self


def detect_regime_change(
    current_regime: MarketRegime,
    previous_regime: Optional[MarketRegime]
) -> bool:
    """
    Detect if a regime change has occurred.

    Args:
        current_regime: Current market regime
        previous_regime: Previous market regime

    Returns:
        True if regime changed
    """
    if previous_regime is None:
        return False

    # Check for trend direction change
    return current_regime.trend != previous_regime.trend


def should_filter_signal(
    signal_direction: str,
    regime_state: RegimeChangeState,
    filter_settings: Optional[SignalFilterSettings] = None
) -> bool:
    """
    Determine if a signal should be filtered due to regime change.

    From MACD_EMA_Trend strategy: After a regime change from bullish to bearish
    (or vice versa), the first N bearish (or bullish) signals are filtered out
    to avoid whipsaw trades during the transition period.

    Args:
        signal_direction: "bullish" or "bearish"
        regime_state: Current regime change tracking state
        filter_settings: Signal filter settings

    Returns:
        True if signal should be filtered (ignored)
    """
    if filter_settings is None:
        filter_settings = SignalFilterSettings()

    if not filter_settings.use_regime_change_filter:
        return False

    if not regime_state.regime_changed:
        return False

    signals_to_ignore = filter_settings.regime_change_ignore_signals

    # Check if this is an opposite signal
    is_opposite = False

    if regime_state.previous_regime == TrendDirection.UP and signal_direction == "bearish":
        # Changed from bullish to something else, first bearish signals may be false
        is_opposite = True
    elif regime_state.previous_regime == TrendDirection.DOWN and signal_direction == "bullish":
        # Changed from bearish to something else, first bullish signals may be false
        is_opposite = True

    if is_opposite and regime_state.opposite_signals_since_change < signals_to_ignore:
        # Increment counter so we only filter N signals, not all of them
        regime_state.opposite_signals_since_change += 1
        return True

    return False


def create_regime_state(initial_regime: TrendDirection) -> RegimeChangeState:
    """
    Create initial regime change tracking state.

    Args:
        initial_regime: Initial trend direction

    Returns:
        RegimeChangeState object
    """
    return RegimeChangeState(
        current_regime=initial_regime,
        previous_regime=None,
        regime_changed=False,
        bars_since_change=0,
        opposite_signals_since_change=0
    )


def detect_regime_history(
    data: pd.DataFrame,
    lookback: int = 20
) -> List[TrendDirection]:
    """
    Detect regime history over a lookback period.

    Useful for identifying recent regime changes.

    Args:
        data: DataFrame with indicators
        lookback: Number of bars to look back

    Returns:
        List of TrendDirection for each bar
    """
    if len(data) < lookback:
        lookback = len(data)

    regimes = []
    config = REGIME_CONFIG
    adx_threshold = config.get('adx_trending_threshold', 25)

    for i in range(len(data) - lookback, len(data)):
        row = data.iloc[i]
        adx = _safe_float(row.get('adx'), 20)
        plus_di = _safe_float(row.get('plus_di'), 50)
        minus_di = _safe_float(row.get('minus_di'), 50)

        if adx > adx_threshold:
            if plus_di > minus_di:
                regimes.append(TrendDirection.UP)
            else:
                regimes.append(TrendDirection.DOWN)
        else:
            regimes.append(TrendDirection.RANGING)

    return regimes


def get_previous_regime_before_change(
    data: pd.DataFrame,
    lookback: int = 20
) -> Optional[TrendDirection]:
    """
    Find the previous regime before the most recent regime change.

    Scans backwards through the regime history to find what the regime was
    before it changed to the current state.

    Args:
        data: DataFrame with indicators
        lookback: Number of bars to look back

    Returns:
        Previous TrendDirection before the change, or None if no change found
    """
    regimes = detect_regime_history(data, lookback)

    if len(regimes) < 2:
        return None

    current = regimes[-1]

    # Scan backwards to find the first different regime
    for i in range(len(regimes) - 2, -1, -1):
        if regimes[i] != current:
            return regimes[i]

    return None


def count_recent_regime_changes(
    data: pd.DataFrame,
    lookback: int = 20
) -> int:
    """
    Count the number of regime changes in the lookback period.

    High regime change count indicates choppy/whipsaw conditions.

    Args:
        data: DataFrame with indicators
        lookback: Number of bars to look back

    Returns:
        Number of regime changes
    """
    regimes = detect_regime_history(data, lookback)

    if len(regimes) < 2:
        return 0

    changes = 0
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i-1]:
            changes += 1

    return changes
