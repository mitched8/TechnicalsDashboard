"""Market regime detection for FX Trading Dashboard."""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from config import REGIME_CONFIG, SignalFilterSettings


class TrendDirection(Enum):
    UP = "trending_up"
    DOWN = "trending_down"
    RANGING = "ranging"


class VolatilityState(Enum):
    HIGH = "high"
    LOW = "low"
    NORMAL = "normal"


class ADXSlope(Enum):
    """ADX slope direction - more important than absolute level."""
    RISING = "rising"      # Trend conditions improving
    FALLING = "falling"    # Trend conditions degrading
    FLAT = "flat"          # No significant change


class ADXRegime(Enum):
    """
    ADX-based trading regime framework.

    From professional FX trading: ADX is a "style selector" that tells you
    HOW to trade, not WHAT to trade. These regimes determine which strategies
    have a tailwind.
    """
    RANGE_MEAN_REVERSION = "range"       # ADX low and flat/falling - fade extremes
    TRANSITION_BREAKOUT = "transition"    # ADX low but turning up - breakout risk
    TREND_CONTINUATION = "trend"          # ADX rising and elevated - continuation plays


@dataclass
class ADXAnalysis:
    """
    Comprehensive ADX analysis for regime-based trading decisions.

    Key insight from professional trading: "ADX slope > ADX level"
    A rising ADX from 18 to 25 is more significant than a static ADX at 30.
    """
    # Current values
    adx: float
    plus_di: float
    minus_di: float

    # Slope analysis (the most important aspect)
    slope: ADXSlope
    slope_value: float  # Rate of change

    # Regime classification
    regime: ADXRegime

    # Historical context
    bars_since_low: int = 0      # Bars since ADX was at local minimum
    recent_low: float = 0.0      # Recent low ADX value
    is_compression_breakout: bool = False  # ADX was low for a while, now rising

    # Trading style recommendations
    trading_style: str = ""
    style_details: List[str] = field(default_factory=list)

    # S/R behavior prediction
    sr_behavior: str = ""  # "wall" or "liquidity_target"
    sr_behavior_reason: str = ""

    @property
    def regime_emoji(self) -> str:
        """Get emoji for regime."""
        if self.regime == ADXRegime.RANGE_MEAN_REVERSION:
            return "↔"
        elif self.regime == ADXRegime.TRANSITION_BREAKOUT:
            return "⚡"
        else:
            return "→→"

    @property
    def slope_emoji(self) -> str:
        """Get emoji for slope."""
        if self.slope == ADXSlope.RISING:
            return "📈"
        elif self.slope == ADXSlope.FALLING:
            return "📉"
        else:
            return "➡"

    @property
    def di_direction(self) -> str:
        """Get trend direction from DI."""
        if self.plus_di > self.minus_di:
            return "bullish"
        elif self.minus_di > self.plus_di:
            return "bearish"
        else:
            return "neutral"


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
# ADX REGIME ANALYSIS - Professional Trading Framework
# ============================================================================

def analyze_adx(
    data: pd.DataFrame,
    lookback: int = 14,
    slope_period: int = 5,
    config: dict = None
) -> ADXAnalysis:
    """
    Comprehensive ADX analysis implementing professional trading concepts.

    Key insights from quant FX trading:
    1. ADX slope is MORE important than ADX level
    2. ADX is a "style selector" - tells you HOW to trade, not WHAT
    3. ADX determines if S/R acts as wall (low) or liquidity target (high)

    Args:
        data: DataFrame with ADX, plus_di, minus_di columns
        lookback: Bars to analyze for compression detection
        slope_period: Bars to calculate slope
        config: Optional config dict

    Returns:
        ADXAnalysis object with regime classification and trading guidance
    """
    if config is None:
        config = REGIME_CONFIG

    if len(data) < slope_period + 1:
        return _default_adx_analysis()

    # Get current and historical ADX values
    adx_series = data['adx'].dropna()
    if len(adx_series) < slope_period + 1:
        return _default_adx_analysis()

    current_adx = _safe_float(adx_series.iloc[-1], 20)
    plus_di = _safe_float(data['plus_di'].iloc[-1], 50)
    minus_di = _safe_float(data['minus_di'].iloc[-1], 50)

    # Calculate ADX slope (rate of change)
    adx_slope_val, adx_slope = _calculate_adx_slope(adx_series, slope_period)

    # Detect compression breakout pattern
    is_compression, bars_since_low, recent_low = _detect_adx_compression(
        adx_series, lookback, config
    )

    # Determine ADX regime
    regime = _classify_adx_regime(current_adx, adx_slope, config)

    # Generate trading style recommendations
    trading_style, style_details = _generate_trading_style(regime, adx_slope, current_adx)

    # Predict S/R behavior based on ADX
    sr_behavior, sr_reason = _predict_sr_behavior(current_adx, adx_slope, config)

    return ADXAnalysis(
        adx=current_adx,
        plus_di=plus_di,
        minus_di=minus_di,
        slope=adx_slope,
        slope_value=adx_slope_val,
        regime=regime,
        bars_since_low=bars_since_low,
        recent_low=recent_low,
        is_compression_breakout=is_compression,
        trading_style=trading_style,
        style_details=style_details,
        sr_behavior=sr_behavior,
        sr_behavior_reason=sr_reason,
    )


def _default_adx_analysis() -> ADXAnalysis:
    """Return default ADX analysis when data is insufficient."""
    return ADXAnalysis(
        adx=20,
        plus_di=50,
        minus_di=50,
        slope=ADXSlope.FLAT,
        slope_value=0.0,
        regime=ADXRegime.RANGE_MEAN_REVERSION,
        trading_style="Insufficient data",
        style_details=["Need more price history for ADX analysis"],
        sr_behavior="wall",
        sr_behavior_reason="Default to defensive stance with limited data",
    )


def _calculate_adx_slope(
    adx_series: pd.Series,
    period: int = 5
) -> Tuple[float, ADXSlope]:
    """
    Calculate ADX slope (rate of change).

    From professional trading: "ADX slope > ADX level"
    Rising/falling tells you more about "now" than absolute level.

    Args:
        adx_series: ADX values
        period: Bars to calculate slope

    Returns:
        (slope_value, ADXSlope enum)
    """
    if len(adx_series) < period + 1:
        return 0.0, ADXSlope.FLAT

    recent = adx_series.tail(period + 1)
    slope_val = (recent.iloc[-1] - recent.iloc[0]) / period

    # Thresholds for slope classification
    # ADX typically moves 0.5-2 points per bar in active conditions
    if slope_val > 0.5:
        return slope_val, ADXSlope.RISING
    elif slope_val < -0.5:
        return slope_val, ADXSlope.FALLING
    else:
        return slope_val, ADXSlope.FLAT


def _detect_adx_compression(
    adx_series: pd.Series,
    lookback: int = 14,
    config: dict = None
) -> Tuple[bool, int, float]:
    """
    Detect ADX compression breakout pattern.

    Powerful pattern from professional trading:
    - ADX stays low for a while (compression)
    - Breakout happens
    - ADX starts rising (trend regime "turns on")

    Args:
        adx_series: ADX values
        lookback: Bars to analyze
        config: Config with thresholds

    Returns:
        (is_compression_breakout, bars_since_low, recent_low_value)
    """
    if config is None:
        config = REGIME_CONFIG

    if len(adx_series) < lookback:
        return False, 0, 0.0

    recent_adx = adx_series.tail(lookback)
    current_adx = recent_adx.iloc[-1]
    low_threshold = config.get('adx_low_threshold', 20)

    # Find the minimum ADX in lookback period
    min_idx = recent_adx.idxmin()
    min_val = recent_adx.min()
    bars_since_low = len(recent_adx) - list(recent_adx.index).index(min_idx) - 1

    # Compression breakout: ADX was below threshold, now rising significantly above
    was_compressed = min_val < low_threshold
    now_rising = current_adx > min_val + 5  # At least 5 points above recent low
    low_was_recent = bars_since_low < lookback // 2  # Low was in first half of lookback

    is_compression = was_compressed and now_rising and low_was_recent

    return is_compression, bars_since_low, min_val


def _classify_adx_regime(
    adx: float,
    slope: ADXSlope,
    config: dict = None
) -> ADXRegime:
    """
    Classify the ADX regime for trading style selection.

    Three regimes from professional FX trading:
    1. Range/Mean-reversion: ADX low and flat/falling - fade extremes
    2. Transition/Breakout: ADX low but turning up - breakout risk
    3. Trend/Continuation: ADX rising and elevated - continuation plays

    Args:
        adx: Current ADX value
        slope: ADX slope direction
        config: Config with thresholds

    Returns:
        ADXRegime enum
    """
    if config is None:
        config = REGIME_CONFIG

    low_threshold = config.get('adx_low_threshold', 20)
    trending_threshold = config.get('adx_trending_threshold', 25)
    strong_threshold = config.get('adx_strong_threshold', 30)

    # Regime A: Range / Mean-reversion
    # ADX low and flat OR falling
    if adx < low_threshold and slope in [ADXSlope.FLAT, ADXSlope.FALLING]:
        return ADXRegime.RANGE_MEAN_REVERSION

    # Also range if ADX is moderate but falling (trend degrading)
    if adx < trending_threshold and slope == ADXSlope.FALLING:
        return ADXRegime.RANGE_MEAN_REVERSION

    # Regime B: Transition / Breakout risk
    # ADX low but turning up
    if adx < trending_threshold and slope == ADXSlope.RISING:
        return ADXRegime.TRANSITION_BREAKOUT

    # Also transition if ADX just crossed above low threshold
    if low_threshold <= adx < trending_threshold and slope in [ADXSlope.RISING, ADXSlope.FLAT]:
        return ADXRegime.TRANSITION_BREAKOUT

    # Regime C: Trend / Continuation
    # ADX rising and elevated
    if adx >= trending_threshold and slope == ADXSlope.RISING:
        return ADXRegime.TREND_CONTINUATION

    # Also trend if ADX is strong even if flat (established trend)
    if adx >= strong_threshold:
        return ADXRegime.TREND_CONTINUATION

    # Default to transition for ambiguous cases
    return ADXRegime.TRANSITION_BREAKOUT


def _generate_trading_style(
    regime: ADXRegime,
    slope: ADXSlope,
    adx: float
) -> Tuple[str, List[str]]:
    """
    Generate trading style recommendations based on ADX regime.

    From professional FX trading: ADX is a "style selector"
    - Low ADX -> fade / mean revert
    - Rising ADX -> trend continuation or breakout-retest

    Args:
        regime: ADX regime classification
        slope: ADX slope
        adx: Current ADX value

    Returns:
        (style_name, list of style details)
    """
    if regime == ADXRegime.RANGE_MEAN_REVERSION:
        return "Mean Reversion", [
            "Fade range edges and stop-runs",
            "S/R levels likely to act as walls (hold)",
            "Tighter take-profits recommended",
            "Don't chase breakouts - they often fail",
            "RSI overbought/oversold extremes more reliable",
        ]

    elif regime == ADXRegime.TRANSITION_BREAKOUT:
        return "Breakout Watch", [
            "Smaller position size - higher uncertainty",
            "Wait for acceptance (close + hold beyond level)",
            "False breakouts common until ADX confirms",
            "Watch for ADX to continue rising to confirm trend",
            f"ADX at {adx:.1f} - not yet confirming strong trend",
        ]

    else:  # TREND_CONTINUATION
        style_details = [
            "Pullbacks into structure are higher probability",
            "Breakout-retest setups work well",
            "Use runners with trailing stops",
            "S/R levels are liquidity targets (breaks more likely)",
            "RSI overbought is NOT a sell signal in uptrends",
        ]

        if slope == ADXSlope.RISING:
            style_details.append("Don't step in front of this trend")
        elif slope == ADXSlope.FLAT:
            style_details.append("Trend established but watch for ADX rollover")
        else:  # FALLING from high
            style_details.append("WARNING: ADX falling - trend may be exhausting")

        return "Trend Continuation", style_details


def _predict_sr_behavior(
    adx: float,
    slope: ADXSlope,
    config: dict = None
) -> Tuple[str, str]:
    """
    Predict how S/R levels will behave based on ADX.

    Key insight from professional trading:
    - Low ADX: S/R tends to behave like a wall (fades/bounces reliable)
    - Rising ADX: S/R behaves more like a liquidity target (breaks more likely)

    This answers: "Is this level likely to reject... or get eaten through?"

    Args:
        adx: Current ADX value
        slope: ADX slope
        config: Config with thresholds

    Returns:
        (behavior, reason)
    """
    if config is None:
        config = REGIME_CONFIG

    low_threshold = config.get('adx_low_threshold', 20)
    trending_threshold = config.get('adx_trending_threshold', 25)

    if adx < low_threshold:
        return "wall", f"ADX low ({adx:.1f}) - levels likely to hold, fades reliable"

    if adx < trending_threshold:
        if slope == ADXSlope.RISING:
            return "transitioning", f"ADX rising from {adx:.1f} - watch for breakout acceptance"
        else:
            return "wall", f"ADX moderate ({adx:.1f}) and not rising - levels should hold"

    # ADX >= trending_threshold
    if slope == ADXSlope.RISING:
        return "liquidity_target", f"ADX high ({adx:.1f}) and rising - levels are break targets"
    elif slope == ADXSlope.FLAT:
        return "liquidity_target", f"ADX elevated ({adx:.1f}) - levels are liquidity pools"
    else:  # FALLING
        return "transitioning", f"ADX falling from {adx:.1f} - breaks less reliable, watch for reversal"


def get_rsi_interpretation(
    rsi: float,
    adx_analysis: ADXAnalysis
) -> Tuple[str, str]:
    """
    Get regime-dependent RSI interpretation.

    Most people use RSI as "overbought/oversold" but better use is regime-dependent:
    - High/rising ADX (trend): RSI overbought is NOT a sell signal
      Look for RSI holding above 40-50 on pullbacks in uptrends
    - Low ADX (range): RSI extremes work better (fade 70/30 behavior)

    Args:
        rsi: Current RSI value
        adx_analysis: ADX analysis result

    Returns:
        (interpretation, detail)
    """
    regime = adx_analysis.regime
    direction = adx_analysis.di_direction

    if regime == ADXRegime.RANGE_MEAN_REVERSION:
        # Range regime - traditional RSI interpretation works
        if rsi >= 70:
            return "bearish", "RSI overbought in range - fade potential"
        elif rsi <= 30:
            return "bullish", "RSI oversold in range - fade potential"
        else:
            return "neutral", "RSI neutral in range regime"

    elif regime == ADXRegime.TREND_CONTINUATION:
        # Trend regime - RSI interpretation changes
        if direction == "bullish":
            if rsi >= 70:
                return "neutral", "RSI overbought but trend is UP - not a sell signal"
            elif rsi >= 50:
                return "bullish", "RSI holding above 50 in uptrend - healthy"
            elif rsi >= 40:
                return "bullish", "RSI pullback in uptrend - potential entry zone"
            else:
                return "warning", "RSI below 40 in uptrend - watch for trend break"
        elif direction == "bearish":
            if rsi <= 30:
                return "neutral", "RSI oversold but trend is DOWN - not a buy signal"
            elif rsi <= 50:
                return "bearish", "RSI holding below 50 in downtrend - healthy"
            elif rsi <= 60:
                return "bearish", "RSI pullback in downtrend - potential entry zone"
            else:
                return "warning", "RSI above 60 in downtrend - watch for trend break"
        else:
            return "neutral", "Direction unclear despite trending ADX"

    else:  # TRANSITION_BREAKOUT
        if rsi >= 70:
            return "caution", "RSI overbought during transition - wait for confirmation"
        elif rsi <= 30:
            return "caution", "RSI oversold during transition - wait for confirmation"
        else:
            return "neutral", "RSI neutral during transition period"


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
