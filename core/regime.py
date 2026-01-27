"""Market regime detection for FX Trading Dashboard."""

import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import List

from config import REGIME_CONFIG


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
