"""Volatility analysis for options traders.

Provides realized volatility calculations and vol regime forecasting
based on technical analysis signals.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum


class VolRegime(Enum):
    """Volatility regime classification."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    SQUEEZE = "squeeze"  # Bollinger squeeze - expansion imminent


class VolForecast(Enum):
    """Volatility forecast direction."""
    EXPANDING = "expanding"  # Vol likely to increase
    COMPRESSING = "compressing"  # Vol likely to decrease
    NEUTRAL = "neutral"


@dataclass
class VolatilityMetrics:
    """Container for volatility metrics."""
    # Realized volatility (annualized)
    rv_1m: float  # 1-month (21 trading days)
    rv_3m: float  # 3-month (63 trading days)
    rv_1y: float  # 1-year (252 trading days)

    # Current ATR-based vol
    atr_vol: float  # ATR as % of price (annualized)

    # Percentile rankings (0-100)
    rv_percentile: float  # Where current 1M vol sits historically

    # Regime classification
    regime: VolRegime
    regime_description: str

    # Bollinger squeeze detection
    bb_width: float
    bb_width_percentile: float
    is_squeeze: bool

    # Vol trend
    vol_trend: str  # "rising", "falling", "flat"
    vol_change_5d: float  # 5-day vol change in percentage points


@dataclass
class VolForecastSignal:
    """Individual volatility forecast signal."""
    source: str  # "ADX", "BB_Squeeze", "RSI", etc.
    direction: VolForecast
    confidence: str  # "high", "medium", "low"
    reasoning: str


@dataclass
class VolatilityAnalysis:
    """Complete volatility analysis for options trading."""
    metrics: VolatilityMetrics
    forecast: VolForecast
    forecast_confidence: float  # 0-1
    signals: List[VolForecastSignal]
    strategy_suggestion: str

    def get_vol_bias_emoji(self) -> str:
        """Get emoji for vol forecast."""
        if self.forecast == VolForecast.EXPANDING:
            return "📈"
        elif self.forecast == VolForecast.COMPRESSING:
            return "📉"
        return "➡️"


def calculate_realized_volatility(
    returns: pd.Series,
    window: int,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Calculate realized volatility (annualized standard deviation of returns).

    Args:
        returns: Series of log returns
        window: Rolling window size
        annualization_factor: Trading days per year

    Returns:
        Series of annualized volatility
    """
    return returns.rolling(window=window).std() * np.sqrt(annualization_factor) * 100


def calculate_volatility_metrics(
    df: pd.DataFrame,
    lookback_percentile: int = 252
) -> VolatilityMetrics:
    """
    Calculate comprehensive volatility metrics from price data.

    Args:
        df: DataFrame with OHLC data and indicators
        lookback_percentile: Bars to use for percentile calculation

    Returns:
        VolatilityMetrics object
    """
    # Calculate log returns
    close = df['Close'].squeeze()
    returns = np.log(close / close.shift(1))

    # Realized volatility for different windows
    rv_1m = calculate_realized_volatility(returns, 21)
    rv_3m = calculate_realized_volatility(returns, 63)
    rv_1y = calculate_realized_volatility(returns, 252)

    # Get current values
    current_rv_1m = _safe_float(rv_1m.iloc[-1])
    current_rv_3m = _safe_float(rv_3m.iloc[-1])
    current_rv_1y = _safe_float(rv_1y.iloc[-1])

    # ATR-based volatility (annualized)
    if 'atr' in df.columns:
        atr = df['atr'].iloc[-1]
        current_price = close.iloc[-1]
        atr_vol = (atr / current_price) * np.sqrt(252) * 100
    else:
        atr_vol = current_rv_1m

    # Percentile ranking
    rv_history = rv_1m.tail(lookback_percentile).dropna()
    if len(rv_history) > 0:
        rv_percentile = (rv_history < current_rv_1m).sum() / len(rv_history) * 100
    else:
        rv_percentile = 50.0

    # Bollinger Band width analysis
    if 'bb_width' in df.columns:
        bb_width = _safe_float(df['bb_width'].iloc[-1])
        bb_history = df['bb_width'].tail(lookback_percentile).dropna()
        if len(bb_history) > 0:
            bb_width_percentile = (bb_history < bb_width).sum() / len(bb_history) * 100
        else:
            bb_width_percentile = 50.0
    else:
        bb_width = 0.0
        bb_width_percentile = 50.0

    # Squeeze detection (BB width in bottom 20th percentile)
    is_squeeze = bb_width_percentile < 20

    # Vol trend (compare current 1M vol to 5 days ago)
    if len(rv_1m) > 5:
        vol_5d_ago = _safe_float(rv_1m.iloc[-6])
        vol_change_5d = current_rv_1m - vol_5d_ago

        if vol_change_5d > 0.5:  # More than 0.5% increase
            vol_trend = "rising"
        elif vol_change_5d < -0.5:  # More than 0.5% decrease
            vol_trend = "falling"
        else:
            vol_trend = "flat"
    else:
        vol_change_5d = 0.0
        vol_trend = "flat"

    # Regime classification
    if is_squeeze:
        regime = VolRegime.SQUEEZE
        regime_description = "Squeeze - vol expansion likely"
    elif rv_percentile >= 75:
        regime = VolRegime.HIGH
        regime_description = "High vol environment"
    elif rv_percentile <= 25:
        regime = VolRegime.LOW
        regime_description = "Low vol environment"
    else:
        regime = VolRegime.NORMAL
        regime_description = "Normal vol environment"

    return VolatilityMetrics(
        rv_1m=current_rv_1m,
        rv_3m=current_rv_3m,
        rv_1y=current_rv_1y,
        atr_vol=atr_vol,
        rv_percentile=rv_percentile,
        regime=regime,
        regime_description=regime_description,
        bb_width=bb_width,
        bb_width_percentile=bb_width_percentile,
        is_squeeze=is_squeeze,
        vol_trend=vol_trend,
        vol_change_5d=vol_change_5d
    )


def generate_vol_forecast_signals(
    df: pd.DataFrame,
    vol_metrics: VolatilityMetrics,
    indicators: dict
) -> List[VolForecastSignal]:
    """
    Generate volatility forecast signals from TA indicators.

    Options trader logic:
    - BB Squeeze → Vol expansion imminent (BUY VOL)
    - High ADX + Strong trend → Vol compression (SELL VOL)
    - Low ADX + Ranging → Breakout risk = vol spike potential
    - RSI extremes → Reversal = vol expansion
    - Regime change → Vol expansion during transition

    Args:
        df: DataFrame with indicators
        vol_metrics: Current volatility metrics
        indicators: Dict of indicator values

    Returns:
        List of VolForecastSignal objects
    """
    signals = []

    # 1. Bollinger Band Squeeze Signal (highest priority)
    if vol_metrics.is_squeeze:
        signals.append(VolForecastSignal(
            source="BB Squeeze",
            direction=VolForecast.EXPANDING,
            confidence="high",
            reasoning=f"BB width at {vol_metrics.bb_width_percentile:.0f}th percentile - squeeze breakout imminent"
        ))
    elif vol_metrics.bb_width_percentile > 80:
        signals.append(VolForecastSignal(
            source="BB Width",
            direction=VolForecast.COMPRESSING,
            confidence="medium",
            reasoning=f"BB width at {vol_metrics.bb_width_percentile:.0f}th percentile - extended, likely to contract"
        ))

    # 2. ADX Trend Strength Signal
    adx = indicators.get('adx', 25)
    plus_di = indicators.get('plus_di', 50)
    minus_di = indicators.get('minus_di', 50)

    if adx > 30:
        # Strong trend = vol typically compresses
        signals.append(VolForecastSignal(
            source="ADX Trend",
            direction=VolForecast.COMPRESSING,
            confidence="medium",
            reasoning=f"Strong trend (ADX={adx:.1f}) - vol tends to compress in trending markets"
        ))
    elif adx < 20:
        # Weak trend / ranging = breakout risk
        signals.append(VolForecastSignal(
            source="ADX Range",
            direction=VolForecast.EXPANDING,
            confidence="medium",
            reasoning=f"Ranging market (ADX={adx:.1f}) - breakout could spike vol"
        ))

    # 3. RSI Extreme Signal
    rsi = indicators.get('rsi', 50)
    if rsi <= 25 or rsi >= 75:
        signals.append(VolForecastSignal(
            source="RSI Extreme",
            direction=VolForecast.EXPANDING,
            confidence="medium",
            reasoning=f"RSI at extreme ({rsi:.1f}) - potential reversal could increase vol"
        ))

    # 4. Vol Trend Signal
    if vol_metrics.vol_trend == "rising" and vol_metrics.rv_percentile < 50:
        signals.append(VolForecastSignal(
            source="Vol Trend",
            direction=VolForecast.EXPANDING,
            confidence="medium",
            reasoning=f"Vol rising from low base ({vol_metrics.vol_change_5d:+.1f}% 5d change)"
        ))
    elif vol_metrics.vol_trend == "falling" and vol_metrics.rv_percentile > 50:
        signals.append(VolForecastSignal(
            source="Vol Trend",
            direction=VolForecast.COMPRESSING,
            confidence="medium",
            reasoning=f"Vol falling from high base ({vol_metrics.vol_change_5d:+.1f}% 5d change)"
        ))

    # 5. Vol Percentile Mean Reversion
    if vol_metrics.rv_percentile >= 90:
        signals.append(VolForecastSignal(
            source="Vol Percentile",
            direction=VolForecast.COMPRESSING,
            confidence="high",
            reasoning=f"Vol at {vol_metrics.rv_percentile:.0f}th percentile - mean reversion likely"
        ))
    elif vol_metrics.rv_percentile <= 10:
        signals.append(VolForecastSignal(
            source="Vol Percentile",
            direction=VolForecast.EXPANDING,
            confidence="high",
            reasoning=f"Vol at {vol_metrics.rv_percentile:.0f}th percentile - expansion likely"
        ))

    # 6. MACD Histogram momentum shift
    macd_hist = indicators.get('macd_histogram', 0)
    if 'macd_histogram' in df.columns and len(df) > 5:
        hist_5d_ago = _safe_float(df['macd_histogram'].iloc[-6])
        hist_change = abs(macd_hist) - abs(hist_5d_ago)

        if hist_change > 0 and abs(macd_hist) > abs(hist_5d_ago) * 1.5:
            signals.append(VolForecastSignal(
                source="MACD Momentum",
                direction=VolForecast.EXPANDING,
                confidence="low",
                reasoning="MACD momentum accelerating - increased price activity"
            ))

    return signals


def calculate_vol_forecast(
    signals: List[VolForecastSignal]
) -> Tuple[VolForecast, float]:
    """
    Aggregate signals into overall forecast with confidence.

    Args:
        signals: List of forecast signals

    Returns:
        Tuple of (forecast direction, confidence 0-1)
    """
    if not signals:
        return VolForecast.NEUTRAL, 0.5

    # Weight by confidence
    confidence_weights = {"high": 3, "medium": 2, "low": 1}

    expanding_score = 0
    compressing_score = 0
    total_weight = 0

    for signal in signals:
        weight = confidence_weights.get(signal.confidence, 1)
        total_weight += weight

        if signal.direction == VolForecast.EXPANDING:
            expanding_score += weight
        elif signal.direction == VolForecast.COMPRESSING:
            compressing_score += weight

    # Determine direction
    if expanding_score > compressing_score:
        forecast = VolForecast.EXPANDING
        confidence = expanding_score / total_weight if total_weight > 0 else 0.5
    elif compressing_score > expanding_score:
        forecast = VolForecast.COMPRESSING
        confidence = compressing_score / total_weight if total_weight > 0 else 0.5
    else:
        forecast = VolForecast.NEUTRAL
        confidence = 0.5

    return forecast, min(confidence, 1.0)


def get_strategy_suggestion(
    forecast: VolForecast,
    confidence: float,
    vol_metrics: VolatilityMetrics
) -> str:
    """
    Generate options strategy suggestion based on vol forecast.

    Args:
        forecast: Vol forecast direction
        confidence: Forecast confidence
        vol_metrics: Current vol metrics

    Returns:
        Strategy suggestion string
    """
    if confidence < 0.4:
        return "Mixed signals - consider neutral strategies (iron condors, butterflies)"

    if forecast == VolForecast.EXPANDING:
        if vol_metrics.rv_percentile < 30:
            return "BUY VOL: Consider long straddles/strangles - vol cheap and likely to expand"
        elif vol_metrics.is_squeeze:
            return "BUY VOL: BB squeeze detected - consider long gamma positions for breakout"
        else:
            return "LONG VOL BIAS: Consider owning optionality (calls/puts, ratio backspreads)"

    elif forecast == VolForecast.COMPRESSING:
        if vol_metrics.rv_percentile > 70:
            return "SELL VOL: Consider short straddles/strangles - vol elevated and likely to compress"
        else:
            return "SHORT VOL BIAS: Consider premium selling (covered calls, cash-secured puts)"

    return "NEUTRAL: Consider delta-neutral strategies with limited vol exposure"


def analyze_volatility(
    df: pd.DataFrame,
    indicators: dict
) -> VolatilityAnalysis:
    """
    Complete volatility analysis for options trading.

    Args:
        df: DataFrame with OHLC and indicators
        indicators: Dict of current indicator values

    Returns:
        VolatilityAnalysis object
    """
    # Calculate metrics
    metrics = calculate_volatility_metrics(df)

    # Generate forecast signals
    signals = generate_vol_forecast_signals(df, metrics, indicators)

    # Calculate aggregate forecast
    forecast, confidence = calculate_vol_forecast(signals)

    # Get strategy suggestion
    strategy = get_strategy_suggestion(forecast, confidence, metrics)

    return VolatilityAnalysis(
        metrics=metrics,
        forecast=forecast,
        forecast_confidence=confidence,
        signals=signals,
        strategy_suggestion=strategy
    )


def calculate_vol_time_series(
    df: pd.DataFrame,
    window: int = 21
) -> pd.DataFrame:
    """
    Calculate volatility time series for charting.

    Args:
        df: DataFrame with OHLC data
        window: Rolling window for vol calculation

    Returns:
        DataFrame with vol columns added
    """
    result = df.copy()
    close = df['Close'].squeeze()
    returns = np.log(close / close.shift(1))

    # Realized vol (annualized)
    result['rv_1m'] = calculate_realized_volatility(returns, 21)
    result['rv_3m'] = calculate_realized_volatility(returns, 63)

    # Vol of vol (volatility clustering indicator)
    result['vol_of_vol'] = result['rv_1m'].rolling(window=21).std()

    # Vol percentile (rolling)
    def rolling_percentile(series, lookback=252):
        return series.rolling(window=lookback).apply(
            lambda x: (x[:-1] < x.iloc[-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else 50,
            raw=False
        )

    result['rv_percentile'] = rolling_percentile(result['rv_1m'])

    return result


def _safe_float(value, default: float = 0.0) -> float:
    """Convert value to float, handling NaN and None."""
    if value is None or pd.isna(value):
        return default
    return float(value)
