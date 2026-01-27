"""Technical indicator calculations for FX Trading Dashboard."""

import pandas as pd
import ta
from dataclasses import dataclass
from typing import Tuple


@dataclass
class IndicatorValues:
    """Container for latest indicator values."""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    sma_20: float
    sma_50: float
    sma_200: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    stoch_k: float
    stoch_d: float
    atr: float
    adx: float
    plus_di: float
    minus_di: float


def calculate_all_indicators(data: pd.DataFrame) -> Tuple[pd.DataFrame, IndicatorValues]:
    """
    Calculate all technical indicators.

    Args:
        data: OHLCV DataFrame

    Returns:
        Tuple of (enriched DataFrame, IndicatorValues with latest values)
    """
    df = data.copy()

    # Ensure we have 1D series (handle yfinance MultiIndex)
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()

    # Trend Indicators - Moving Averages
    df['sma_20'] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    df['sma_50'] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df['sma_200'] = ta.trend.SMAIndicator(close, window=200).sma_indicator()

    # MACD
    macd_indicator = ta.trend.MACD(close)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_histogram'] = macd_indicator.macd_diff()

    # ADX for trend strength
    adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
    df['adx'] = adx_indicator.adx()
    df['plus_di'] = adx_indicator.adx_pos()
    df['minus_di'] = adx_indicator.adx_neg()

    # Momentum Indicators
    df['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi()

    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # Volatility Indicators
    bollinger = ta.volatility.BollingerBands(close, window=20)
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_middle'] = bollinger.bollinger_mavg()
    df['bb_lower'] = bollinger.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    df['atr'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # Get latest values
    latest = df.iloc[-1]
    indicator_values = IndicatorValues(
        rsi=_safe_float(latest.get('rsi')),
        macd=_safe_float(latest.get('macd')),
        macd_signal=_safe_float(latest.get('macd_signal')),
        macd_histogram=_safe_float(latest.get('macd_histogram')),
        sma_20=_safe_float(latest.get('sma_20')),
        sma_50=_safe_float(latest.get('sma_50')),
        sma_200=_safe_float(latest.get('sma_200')),
        bb_upper=_safe_float(latest.get('bb_upper')),
        bb_middle=_safe_float(latest.get('bb_middle')),
        bb_lower=_safe_float(latest.get('bb_lower')),
        bb_width=_safe_float(latest.get('bb_width')),
        stoch_k=_safe_float(latest.get('stoch_k')),
        stoch_d=_safe_float(latest.get('stoch_d')),
        atr=_safe_float(latest.get('atr')),
        adx=_safe_float(latest.get('adx')),
        plus_di=_safe_float(latest.get('plus_di')),
        minus_di=_safe_float(latest.get('minus_di')),
    )

    return df, indicator_values


def _safe_float(value) -> float:
    """Convert value to float, handling NaN and None."""
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def get_indicator_summary(indicators: IndicatorValues, close: float) -> dict:
    """
    Generate a summary of indicator signals.

    Args:
        indicators: IndicatorValues object
        close: Current close price

    Returns:
        Dict with indicator summaries
    """
    return {
        'trend': {
            'sma_20_position': 'above' if close > indicators.sma_20 else 'below',
            'sma_50_position': 'above' if close > indicators.sma_50 else 'below',
            'macd_signal': 'bullish' if indicators.macd > indicators.macd_signal else 'bearish',
            'macd_zero': 'bullish' if indicators.macd > 0 else 'bearish',
        },
        'momentum': {
            'rsi_value': indicators.rsi,
            'rsi_zone': _get_rsi_zone(indicators.rsi),
            'stoch_zone': _get_stoch_zone(indicators.stoch_k),
        },
        'volatility': {
            'bb_position': _get_bb_position(close, indicators),
            'atr': indicators.atr,
        },
        'trend_strength': {
            'adx': indicators.adx,
            'trending': indicators.adx > 25,
            'direction': 'up' if indicators.plus_di > indicators.minus_di else 'down',
        }
    }


def _get_rsi_zone(rsi: float) -> str:
    """Classify RSI zone."""
    if rsi <= 30:
        return 'oversold'
    elif rsi >= 70:
        return 'overbought'
    else:
        return 'neutral'


def _get_stoch_zone(stoch_k: float) -> str:
    """Classify Stochastic zone."""
    if stoch_k <= 20:
        return 'oversold'
    elif stoch_k >= 80:
        return 'overbought'
    else:
        return 'neutral'


def _get_bb_position(close: float, indicators: IndicatorValues) -> str:
    """Determine position relative to Bollinger Bands."""
    if close >= indicators.bb_upper:
        return 'above_upper'
    elif close <= indicators.bb_lower:
        return 'below_lower'
    elif close > indicators.bb_middle:
        return 'upper_half'
    else:
        return 'lower_half'
