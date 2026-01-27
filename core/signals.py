"""Signal generation and confluence scoring for FX Trading Dashboard."""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

from config import CONFLUENCE_WEIGHTS


class SignalStrength(Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEUTRAL = "neutral"


@dataclass
class Signal:
    """Individual trading signal."""
    name: str
    direction: str  # "bullish", "bearish", "neutral"
    strength: SignalStrength
    value: float
    description: str


def calculate_confluence_score(
    signals: List[Signal],
    mtf_aligned: bool = False,
    regime_favorable: bool = False,
    near_sr: bool = False
) -> float:
    """
    Calculate overall confluence score from multiple signals.

    Args:
        signals: List of Signal objects
        mtf_aligned: Whether multi-timeframe is aligned
        regime_favorable: Whether regime supports the direction
        near_sr: Whether price is near significant S/R

    Returns:
        Confluence score (0-10 scale)
    """
    weights = CONFLUENCE_WEIGHTS
    score = 0.0

    # Count bullish vs bearish signals
    bullish = sum(1 for s in signals if s.direction == "bullish")
    bearish = sum(1 for s in signals if s.direction == "bearish")

    # Base score from signal agreement
    total_signals = len(signals)
    if total_signals > 0:
        agreement = max(bullish, bearish) / total_signals
        score += agreement * 3.0  # Up to 3 points for signal agreement

    # MTF alignment bonus
    if mtf_aligned:
        score += weights.get("mtf_alignment", 2.0)

    # Regime bonus
    if regime_favorable:
        score += weights.get("regime_favorable", 1.5)

    # S/R proximity bonus
    if near_sr:
        score += weights.get("swing_level", 2.0)

    # Cap at 10
    return min(score, 10.0)


def generate_indicator_signals(
    close: float,
    indicators: dict
) -> List[Signal]:
    """
    Generate signals from technical indicators.

    Args:
        close: Current close price
        indicators: Dict with indicator values

    Returns:
        List of Signal objects
    """
    signals = []

    # RSI Signal
    rsi = indicators.get('rsi', 50)
    if rsi <= 30:
        signals.append(Signal(
            name="RSI",
            direction="bullish",
            strength=SignalStrength.STRONG,
            value=rsi,
            description="Oversold - potential bounce"
        ))
    elif rsi >= 70:
        signals.append(Signal(
            name="RSI",
            direction="bearish",
            strength=SignalStrength.STRONG,
            value=rsi,
            description="Overbought - potential pullback"
        ))
    else:
        signals.append(Signal(
            name="RSI",
            direction="neutral",
            strength=SignalStrength.NEUTRAL,
            value=rsi,
            description="Neutral zone"
        ))

    # MACD Signal
    macd = indicators.get('macd', 0)
    macd_signal = indicators.get('macd_signal', 0)
    if macd > macd_signal and macd > 0:
        signals.append(Signal(
            name="MACD",
            direction="bullish",
            strength=SignalStrength.STRONG,
            value=macd,
            description="Bullish momentum above zero"
        ))
    elif macd > macd_signal:
        signals.append(Signal(
            name="MACD",
            direction="bullish",
            strength=SignalStrength.MODERATE,
            value=macd,
            description="Bullish crossover"
        ))
    elif macd < macd_signal and macd < 0:
        signals.append(Signal(
            name="MACD",
            direction="bearish",
            strength=SignalStrength.STRONG,
            value=macd,
            description="Bearish momentum below zero"
        ))
    elif macd < macd_signal:
        signals.append(Signal(
            name="MACD",
            direction="bearish",
            strength=SignalStrength.MODERATE,
            value=macd,
            description="Bearish crossover"
        ))
    else:
        signals.append(Signal(
            name="MACD",
            direction="neutral",
            strength=SignalStrength.NEUTRAL,
            value=macd,
            description="No clear signal"
        ))

    # SMA Signal
    sma_20 = indicators.get('sma_20', close)
    sma_50 = indicators.get('sma_50', close)
    if close > sma_20 > sma_50:
        signals.append(Signal(
            name="SMA",
            direction="bullish",
            strength=SignalStrength.STRONG,
            value=sma_20,
            description="Price above aligned SMAs"
        ))
    elif close > sma_20:
        signals.append(Signal(
            name="SMA",
            direction="bullish",
            strength=SignalStrength.MODERATE,
            value=sma_20,
            description="Price above SMA 20"
        ))
    elif close < sma_20 < sma_50:
        signals.append(Signal(
            name="SMA",
            direction="bearish",
            strength=SignalStrength.STRONG,
            value=sma_20,
            description="Price below aligned SMAs"
        ))
    elif close < sma_20:
        signals.append(Signal(
            name="SMA",
            direction="bearish",
            strength=SignalStrength.MODERATE,
            value=sma_20,
            description="Price below SMA 20"
        ))
    else:
        signals.append(Signal(
            name="SMA",
            direction="neutral",
            strength=SignalStrength.NEUTRAL,
            value=sma_20,
            description="Price at SMA"
        ))

    # Stochastic Signal
    stoch_k = indicators.get('stoch_k', 50)
    stoch_d = indicators.get('stoch_d', 50)
    if stoch_k <= 20 and stoch_k > stoch_d:
        signals.append(Signal(
            name="Stochastic",
            direction="bullish",
            strength=SignalStrength.STRONG,
            value=stoch_k,
            description="Oversold with bullish crossover"
        ))
    elif stoch_k >= 80 and stoch_k < stoch_d:
        signals.append(Signal(
            name="Stochastic",
            direction="bearish",
            strength=SignalStrength.STRONG,
            value=stoch_k,
            description="Overbought with bearish crossover"
        ))
    else:
        signals.append(Signal(
            name="Stochastic",
            direction="neutral",
            strength=SignalStrength.NEUTRAL,
            value=stoch_k,
            description="No extreme reading"
        ))

    return signals


def summarize_signals(signals: List[Signal]) -> Dict[str, int]:
    """
    Summarize signal directions.

    Args:
        signals: List of Signal objects

    Returns:
        Dict with counts of bullish, bearish, neutral
    """
    return {
        "bullish": sum(1 for s in signals if s.direction == "bullish"),
        "bearish": sum(1 for s in signals if s.direction == "bearish"),
        "neutral": sum(1 for s in signals if s.direction == "neutral"),
    }
