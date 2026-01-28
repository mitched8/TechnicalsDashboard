"""Signal generation and confluence scoring for FX Trading Dashboard."""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import pandas as pd
import numpy as np

from config import CONFLUENCE_WEIGHTS, StrategySettings, EntrySignalSettings


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


# ============================================================================
# NEW SIGNAL METHODS FROM MACD_EMA_TREND STRATEGY
# ============================================================================

def generate_pullback_resumption_signal(
    df: pd.DataFrame,
    lookback: int = 3
) -> Optional[Signal]:
    """
    Generate pullback resumption signal from MACD.

    From MACD_EMA_Trend strategy: A resumption trade occurs when MACD crosses
    its signal line while BOTH lines are on the same side of zero (recovering
    from a pullback).

    Long setup: MACD crosses above signal while both are BELOW zero
    Short setup: MACD crosses below signal while both are ABOVE zero

    Args:
        df: DataFrame with MACD columns
        lookback: Number of bars to check for crossover

    Returns:
        Signal object or None if no signal
    """
    if len(df) < lookback + 1:
        return None

    if 'macd' not in df.columns or 'macd_signal' not in df.columns:
        return None

    # Get recent data
    recent = df.tail(lookback + 1)
    macd = recent['macd'].values
    signal_line = recent['macd_signal'].values

    # Current values
    current_macd = macd[-1]
    current_signal = signal_line[-1]
    prev_macd = macd[-2]
    prev_signal = signal_line[-2]

    # Check for bullish crossover (MACD crosses above signal)
    bullish_cross = (prev_macd <= prev_signal) and (current_macd > current_signal)

    # Check for bearish crossover (MACD crosses below signal)
    bearish_cross = (prev_macd >= prev_signal) and (current_macd < current_signal)

    # PULLBACK RESUMPTION LOGIC:
    # Long: Both MACD and Signal are below zero (pullback) and MACD crosses up
    if bullish_cross and current_macd < 0 and current_signal < 0:
        return Signal(
            name="MACD Pullback",
            direction="bullish",
            strength=SignalStrength.STRONG,
            value=current_macd,
            description="Pullback resumption: MACD cross up while both lines below zero"
        )

    # Short: Both MACD and Signal are above zero (pullback) and MACD crosses down
    if bearish_cross and current_macd > 0 and current_signal > 0:
        return Signal(
            name="MACD Pullback",
            direction="bearish",
            strength=SignalStrength.STRONG,
            value=current_macd,
            description="Pullback resumption: MACD cross down while both lines above zero"
        )

    return None


def generate_histogram_pattern_signal(
    df: pd.DataFrame,
    lookback: int = 7,
    threshold: float = 4e-6
) -> Optional[Signal]:
    """
    Generate signal from MACD histogram patterns.

    From MACD_EMA_Trend strategy: Looks for histogram pullback patterns where
    the histogram shrinks toward zero then reverses.

    Args:
        df: DataFrame with macd_histogram column
        lookback: Number of bars to analyze
        threshold: Minimum histogram change to consider significant

    Returns:
        Signal object or None if no signal
    """
    if len(df) < lookback:
        return None

    if 'macd_histogram' not in df.columns:
        return None

    # Get histogram values
    hist = df['macd_histogram'].tail(lookback).values

    # Remove NaN values
    hist = hist[~np.isnan(hist)]
    if len(hist) < 3:
        return None

    current_hist = hist[-1]
    prev_hist = hist[-2]

    # Check for histogram reversal (pullback completion)

    # Bullish histogram pattern:
    # - Histogram was negative (below zero)
    # - Histogram is now less negative or turning positive (shrinking toward/past zero)
    # - Recent change shows upward momentum
    if prev_hist < -threshold and current_hist > prev_hist:
        # Check if we're seeing a pullback recovery pattern
        # Look for histogram that was declining and is now rising
        min_idx = np.argmin(hist)
        if min_idx > 0 and min_idx < len(hist) - 1:  # Min is not at edges
            if hist[-1] > hist[min_idx]:  # Rising from the minimum
                return Signal(
                    name="Histogram Pattern",
                    direction="bullish",
                    strength=SignalStrength.MODERATE,
                    value=current_hist,
                    description="Histogram pullback recovery - momentum turning bullish"
                )

    # Bearish histogram pattern:
    # - Histogram was positive (above zero)
    # - Histogram is now less positive or turning negative
    # - Recent change shows downward momentum
    if prev_hist > threshold and current_hist < prev_hist:
        # Look for histogram that was rising and is now falling
        max_idx = np.argmax(hist)
        if max_idx > 0 and max_idx < len(hist) - 1:  # Max is not at edges
            if hist[-1] < hist[max_idx]:  # Falling from the maximum
                return Signal(
                    name="Histogram Pattern",
                    direction="bearish",
                    strength=SignalStrength.MODERATE,
                    value=current_hist,
                    description="Histogram pullback recovery - momentum turning bearish"
                )

    return None


def generate_ema_200_signal(
    close: float,
    ema_200: float,
    ema_200_trend: str,
    bars_confirmed: int,
    window_required: int = 6
) -> Optional[Signal]:
    """
    Generate signal based on EMA 200 filter.

    From MACD_EMA_Trend strategy: Price must be consistently above/below EMA 200
    for trend confirmation.

    Args:
        close: Current close price
        ema_200: Current EMA 200 value
        ema_200_trend: Calculated trend direction ("bullish", "bearish", "neutral")
        bars_confirmed: Number of consecutive bars confirming the trend
        window_required: Minimum bars required for confirmation

    Returns:
        Signal object or None
    """
    if ema_200 == 0:
        return None

    if ema_200_trend == "bullish" and bars_confirmed >= window_required:
        return Signal(
            name="EMA 200",
            direction="bullish",
            strength=SignalStrength.STRONG,
            value=ema_200,
            description=f"Price above EMA 200 for {bars_confirmed} bars - strong uptrend"
        )
    elif ema_200_trend == "bearish" and bars_confirmed >= window_required:
        return Signal(
            name="EMA 200",
            direction="bearish",
            strength=SignalStrength.STRONG,
            value=ema_200,
            description=f"Price below EMA 200 for {bars_confirmed} bars - strong downtrend"
        )
    elif close > ema_200:
        return Signal(
            name="EMA 200",
            direction="bullish",
            strength=SignalStrength.WEAK,
            value=ema_200,
            description="Price above EMA 200 but not confirmed"
        )
    elif close < ema_200:
        return Signal(
            name="EMA 200",
            direction="bearish",
            strength=SignalStrength.WEAK,
            value=ema_200,
            description="Price below EMA 200 but not confirmed"
        )

    return None


def generate_signals_with_settings(
    df: pd.DataFrame,
    close: float,
    indicators: dict,
    settings: Optional[EntrySignalSettings] = None
) -> List[Signal]:
    """
    Generate signals based on user-configured settings.

    This is the main entry point for signal generation that respects user settings.

    Args:
        df: Full DataFrame with indicator data
        close: Current close price
        indicators: Dict with indicator values
        settings: Entry signal settings (uses defaults if None)

    Returns:
        List of Signal objects
    """
    if settings is None:
        settings = EntrySignalSettings()

    signals = []

    # Standard MACD crossover (original method)
    if settings.use_standard_macd:
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

    # Pullback resumption (new method from MACD_EMA_Trend)
    if settings.use_pullback_resumption:
        pullback_signal = generate_pullback_resumption_signal(df)
        if pullback_signal:
            signals.append(pullback_signal)

    # Histogram patterns (new method from MACD_EMA_Trend)
    if settings.use_histogram_patterns:
        hist_signal = generate_histogram_pattern_signal(
            df,
            lookback=settings.histogram_lookback,
            threshold=settings.histogram_threshold
        )
        if hist_signal:
            signals.append(hist_signal)

    # RSI signals
    if settings.use_rsi_signals:
        rsi = indicators.get('rsi', 50)
        if rsi <= settings.rsi_oversold:
            signals.append(Signal(
                name="RSI",
                direction="bullish",
                strength=SignalStrength.STRONG,
                value=rsi,
                description=f"Oversold (RSI={rsi:.1f}) - potential bounce"
            ))
        elif rsi >= settings.rsi_overbought:
            signals.append(Signal(
                name="RSI",
                direction="bearish",
                strength=SignalStrength.STRONG,
                value=rsi,
                description=f"Overbought (RSI={rsi:.1f}) - potential pullback"
            ))

    # Stochastic signals
    if settings.use_stochastic_signals:
        stoch_k = indicators.get('stoch_k', 50)
        stoch_d = indicators.get('stoch_d', 50)
        if stoch_k <= settings.stoch_oversold and stoch_k > stoch_d:
            signals.append(Signal(
                name="Stochastic",
                direction="bullish",
                strength=SignalStrength.STRONG,
                value=stoch_k,
                description="Oversold with bullish crossover"
            ))
        elif stoch_k >= settings.stoch_overbought and stoch_k < stoch_d:
            signals.append(Signal(
                name="Stochastic",
                direction="bearish",
                strength=SignalStrength.STRONG,
                value=stoch_k,
                description="Overbought with bearish crossover"
            ))

    return signals
