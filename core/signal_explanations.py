"""Technical Analysis Signal Explanations for FX Trading Dashboard.

Provides educational content on each TA signal: what it is, how it's calculated,
and how professional traders use it.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class SignalExplanation:
    """Explanation of a technical analysis signal."""
    name: str
    category: str  # "momentum", "trend", "volatility", "support_resistance"
    what_it_is: str
    calculation: str
    professional_use: str
    signal_interpretation: str


# Comprehensive explanations for each signal type
SIGNAL_EXPLANATIONS: Dict[str, SignalExplanation] = {
    "RSI": SignalExplanation(
        name="Relative Strength Index (RSI)",
        category="momentum",
        what_it_is=(
            "RSI is a momentum oscillator that measures the speed and magnitude of "
            "price movements on a scale of 0 to 100. Developed by J. Welles Wilder in 1978, "
            "it compares the magnitude of recent gains to recent losses."
        ),
        calculation=(
            "RSI = 100 - (100 / (1 + RS))\n"
            "Where RS = Average Gain over N periods / Average Loss over N periods\n"
            "Standard period is 14. First calculation uses simple averages; "
            "subsequent calculations use smoothed averages (Wilder's smoothing)."
        ),
        professional_use=(
            "Institutional traders use RSI primarily for:\n"
            "- Identifying overbought (>70) and oversold (<30) conditions\n"
            "- Divergence analysis: price makes new high but RSI doesn't (bearish divergence)\n"
            "- Failure swings: RSI breaks its own support/resistance before price\n"
            "- Trend confirmation: RSI staying above 40 in uptrends, below 60 in downtrends"
        ),
        signal_interpretation=(
            "BULLISH: RSI crosses above 30 from oversold territory\n"
            "BEARISH: RSI crosses below 70 from overbought territory\n"
            "NEUTRAL: RSI between 40-60 in ranging markets"
        )
    ),

    "MACD": SignalExplanation(
        name="Moving Average Convergence Divergence (MACD)",
        category="trend",
        what_it_is=(
            "MACD is a trend-following momentum indicator showing the relationship "
            "between two exponential moving averages. Created by Gerald Appel in the 1970s, "
            "it reveals changes in strength, direction, momentum, and duration of a trend."
        ),
        calculation=(
            "MACD Line = 12-period EMA - 26-period EMA\n"
            "Signal Line = 9-period EMA of MACD Line\n"
            "Histogram = MACD Line - Signal Line\n"
            "The histogram visualizes the distance between MACD and signal lines."
        ),
        professional_use=(
            "Professional traders use MACD for:\n"
            "- Signal line crossovers: MACD crossing above signal = bullish\n"
            "- Zero line crossovers: MACD crossing above zero = bullish trend\n"
            "- Divergence: price trend not confirmed by MACD momentum\n"
            "- Histogram patterns: shrinking histogram warns of potential reversal"
        ),
        signal_interpretation=(
            "BULLISH: MACD crosses above signal line, especially below zero line\n"
            "BEARISH: MACD crosses below signal line, especially above zero line\n"
            "STRONG: Both MACD lines on same side of zero as the crossover direction"
        )
    ),

    "MACD_HISTOGRAM": SignalExplanation(
        name="MACD Histogram Patterns",
        category="momentum",
        what_it_is=(
            "The MACD histogram represents the difference between MACD and its signal line. "
            "It provides early warning of potential trend changes by showing momentum shifts "
            "before the actual MACD crossover occurs."
        ),
        calculation=(
            "Histogram = MACD Line - Signal Line\n"
            "Positive histogram: MACD above signal (bullish momentum)\n"
            "Negative histogram: MACD below signal (bearish momentum)\n"
            "Histogram peaks/troughs often precede price reversals."
        ),
        professional_use=(
            "Sophisticated traders watch histogram patterns for:\n"
            "- Peak/trough reversals: histogram turning before price\n"
            "- Divergence: new price highs with lower histogram peaks\n"
            "- Zero crossings: histogram crossing zero confirms MACD crossover\n"
            "- Rate of change: accelerating/decelerating histogram bars"
        ),
        signal_interpretation=(
            "BULLISH: Histogram turns up from negative extreme (pullback resumption)\n"
            "BEARISH: Histogram turns down from positive extreme\n"
            "STRONG: Histogram accelerating in direction of trend"
        )
    ),

    "PULLBACK_RESUMPTION": SignalExplanation(
        name="MACD Pullback Resumption",
        category="trend",
        what_it_is=(
            "A pullback resumption signal occurs when MACD lines cross while both remain "
            "on the same side of zero. This indicates a temporary pullback within an "
            "established trend, rather than a trend reversal."
        ),
        calculation=(
            "Bullish: MACD crosses above signal while both lines are below zero\n"
            "Bearish: MACD crosses below signal while both lines are above zero\n"
            "The key is that the cross happens before reaching the zero line, "
            "suggesting the pullback is ending and the main trend resuming."
        ),
        professional_use=(
            "This is a favorite setup among trend-following traders:\n"
            "- Lower risk entry: entering on pullback rather than breakout\n"
            "- Better R:R: closer stops possible at pullback extreme\n"
            "- Trend confirmation: main trend still intact (lines same side of zero)\n"
            "- Often combined with support/resistance for entry timing"
        ),
        signal_interpretation=(
            "BULLISH: MACD cross up with both lines < 0 (buy the dip in uptrend)\n"
            "BEARISH: MACD cross down with both lines > 0 (sell the rally in downtrend)\n"
            "Best when aligned with higher timeframe trend direction"
        )
    ),

    "EMA_200": SignalExplanation(
        name="200 Exponential Moving Average",
        category="trend",
        what_it_is=(
            "The 200 EMA is the most widely followed long-term trend indicator. "
            "It represents approximately 40 weeks of price data and is used globally "
            "by institutions to define the major trend direction."
        ),
        calculation=(
            "EMA = Price(t) x k + EMA(y) x (1-k)\n"
            "Where k = 2 / (N+1), N = 200\n"
            "Unlike SMA, EMA gives more weight to recent prices, making it more "
            "responsive while still smoothing out noise."
        ),
        professional_use=(
            "Institutional traders use the 200 EMA as:\n"
            "- Trend filter: only take longs above, shorts below\n"
            "- Dynamic support/resistance: price often bounces off 200 EMA\n"
            "- Trend strength: distance from 200 EMA indicates trend strength\n"
            "- Regime classification: above = bullish regime, below = bearish"
        ),
        signal_interpretation=(
            "BULLISH: Price above 200 EMA, confirmed for N consecutive bars\n"
            "BEARISH: Price below 200 EMA, confirmed for N consecutive bars\n"
            "CAUTION: Price crossing 200 EMA = potential regime change"
        )
    ),

    "STOCHASTIC": SignalExplanation(
        name="Stochastic Oscillator",
        category="momentum",
        what_it_is=(
            "The Stochastic Oscillator compares a closing price to its price range "
            "over a given period. Developed by George Lane in the 1950s, it measures "
            "where the close is relative to the high-low range."
        ),
        calculation=(
            "%K = 100 x (Close - Lowest Low) / (Highest High - Lowest Low)\n"
            "Period typically 14 bars for raw %K\n"
            "%D = 3-period SMA of %K (the 'signal' line)\n"
            "Slow Stochastic applies additional smoothing to reduce noise."
        ),
        professional_use=(
            "Traders use Stochastic for:\n"
            "- Overbought/oversold: >80 overbought, <20 oversold\n"
            "- %K/%D crossovers: %K crossing %D signals momentum shift\n"
            "- Divergence: price makes new extreme but Stochastic doesn't\n"
            "- Best in ranging markets; less reliable in strong trends"
        ),
        signal_interpretation=(
            "BULLISH: %K crosses above %D below 20 (oversold reversal)\n"
            "BEARISH: %K crosses below %D above 80 (overbought reversal)\n"
            "CAUTION: Can stay overbought/oversold for extended periods in trends"
        )
    ),

    "BOLLINGER_BANDS": SignalExplanation(
        name="Bollinger Bands",
        category="volatility",
        what_it_is=(
            "Bollinger Bands are volatility bands placed above and below a moving average. "
            "Created by John Bollinger in the 1980s, they adapt to volatility by widening "
            "during volatile periods and contracting during calm periods."
        ),
        calculation=(
            "Middle Band = 20-period SMA\n"
            "Upper Band = Middle Band + (2 x 20-period standard deviation)\n"
            "Lower Band = Middle Band - (2 x 20-period standard deviation)\n"
            "BB Width = (Upper - Lower) / Middle (measures squeeze)"
        ),
        professional_use=(
            "Professional applications include:\n"
            "- Squeeze detection: narrow bands = low volatility, often precedes big move\n"
            "- Mean reversion: price returning to middle band after touching outer\n"
            "- Trend riding: in strong trends, price 'walks the band'\n"
            "- Volatility trading: narrow BB = buy options, wide BB = sell options"
        ),
        signal_interpretation=(
            "SQUEEZE: BB Width at 6-month low = volatility expansion expected\n"
            "BREAKOUT: Close outside bands with expanding width = trend continuation\n"
            "REVERSAL: Close outside bands with contracting width = potential reversal"
        )
    ),

    "ADX": SignalExplanation(
        name="Average Directional Index (ADX)",
        category="trend",
        what_it_is=(
            "ADX measures trend strength regardless of direction. Part of the Directional "
            "Movement System by Wilder, it tells you HOW MUCH the market is trending, "
            "not which direction. Values range from 0 to 100."
        ),
        calculation=(
            "+DI = 100 x Smoothed +DM / ATR\n"
            "-DI = 100 x Smoothed -DM / ATR\n"
            "DX = 100 x |+DI - -DI| / (+DI + -DI)\n"
            "ADX = Smoothed average of DX over 14 periods"
        ),
        professional_use=(
            "Traders use ADX to:\n"
            "- Identify trending vs ranging markets: >25 trending, <20 ranging\n"
            "- Measure trend strength: rising ADX = strengthening trend\n"
            "- Choose strategy: trend-following when ADX high, mean-reversion when low\n"
            "- Time exits: falling ADX from high levels = trend weakening"
        ),
        signal_interpretation=(
            "STRONG TREND: ADX > 25 with +DI > -DI (bullish) or -DI > +DI (bearish)\n"
            "RANGING: ADX < 20, use oscillator-based strategies\n"
            "TREND STARTING: ADX rising from below 20, prepare for breakout"
        )
    ),

    "SUPPORT_RESISTANCE": SignalExplanation(
        name="Support & Resistance Zones",
        category="support_resistance",
        what_it_is=(
            "Support and resistance are price levels where buying or selling pressure "
            "has historically been strong enough to halt or reverse price movement. "
            "These form due to market memory and clustering of orders."
        ),
        calculation=(
            "Identified through multiple methods:\n"
            "- Swing highs/lows: local price extremes\n"
            "- Volume clusters: price levels with high trading activity\n"
            "- Round numbers: psychological levels (1.1000, 1.2000)\n"
            "- DBSCAN clustering: algorithmic identification of price clusters"
        ),
        professional_use=(
            "Institutions use S/R for:\n"
            "- Entry/exit points: buy at support, sell at resistance\n"
            "- Stop placement: stops beyond S/R levels\n"
            "- Breakout trading: breaks of major S/R signal trend changes\n"
            "- Target setting: next S/R level as profit target"
        ),
        signal_interpretation=(
            "BULLISH: Price bounces off support zone with confirming momentum\n"
            "BEARISH: Price rejects from resistance zone with confirming momentum\n"
            "BREAKOUT: Price closes beyond zone = potential trend continuation"
        )
    ),

    "SMA_ALIGNMENT": SignalExplanation(
        name="SMA Trend Alignment",
        category="trend",
        what_it_is=(
            "SMA alignment refers to the relative positioning of multiple simple moving "
            "averages. When shorter SMAs are above longer ones (20 > 50 > 200), it indicates "
            "a bullish trend structure; the reverse indicates bearish."
        ),
        calculation=(
            "SMA = Sum of closing prices over N periods / N\n"
            "Common periods: 20 (short-term), 50 (medium), 200 (long-term)\n"
            "Alignment score based on relative positioning and spacing."
        ),
        professional_use=(
            "Used by trend traders for:\n"
            "- Trend confirmation: aligned SMAs confirm trend direction\n"
            "- Trade filtering: only take trades in direction of SMA alignment\n"
            "- Dynamic support: price often bounces off aligned SMAs\n"
            "- Trend transitions: SMA crossovers signal potential trend changes"
        ),
        signal_interpretation=(
            "BULLISH: SMA 20 > SMA 50 > SMA 200 (perfect alignment)\n"
            "BEARISH: SMA 20 < SMA 50 < SMA 200 (perfect alignment)\n"
            "MIXED: SMAs not aligned = ranging or transitioning market"
        )
    ),
}


def get_signal_explanation(signal_name: str) -> SignalExplanation:
    """Get explanation for a specific signal type."""
    # Try exact match first
    if signal_name in SIGNAL_EXPLANATIONS:
        return SIGNAL_EXPLANATIONS[signal_name]

    # Try to match by partial name
    name_upper = signal_name.upper()
    for key, explanation in SIGNAL_EXPLANATIONS.items():
        if key in name_upper or name_upper in key:
            return explanation

    # Return a generic explanation if not found
    return SignalExplanation(
        name=signal_name,
        category="other",
        what_it_is="Technical indicator used for market analysis.",
        calculation="Varies by implementation.",
        professional_use="Used for identifying trading opportunities.",
        signal_interpretation="Depends on market context and other confirmations."
    )


def get_all_explanations() -> Dict[str, SignalExplanation]:
    """Get all signal explanations."""
    return SIGNAL_EXPLANATIONS


def get_explanations_by_category(category: str) -> Dict[str, SignalExplanation]:
    """Get explanations filtered by category."""
    return {
        name: exp for name, exp in SIGNAL_EXPLANATIONS.items()
        if exp.category == category
    }
