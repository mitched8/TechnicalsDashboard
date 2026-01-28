"""Trade setup generation for FX Trading Dashboard."""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
import pandas as pd
import numpy as np

from core.support_resistance import SRResult, ZoneType
from core.regime import MarketRegime, MultiTimeframeContext, TrendDirection, get_bias_from_context
from utils.forex_utils import calculate_pips
from config import TRADE_CONFIG, CONFLUENCE_WEIGHTS, ExitStrategySettings


class Bias(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class TradeSetup:
    """Actionable trade setup with entry, stop, and targets."""
    bias: Bias
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    target_1: float
    target_2: Optional[float]
    target_3: Optional[float]
    risk_pips: float
    reward_1_pips: float
    reward_2_pips: Optional[float]
    reward_3_pips: Optional[float]
    risk_reward_1: float
    risk_reward_2: Optional[float]
    risk_reward_3: Optional[float]
    invalidation: float
    confluence_score: float
    reasoning: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Check if setup meets minimum criteria."""
        if self.bias == Bias.NEUTRAL:
            return False
        min_rr = TRADE_CONFIG.get("min_rr_ratio", 1.5)
        return self.risk_reward_1 >= min_rr


def generate_trade_setup(
    sr_result: SRResult,
    regime: MarketRegime,
    mtf_context: MultiTimeframeContext,
    atr: float,
    pip_decimal: int = 4
) -> TradeSetup:
    """
    Generate actionable trade setup based on S/R and regime analysis.

    Logic:
    - If MTF aligned bullish AND near support: Long setup
    - If MTF aligned bearish AND near resistance: Short setup
    - If ranging: Look for mean reversion at extremes
    - Calculate stops below/above next S/R level
    - Targets at subsequent S/R levels

    Args:
        sr_result: Support/resistance analysis result
        regime: Current market regime
        mtf_context: Multi-timeframe context
        atr: Current ATR value
        pip_decimal: Pip decimal places (4 or 2)

    Returns:
        TradeSetup object
    """
    current_price = sr_result.current_price
    next_support = sr_result.next_support
    next_resistance = sr_result.next_resistance

    # Default neutral setup
    neutral_setup = TradeSetup(
        bias=Bias.NEUTRAL,
        entry_zone_low=0, entry_zone_high=0,
        stop_loss=0, target_1=0, target_2=None, target_3=None,
        risk_pips=0, reward_1_pips=0, reward_2_pips=None, reward_3_pips=None,
        risk_reward_1=0, risk_reward_2=None, risk_reward_3=None,
        invalidation=0, confluence_score=0,
        reasoning=["No clear setup - insufficient S/R levels detected"]
    )

    if not next_support or not next_resistance:
        return neutral_setup

    reasoning = []
    confluence_score = 0.0

    # Determine bias from MTF context
    bias_str = get_bias_from_context(mtf_context)

    if bias_str == "long":
        bias = Bias.LONG
        confluence_score += mtf_context.alignment_score * CONFLUENCE_WEIGHTS.get("mtf_alignment", 2.0)
        if mtf_context.alignment == "aligned_bullish":
            reasoning.append(f"MTF Bullish Alignment ({mtf_context.alignment_score:.0%})")
        else:
            reasoning.append(f"Strongest timeframe is bullish (ADX: {mtf_context.dominant_regime.trend_strength:.1f})")
    elif bias_str == "short":
        bias = Bias.SHORT
        confluence_score += mtf_context.alignment_score * CONFLUENCE_WEIGHTS.get("mtf_alignment", 2.0)
        if mtf_context.alignment == "aligned_bearish":
            reasoning.append(f"MTF Bearish Alignment ({mtf_context.alignment_score:.0%})")
        else:
            reasoning.append(f"Strongest timeframe is bearish (ADX: {mtf_context.dominant_regime.trend_strength:.1f})")
    else:
        # Check if we're at extremes for mean reversion
        dist_to_support = current_price - next_support.center
        dist_to_resistance = next_resistance.center - current_price

        if dist_to_support < dist_to_resistance * 0.3:
            bias = Bias.LONG
            reasoning.append("Near support in ranging market - potential bounce")
        elif dist_to_resistance < dist_to_support * 0.3:
            bias = Bias.SHORT
            reasoning.append("Near resistance in ranging market - potential rejection")
        else:
            return TradeSetup(
                bias=Bias.NEUTRAL,
                entry_zone_low=0, entry_zone_high=0,
                stop_loss=0, target_1=0, target_2=None, target_3=None,
                risk_pips=0, reward_1_pips=0, reward_2_pips=None, reward_3_pips=None,
                risk_reward_1=0, risk_reward_2=None, risk_reward_3=None,
                invalidation=0, confluence_score=0,
                reasoning=["No clear bias - price in middle of range, wait for better setup"]
            )

    # Build setup based on bias
    atr_buffer = atr * TRADE_CONFIG.get("atr_buffer_multiplier", 0.5)

    if bias == Bias.LONG:
        entry_zone_low = next_support.center
        entry_zone_high = next_support.upper_bound + atr_buffer
        stop_loss = next_support.lower_bound - atr_buffer
        target_1 = next_resistance.center
        target_2 = sr_result.resistances[1].center if len(sr_result.resistances) > 1 else None
        target_3 = sr_result.resistances[2].center if len(sr_result.resistances) > 2 else None
        invalidation = next_support.lower_bound

        confluence_score += next_support.strength * 0.3
        reasoning.append(f"Entry at support zone: {next_support.center:.5f}")
        reasoning.append(f"Support sources: {', '.join(next_support.sources)}")

    else:  # SHORT
        entry_zone_low = next_resistance.lower_bound - atr_buffer
        entry_zone_high = next_resistance.center
        stop_loss = next_resistance.upper_bound + atr_buffer
        target_1 = next_support.center
        target_2 = sr_result.supports[1].center if len(sr_result.supports) > 1 else None
        target_3 = sr_result.supports[2].center if len(sr_result.supports) > 2 else None
        invalidation = next_resistance.upper_bound

        confluence_score += next_resistance.strength * 0.3
        reasoning.append(f"Entry at resistance zone: {next_resistance.center:.5f}")
        reasoning.append(f"Resistance sources: {', '.join(next_resistance.sources)}")

    # Calculate risk/reward
    entry_mid = (entry_zone_low + entry_zone_high) / 2
    risk_pips = abs(calculate_pips(entry_mid, stop_loss, pip_decimal))
    reward_1_pips = abs(calculate_pips(entry_mid, target_1, pip_decimal))
    reward_2_pips = abs(calculate_pips(entry_mid, target_2, pip_decimal)) if target_2 else None
    reward_3_pips = abs(calculate_pips(entry_mid, target_3, pip_decimal)) if target_3 else None

    rr_1 = reward_1_pips / risk_pips if risk_pips > 0 else 0
    rr_2 = reward_2_pips / risk_pips if reward_2_pips and risk_pips > 0 else None
    rr_3 = reward_3_pips / risk_pips if reward_3_pips and risk_pips > 0 else None

    reasoning.append(f"Risk: {risk_pips:.0f} pips | Reward T1: {reward_1_pips:.0f} pips | R:R = {rr_1:.2f}")

    # Regime bonus
    if regime.trend == TrendDirection.UP and bias == Bias.LONG:
        confluence_score += CONFLUENCE_WEIGHTS.get("regime_favorable", 1.5)
        reasoning.append("Daily regime supports long bias")
    elif regime.trend == TrendDirection.DOWN and bias == Bias.SHORT:
        confluence_score += CONFLUENCE_WEIGHTS.get("regime_favorable", 1.5)
        reasoning.append("Daily regime supports short bias")

    return TradeSetup(
        bias=bias,
        entry_zone_low=entry_zone_low,
        entry_zone_high=entry_zone_high,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        target_3=target_3,
        risk_pips=risk_pips,
        reward_1_pips=reward_1_pips,
        reward_2_pips=reward_2_pips,
        reward_3_pips=reward_3_pips,
        risk_reward_1=rr_1,
        risk_reward_2=rr_2,
        risk_reward_3=rr_3,
        invalidation=invalidation,
        confluence_score=confluence_score,
        reasoning=reasoning,
    )


# ============================================================================
# NEW EXIT STRATEGIES FROM MACD_EMA_TREND
# ============================================================================

@dataclass
class TrailingStopState:
    """State for managing trailing ATR stop."""
    entry_price: float
    direction: str  # "long" or "short"
    initial_stop: float
    current_stop: float
    highest_since_entry: float  # For long positions
    lowest_since_entry: float   # For short positions
    atr_multiplier: float
    atr_at_entry: float
    is_active: bool = True

    def get_stop_distance(self) -> float:
        """Get current stop distance in price."""
        return abs(self.entry_price - self.current_stop)


def calculate_trailing_atr_stop(
    entry_price: float,
    direction: str,
    atr: float,
    multiplier: float = 3.75,
    highest_high: Optional[float] = None,
    lowest_low: Optional[float] = None,
    current_stop: Optional[float] = None
) -> float:
    """
    Calculate trailing ATR-based stop loss.

    From MACD_EMA_Trend strategy: The ATR trailing stop only ratchets tighter,
    it never loosens. This preserves profits while letting winners run.

    Args:
        entry_price: Trade entry price
        direction: "long" or "short"
        atr: Current ATR value
        multiplier: ATR multiplier (optimal: 3.75 from backtest)
        highest_high: Highest high since entry (for longs)
        lowest_low: Lowest low since entry (for shorts)
        current_stop: Current stop level (for ratchet comparison)

    Returns:
        New stop loss level
    """
    atr_distance = atr * multiplier

    if direction == "long":
        # For longs: trail below the highest high since entry
        reference_price = highest_high if highest_high else entry_price
        new_stop = reference_price - atr_distance

        # Only ratchet tighter (stop can only move UP for longs)
        if current_stop and new_stop < current_stop:
            return current_stop
        return new_stop

    else:  # short
        # For shorts: trail above the lowest low since entry
        reference_price = lowest_low if lowest_low else entry_price
        new_stop = reference_price + atr_distance

        # Only ratchet tighter (stop can only move DOWN for shorts)
        if current_stop and new_stop > current_stop:
            return current_stop
        return new_stop


def calculate_swing_based_stop(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 8,
    atr: Optional[float] = None,
    atr_multiplier: float = 2.0
) -> float:
    """
    Calculate swing-based stop loss.

    From MACD_EMA_Trend strategy: Uses swing high/low as stop, with optional
    ATR fallback if swing is too close to entry.

    Args:
        df: DataFrame with OHLC data
        direction: "long" or "short"
        lookback: Bars to look back for swing high/low
        atr: ATR value for hybrid calculation
        atr_multiplier: Multiplier for ATR-based stop

    Returns:
        Stop loss level
    """
    if len(df) < lookback:
        lookback = len(df)

    recent = df.tail(lookback)
    current_price = float(df['Close'].iloc[-1])

    if direction == "long":
        # Stop below recent swing low
        swing_stop = float(recent['Low'].min())

        # Hybrid: use max of swing low and ATR-based stop
        if atr:
            atr_stop = current_price - (atr * atr_multiplier)
            return min(swing_stop, atr_stop)  # Use the one that gives tighter stop
        return swing_stop

    else:  # short
        # Stop above recent swing high
        swing_stop = float(recent['High'].max())

        # Hybrid: use min of swing high and ATR-based stop
        if atr:
            atr_stop = current_price + (atr * atr_multiplier)
            return max(swing_stop, atr_stop)  # Use the one that gives tighter stop
        return swing_stop


def update_trailing_stop_state(
    state: TrailingStopState,
    current_high: float,
    current_low: float,
    current_atr: float
) -> TrailingStopState:
    """
    Update trailing stop state with new price data.

    This should be called on each new bar to update the trailing stop.

    Args:
        state: Current trailing stop state
        current_high: Current bar's high
        current_low: Current bar's low
        current_atr: Current ATR value

    Returns:
        Updated TrailingStopState
    """
    if not state.is_active:
        return state

    # Update highest/lowest since entry
    if state.direction == "long":
        state.highest_since_entry = max(state.highest_since_entry, current_high)
        new_stop = calculate_trailing_atr_stop(
            entry_price=state.entry_price,
            direction="long",
            atr=current_atr,
            multiplier=state.atr_multiplier,
            highest_high=state.highest_since_entry,
            current_stop=state.current_stop
        )
    else:  # short
        state.lowest_since_entry = min(state.lowest_since_entry, current_low)
        new_stop = calculate_trailing_atr_stop(
            entry_price=state.entry_price,
            direction="short",
            atr=current_atr,
            multiplier=state.atr_multiplier,
            lowest_low=state.lowest_since_entry,
            current_stop=state.current_stop
        )

    state.current_stop = new_stop
    return state


def create_trailing_stop_state(
    entry_price: float,
    direction: str,
    atr: float,
    multiplier: float = 3.75
) -> TrailingStopState:
    """
    Create initial trailing stop state for a new trade.

    Args:
        entry_price: Trade entry price
        direction: "long" or "short"
        atr: ATR at entry
        multiplier: ATR multiplier

    Returns:
        Initial TrailingStopState
    """
    initial_stop = calculate_trailing_atr_stop(
        entry_price=entry_price,
        direction=direction,
        atr=atr,
        multiplier=multiplier
    )

    return TrailingStopState(
        entry_price=entry_price,
        direction=direction,
        initial_stop=initial_stop,
        current_stop=initial_stop,
        highest_since_entry=entry_price if direction == "long" else float('inf'),
        lowest_since_entry=entry_price if direction == "short" else 0,
        atr_multiplier=multiplier,
        atr_at_entry=atr,
        is_active=True
    )


def apply_exit_strategy(
    setup: TradeSetup,
    df: pd.DataFrame,
    atr: float,
    exit_settings: Optional[ExitStrategySettings] = None,
    pip_decimal: int = 4
) -> TradeSetup:
    """
    Apply exit strategy settings to modify the trade setup's stop loss.

    Args:
        setup: Original TradeSetup
        df: DataFrame with OHLC data
        atr: Current ATR value
        exit_settings: Exit strategy settings
        pip_decimal: Pip decimal places

    Returns:
        Modified TradeSetup with updated stop loss
    """
    if exit_settings is None:
        exit_settings = ExitStrategySettings()

    if setup.bias == Bias.NEUTRAL:
        return setup

    direction = "long" if setup.bias == Bias.LONG else "short"
    entry_mid = (setup.entry_zone_low + setup.entry_zone_high) / 2

    # Determine new stop based on exit strategy
    if exit_settings.exit_strategy == "trailing_atr":
        new_stop = calculate_trailing_atr_stop(
            entry_price=entry_mid,
            direction=direction,
            atr=atr,
            multiplier=exit_settings.trailing_atr_multiplier
        )
        setup.reasoning.append(
            f"Using trailing ATR stop (ATR x {exit_settings.trailing_atr_multiplier})"
        )

    elif exit_settings.exit_strategy == "swing_based":
        if exit_settings.use_swing_atr_hybrid:
            new_stop = calculate_swing_based_stop(
                df=df,
                direction=direction,
                lookback=exit_settings.swing_lookback,
                atr=atr,
                atr_multiplier=2.0
            )
            setup.reasoning.append(
                f"Using swing-based stop (hybrid with ATR, lookback={exit_settings.swing_lookback})"
            )
        else:
            new_stop = calculate_swing_based_stop(
                df=df,
                direction=direction,
                lookback=exit_settings.swing_lookback
            )
            setup.reasoning.append(
                f"Using swing-based stop (lookback={exit_settings.swing_lookback} bars)"
            )

    else:  # fixed_targets - keep original stop
        return setup

    # Update the setup with new stop
    setup.stop_loss = new_stop
    setup.risk_pips = abs(calculate_pips(entry_mid, new_stop, pip_decimal))

    # Recalculate R:R ratios
    if setup.risk_pips > 0:
        setup.risk_reward_1 = setup.reward_1_pips / setup.risk_pips
        if setup.reward_2_pips:
            setup.risk_reward_2 = setup.reward_2_pips / setup.risk_pips
        if setup.reward_3_pips:
            setup.risk_reward_3 = setup.reward_3_pips / setup.risk_pips

    return setup
