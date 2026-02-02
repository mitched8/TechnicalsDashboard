"""Professional Support/Resistance Analysis Module.

Implements advanced S/R concepts including:
- Liquidity consumption model (discovery vs depletion phases)
- Multi-timeframe hierarchy scoring
- Reaction quality analysis (rejection vs acceptance)
- Compression pattern detection
- Touch pattern analysis with diminishing bounce detection
- Comprehensive confluence scoring

Based on professional FX trading S/R methodology.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from scipy.signal import argrelextrema

from core.support_resistance import (
    ZoneType, PriceZone, detect_swing_levels,
    detect_round_numbers, cluster_and_merge_levels
)
from config import SR_SCORING_CONFIG


class LiquidityPhase(Enum):
    """Phase in the liquidity consumption model."""
    DISCOVERY = "discovery"      # 1-3 tests, level gaining strength
    ESTABLISHED = "established"  # Level proven, still holding
    DEPLETION = "depletion"      # Many tests, liquidity being consumed
    EXHAUSTED = "exhausted"      # Level likely to break


class ReactionQuality(Enum):
    """Quality of price reaction at a level."""
    STRONG_REJECTION = "strong_rejection"  # Long wick, closed away
    MILD_REJECTION = "mild_rejection"      # Some wick, closed near level
    ACCEPTANCE = "acceptance"              # Closed through level
    CLEAN_BREAK = "clean_break"            # Broke through with momentum


class ApproachPattern(Enum):
    """How price is approaching a level."""
    COMPRESSION = "compression"    # Higher lows into resistance / lower highs into support
    IMPULSE = "impulse"            # Strong move toward level
    DRIFT = "drift"                # Slow, weak approach
    RETEST = "retest"              # Coming back to test after break


@dataclass
class TouchEvent:
    """Record of price touching a level."""
    timestamp: pd.Timestamp
    bar_index: int
    touch_price: float
    high: float
    low: float
    close: float
    reaction_quality: ReactionQuality
    bounce_size_pct: float  # How much price bounced (as % of ATR)
    wick_size_pct: float    # Wick size as % of total bar range
    held: bool              # Did the level hold?


@dataclass
class EnhancedZone:
    """Enhanced S/R zone with professional analysis."""
    # Basic zone info
    zone_type: ZoneType
    center: float
    upper_bound: float
    lower_bound: float

    # Scoring
    total_score: float
    score_breakdown: Dict[str, float]

    # Liquidity analysis
    liquidity_phase: LiquidityPhase
    touch_count: int
    touches: List[TouchEvent] = field(default_factory=list)

    # Touch clustering (for liquidity replenishment analysis)
    recent_cluster_size: int = 0      # Touches in most recent cluster
    total_clusters: int = 0           # Number of separate test clusters
    has_replenished: bool = False     # Level has had time to replenish

    # Timeframe info
    source_timeframe: str
    timeframe_score: float

    # Pattern detection
    approach_pattern: Optional[ApproachPattern] = None
    compression_bars: int = 0

    # Confluence
    sources: List[str] = field(default_factory=list)
    is_round_number: bool = False
    is_role_flip: bool = False

    # Age info
    first_formed_bars_ago: int = 0
    last_tested_bars_ago: int = 0

    @property
    def width(self) -> float:
        return self.upper_bound - self.lower_bound

    @property
    def score_category(self) -> str:
        """Get human-readable score category."""
        if self.total_score >= 8:
            return "Prime"
        elif self.total_score >= 6:
            return "Strong"
        elif self.total_score >= 4:
            return "Moderate"
        else:
            return "Weak"

    @property
    def liquidity_description(self) -> str:
        """Get description of liquidity state with replenishment context."""
        base_descriptions = {
            LiquidityPhase.DISCOVERY: "Fresh level, gaining strength",
            LiquidityPhase.ESTABLISHED: "Proven level, likely to hold",
            LiquidityPhase.DEPLETION: "Rapid tests, liquidity being consumed",
            LiquidityPhase.EXHAUSTED: "High break probability",
        }
        desc = base_descriptions.get(self.liquidity_phase, "Unknown")

        # Add replenishment context
        if self.has_replenished and self.total_clusters > 1:
            desc += f" (replenished {self.total_clusters - 1}x)"
        elif self.recent_cluster_size > 0:
            desc += f" ({self.recent_cluster_size} recent tests)"

        return desc

    @property
    def diminishing_bounces(self) -> bool:
        """Check if bounces are getting smaller (break warning)."""
        if len(self.touches) < 3:
            return False
        recent_bounces = [t.bounce_size_pct for t in self.touches[-3:]]
        return all(recent_bounces[i] > recent_bounces[i+1] for i in range(len(recent_bounces)-1))


@dataclass
class SRAnalysisResult:
    """Complete S/R analysis result."""
    supports: List[EnhancedZone]
    resistances: List[EnhancedZone]
    current_price: float

    # Key levels
    nearest_support: Optional[EnhancedZone]
    nearest_resistance: Optional[EnhancedZone]
    strongest_support: Optional[EnhancedZone]
    strongest_resistance: Optional[EnhancedZone]

    # Compression alerts
    compression_into_resistance: bool = False
    compression_into_support: bool = False

    # Multi-timeframe data
    weekly_levels: List[EnhancedZone] = field(default_factory=list)
    daily_levels: List[EnhancedZone] = field(default_factory=list)
    h4_levels: List[EnhancedZone] = field(default_factory=list)


# ==============================================================================
# SCORING CONFIGURATION
# ==============================================================================

TIMEFRAME_WEIGHTS = {
    "weekly": 3.0,
    "daily": 2.0,
    "4h": 1.0,
    "1h": 0.5,
}

# Use imported config from config.py for centralized configuration
SCORING_CONFIG = SR_SCORING_CONFIG


# ==============================================================================
# CORE ANALYSIS FUNCTIONS
# ==============================================================================

def analyze_touch_quality(
    data: pd.DataFrame,
    level: float,
    zone_width: float,
    atr: pd.Series = None
) -> List[TouchEvent]:
    """
    Analyze all touches of a price level and their quality.

    Args:
        data: OHLCV DataFrame
        level: Price level to analyze
        zone_width: Width of the zone around level
        atr: ATR series for bounce normalization

    Returns:
        List of TouchEvent objects
    """
    touches = []

    half_zone = zone_width / 2
    zone_upper = level + half_zone
    zone_lower = level - half_zone

    # Calculate ATR if not provided
    if atr is None:
        high_low = data['High'] - data['Low']
        atr = high_low.rolling(14).mean()

    for i in range(len(data)):
        row = data.iloc[i]
        high, low, close, open_price = row['High'], row['Low'], row['Close'], row['Open']

        # Check if bar touched the zone
        touched_from_above = low <= zone_upper and high > zone_upper
        touched_from_below = high >= zone_lower and low < zone_lower

        if not (touched_from_above or touched_from_below):
            continue

        # Calculate bar metrics
        bar_range = high - low
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else bar_range

        # Determine reaction quality
        if touched_from_above:  # Testing as support
            wick_below = min(open_price, close) - low
            body_position = (close - low) / bar_range if bar_range > 0 else 0.5

            if close < zone_lower:
                reaction = ReactionQuality.CLEAN_BREAK
                held = False
            elif close < level:
                reaction = ReactionQuality.ACCEPTANCE
                held = False
            elif wick_below / bar_range > 0.6 and close > level:
                reaction = ReactionQuality.STRONG_REJECTION
                held = True
            else:
                reaction = ReactionQuality.MILD_REJECTION
                held = True

            bounce_size = (close - low) / current_atr if current_atr > 0 else 0
            wick_pct = wick_below / bar_range if bar_range > 0 else 0

        else:  # Testing as resistance
            wick_above = high - max(open_price, close)
            body_position = (high - close) / bar_range if bar_range > 0 else 0.5

            if close > zone_upper:
                reaction = ReactionQuality.CLEAN_BREAK
                held = False
            elif close > level:
                reaction = ReactionQuality.ACCEPTANCE
                held = False
            elif wick_above / bar_range > 0.6 and close < level:
                reaction = ReactionQuality.STRONG_REJECTION
                held = True
            else:
                reaction = ReactionQuality.MILD_REJECTION
                held = True

            bounce_size = (high - close) / current_atr if current_atr > 0 else 0
            wick_pct = wick_above / bar_range if bar_range > 0 else 0

        touch = TouchEvent(
            timestamp=data.index[i],
            bar_index=i,
            touch_price=level,
            high=high,
            low=low,
            close=close,
            reaction_quality=reaction,
            bounce_size_pct=bounce_size,
            wick_size_pct=wick_pct,
            held=held
        )
        touches.append(touch)

    return touches


def calculate_touch_clustering(
    touches: List[TouchEvent],
    replenish_threshold_bars: int = 25
) -> Tuple[int, List[List[TouchEvent]]]:
    """
    Analyze touch clustering to determine if liquidity has time to replenish.

    Touches that are spaced far apart (> threshold bars) allow liquidity to
    replenish. Only clustered touches consume liquidity.

    Args:
        touches: List of TouchEvent objects sorted by time
        replenish_threshold_bars: Bars between touches for liquidity to replenish

    Returns:
        (effective_touch_count, list_of_touch_clusters)

    Example:
        Touches at bars [10, 15, 18, 80, 85] with threshold=25:
        - Cluster 1: [10, 15, 18] - rapid tests, consuming liquidity
        - Cluster 2: [80, 85] - new cluster after replenishment
        - Effective count for depletion = max cluster size = 3
    """
    if not touches:
        return 0, []

    # Sort by bar index
    sorted_touches = sorted(touches, key=lambda t: t.bar_index)

    clusters = []
    current_cluster = [sorted_touches[0]]

    for i in range(1, len(sorted_touches)):
        bars_since_last = sorted_touches[i].bar_index - sorted_touches[i-1].bar_index

        if bars_since_last <= replenish_threshold_bars:
            # Close together - same cluster, liquidity being consumed
            current_cluster.append(sorted_touches[i])
        else:
            # Gap is large enough for liquidity to replenish
            # Start a new cluster
            clusters.append(current_cluster)
            current_cluster = [sorted_touches[i]]

    # Don't forget the last cluster
    clusters.append(current_cluster)

    # Effective touch count is the size of the most recent cluster
    # (since earlier clusters had time to replenish)
    recent_cluster_size = len(clusters[-1]) if clusters else 0

    return recent_cluster_size, clusters


def determine_liquidity_phase(
    touches: List[TouchEvent],
    replenish_threshold_bars: int = 25
) -> LiquidityPhase:
    """
    Determine the liquidity phase based on touch pattern WITH time spacing.

    Key insight: Liquidity replenishes over time. A level tested 5 times over
    2 years is still strong. A level tested 5 times in 2 weeks is depleted.

    Args:
        touches: List of touch events
        replenish_threshold_bars: Bars between touches to consider replenishment
            (25 bars ~= 5 weeks on daily, ~= 1 week on 4H)

    Phases:
        Discovery: Recent cluster has 1-3 tests, level gaining strength
        Established: Proven level with quality tests, not over-tested recently
        Depletion: Recent cluster has many rapid tests, liquidity being consumed
        Exhausted: Very high break probability (rapid tests + diminishing bounces)
    """
    if not touches:
        return LiquidityPhase.DISCOVERY

    # Analyze clustering
    effective_touches, clusters = calculate_touch_clustering(touches, replenish_threshold_bars)
    recent_cluster = clusters[-1] if clusters else []

    # Get valid (held) touches in recent cluster
    valid_in_cluster = [t for t in recent_cluster if t.held]
    cluster_held_ratio = len(valid_in_cluster) / len(recent_cluster) if recent_cluster else 1.0

    # Check for diminishing bounces in the recent cluster
    diminishing = False
    if len(valid_in_cluster) >= 3:
        recent_bounces = [t.bounce_size_pct for t in valid_in_cluster[-3:]]
        diminishing = all(
            recent_bounces[i] >= recent_bounces[i+1]
            for i in range(len(recent_bounces)-1)
        )

    # Also check if bounces in current cluster are smaller than previous clusters
    cross_cluster_diminishing = False
    if len(clusters) >= 2 and valid_in_cluster:
        prev_cluster = clusters[-2]
        prev_valid = [t for t in prev_cluster if t.held]
        if prev_valid:
            prev_avg_bounce = sum(t.bounce_size_pct for t in prev_valid) / len(prev_valid)
            curr_avg_bounce = sum(t.bounce_size_pct for t in valid_in_cluster) / len(valid_in_cluster)
            cross_cluster_diminishing = curr_avg_bounce < prev_avg_bounce * 0.7  # 30% smaller

    # Determine phase based on recent cluster behavior
    if effective_touches <= 3:
        if cluster_held_ratio >= 0.6:
            # Fresh tests or replenished level
            if len(clusters) > 1:
                # Has history but recently replenished
                return LiquidityPhase.ESTABLISHED
            return LiquidityPhase.DISCOVERY
        else:
            # Even few tests are failing - weak level
            return LiquidityPhase.DEPLETION

    elif effective_touches <= 6:
        if cluster_held_ratio >= 0.7 and not diminishing:
            return LiquidityPhase.ESTABLISHED
        elif diminishing or cross_cluster_diminishing:
            return LiquidityPhase.DEPLETION
        else:
            return LiquidityPhase.ESTABLISHED

    else:  # Many rapid touches in recent cluster
        if diminishing or cross_cluster_diminishing or cluster_held_ratio < 0.5:
            return LiquidityPhase.EXHAUSTED
        else:
            return LiquidityPhase.DEPLETION


def detect_compression_pattern(
    data: pd.DataFrame,
    level: float,
    zone_type: ZoneType,
    lookback: int = 20
) -> Tuple[ApproachPattern, int]:
    """
    Detect if price is compressing into a level.

    Compression into resistance: Higher lows forming
    Compression into support: Lower highs forming

    Args:
        data: OHLCV DataFrame
        level: The S/R level
        zone_type: SUPPORT or RESISTANCE
        lookback: Bars to analyze

    Returns:
        (ApproachPattern, compression_bar_count)
    """
    if len(data) < lookback:
        lookback = len(data)

    recent_data = data.tail(lookback)
    current_price = recent_data['Close'].iloc[-1]

    # Calculate distance to level
    distance_pct = abs(current_price - level) / level

    # Only check compression if price is approaching
    if distance_pct > 0.02:  # More than 2% away
        return ApproachPattern.DRIFT, 0

    # Find swing points
    window = 3
    if len(recent_data) < window * 2 + 1:
        return ApproachPattern.DRIFT, 0

    highs = recent_data['High'].values
    lows = recent_data['Low'].values

    swing_high_idx = argrelextrema(highs, np.greater_equal, order=window)[0]
    swing_low_idx = argrelextrema(lows, np.less_equal, order=window)[0]

    if zone_type == ZoneType.RESISTANCE:
        # Check for higher lows (compression into resistance)
        if len(swing_low_idx) >= 3:
            recent_lows = [lows[i] for i in swing_low_idx[-3:]]
            if all(recent_lows[i] < recent_lows[i+1] for i in range(len(recent_lows)-1)):
                compression_bars = lookback - swing_low_idx[-3] if len(swing_low_idx) >= 3 else 0
                return ApproachPattern.COMPRESSION, compression_bars
    else:
        # Check for lower highs (compression into support)
        if len(swing_high_idx) >= 3:
            recent_highs = [highs[i] for i in swing_high_idx[-3:]]
            if all(recent_highs[i] > recent_highs[i+1] for i in range(len(recent_highs)-1)):
                compression_bars = lookback - swing_high_idx[-3] if len(swing_high_idx) >= 3 else 0
                return ApproachPattern.COMPRESSION, compression_bars

    # Check for impulse move
    recent_range = recent_data['High'].max() - recent_data['Low'].min()
    atr = (data['High'] - data['Low']).rolling(14).mean().iloc[-1]

    if recent_range > atr * 3:
        return ApproachPattern.IMPULSE, 0

    return ApproachPattern.DRIFT, 0


def check_role_flip(
    data: pd.DataFrame,
    level: float,
    zone_width: float,
    current_zone_type: ZoneType
) -> bool:
    """
    Check if a level has flipped roles (support became resistance or vice versa).

    Role flip adds significant strength to a level.
    """
    half_zone = zone_width / 2
    zone_upper = level + half_zone
    zone_lower = level - half_zone

    # Look for price action on both sides of the level
    above_count = 0
    below_count = 0
    flip_detected = False

    # Need to see price accept above and below the level at different times
    in_zone_above = False
    in_zone_below = False

    for i in range(len(data)):
        close = data['Close'].iloc[i]

        if close > zone_upper:
            if in_zone_below:  # Was below, now above = potential flip
                flip_detected = True
            in_zone_above = True
            in_zone_below = False
            above_count += 1
        elif close < zone_lower:
            if in_zone_above:  # Was above, now below = potential flip
                flip_detected = True
            in_zone_below = True
            in_zone_above = False
            below_count += 1

    # Need meaningful time on both sides
    min_bars = 5
    return flip_detected and above_count >= min_bars and below_count >= min_bars


def calculate_zone_score(
    zone: EnhancedZone,
    config: dict = None
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate comprehensive score for a zone.

    Returns:
        (total_score, score_breakdown_dict)
    """
    if config is None:
        config = SCORING_CONFIG

    breakdown = {}

    # 1. Timeframe significance (0-3 points)
    tf_key = f"tf_{zone.source_timeframe}"
    tf_score = config.get(tf_key, 1.0)
    breakdown["Timeframe"] = tf_score

    # 2. Structure type (0-2 points)
    structure_score = 0
    if "swing_high" in zone.sources or "swing_low" in zone.sources:
        structure_score += config["swing_high_low"]
    if "volume" in zone.sources:
        structure_score += config["volume_cluster"]
    if "bb_upper" in zone.sources or "bb_lower" in zone.sources:
        structure_score += config["bollinger_touch"]
    if zone.is_role_flip:
        structure_score += config["role_flip"]
    breakdown["Structure"] = min(structure_score, 4.0)  # Cap at 4

    # 3. Reaction quality (0-2 points based on recent touches)
    reaction_score = 0
    if zone.touches:
        recent_touches = zone.touches[-3:] if len(zone.touches) >= 3 else zone.touches
        for touch in recent_touches:
            if touch.reaction_quality == ReactionQuality.STRONG_REJECTION:
                reaction_score += config["strong_rejection"] / len(recent_touches)
            elif touch.reaction_quality == ReactionQuality.MILD_REJECTION:
                reaction_score += config["mild_rejection"] / len(recent_touches)
            elif touch.reaction_quality == ReactionQuality.ACCEPTANCE:
                reaction_score += config["acceptance"] / len(recent_touches)
    breakdown["Reactions"] = max(reaction_score, -1.0)  # Floor at -1

    # 4. Recency (0-1 points)
    if zone.last_tested_bars_ago <= 20:
        recency_score = config["recent_test"]
    elif zone.last_tested_bars_ago <= 60:
        recency_score = config["medium_test"]
    else:
        recency_score = config["old_test"]
    breakdown["Recency"] = recency_score

    # 5. Touch pattern / liquidity phase (can be negative)
    phase_scores = {
        LiquidityPhase.DISCOVERY: config["discovery_phase"],
        LiquidityPhase.ESTABLISHED: config["established_phase"],
        LiquidityPhase.DEPLETION: config["depletion_phase"],
        LiquidityPhase.EXHAUSTED: config["exhausted_phase"],
    }
    phase_score = phase_scores.get(zone.liquidity_phase, 0)
    breakdown["Touch Pattern"] = phase_score

    # 6. Approach pattern
    approach_score = 0
    if zone.approach_pattern == ApproachPattern.COMPRESSION:
        approach_score = config["compression"]
    elif zone.approach_pattern == ApproachPattern.IMPULSE:
        approach_score = config["impulse"]
    breakdown["Approach"] = approach_score

    # 7. Confluence bonuses
    confluence_score = 0
    if zone.is_round_number:
        confluence_score += config["round_number"]
    if len(zone.sources) > 1:
        confluence_score += config["multi_source"] * (len(zone.sources) - 1)
    breakdown["Confluence"] = confluence_score

    # Calculate total
    total = sum(breakdown.values())

    return total, breakdown


def detect_enhanced_levels(
    data: pd.DataFrame,
    timeframe: str = "daily",
    pip_step: float = 0.01,
    lookback: int = None
) -> List[EnhancedZone]:
    """
    Detect enhanced S/R zones with full professional analysis.

    Args:
        data: OHLCV DataFrame with indicators
        timeframe: Source timeframe ("weekly", "daily", "4h")
        pip_step: Round number step size
        lookback: Bars to analyze (None = all)

    Returns:
        List of EnhancedZone objects
    """
    if data.empty:
        return []

    if lookback and lookback < len(data):
        analysis_data = data.tail(lookback)
    else:
        analysis_data = data
        lookback = len(data)

    current_price = float(data['Close'].iloc[-1])

    # Calculate ATR for normalization
    atr = (data['High'] - data['Low']).rolling(14).mean()
    avg_atr = atr.mean()
    zone_width = avg_atr * 2  # Zone width based on ATR

    # 1. Detect swing levels
    swings = detect_swing_levels(analysis_data, window=5, lookback=lookback)

    # 2. Detect round numbers
    rounds = detect_round_numbers(current_price, pip_step, num_levels=5)

    # 3. Cluster levels
    all_levels = swings + rounds
    basic_zones = cluster_and_merge_levels(all_levels, threshold_pct=0.002)

    # 4. Enhance each zone with professional analysis
    enhanced_zones = []

    for basic_zone in basic_zones:
        # Analyze touches
        touches = analyze_touch_quality(
            analysis_data,
            basic_zone.center,
            zone_width,
            atr.tail(lookback)
        )

        # Determine liquidity phase with clustering analysis
        # Use different replenishment thresholds based on timeframe (from config)
        replenish_threshold = SCORING_CONFIG.get(
            f"replenish_threshold_{timeframe}",
            25  # Default fallback
        )

        liquidity_phase = determine_liquidity_phase(touches, replenish_threshold)

        # Get clustering info for display
        effective_touches, clusters = calculate_touch_clustering(touches, replenish_threshold)
        recent_cluster_size = effective_touches
        total_clusters = len(clusters)
        has_replenished = total_clusters > 1

        # Determine zone type
        if basic_zone.center < current_price:
            zone_type = ZoneType.SUPPORT
        else:
            zone_type = ZoneType.RESISTANCE

        # Check for compression
        approach_pattern, compression_bars = detect_compression_pattern(
            analysis_data, basic_zone.center, zone_type, lookback=20
        )

        # Check for role flip
        is_role_flip = check_role_flip(
            analysis_data, basic_zone.center, zone_width, zone_type
        )

        # Check if round number
        is_round = "round" in basic_zone.sources

        # Calculate age metrics
        first_formed = basic_zone.first_detected_idx
        last_tested = max(t.bar_index for t in touches) if touches else 0
        last_tested_bars_ago = len(data) - 1 - last_tested if touches else lookback

        # Create enhanced zone
        enhanced = EnhancedZone(
            zone_type=zone_type,
            center=basic_zone.center,
            upper_bound=basic_zone.upper_bound,
            lower_bound=basic_zone.lower_bound,
            total_score=0.0,  # Will be calculated
            score_breakdown={},
            liquidity_phase=liquidity_phase,
            touch_count=len(touches),
            touches=touches,
            recent_cluster_size=recent_cluster_size,
            total_clusters=total_clusters,
            has_replenished=has_replenished,
            source_timeframe=timeframe,
            timeframe_score=TIMEFRAME_WEIGHTS.get(timeframe, 1.0),
            approach_pattern=approach_pattern,
            compression_bars=compression_bars,
            sources=basic_zone.sources,
            is_round_number=is_round,
            is_role_flip=is_role_flip,
            first_formed_bars_ago=first_formed,
            last_tested_bars_ago=last_tested_bars_ago,
        )

        # Calculate score
        total_score, breakdown = calculate_zone_score(enhanced)
        enhanced.total_score = total_score
        enhanced.score_breakdown = breakdown

        enhanced_zones.append(enhanced)

    return enhanced_zones


def analyze_multi_timeframe_sr(
    data_dict: Dict[str, pd.DataFrame],
    pip_step: float = 0.01,
    lookback_config: Dict[str, int] = None
) -> SRAnalysisResult:
    """
    Perform comprehensive multi-timeframe S/R analysis.

    Args:
        data_dict: Dict with keys 'weekly', 'daily', '4h' containing OHLCV DataFrames
        pip_step: Round number step size
        lookback_config: Dict of lookback periods per timeframe

    Returns:
        SRAnalysisResult with all analysis data
    """
    if lookback_config is None:
        lookback_config = {
            "weekly": 104,   # 2 years
            "daily": 504,    # 2 years
            "4h": 252,       # ~2 months of 4h bars
        }

    # Get current price from daily data
    daily_data = data_dict.get("daily", pd.DataFrame())
    if daily_data.empty:
        return SRAnalysisResult(
            supports=[], resistances=[], current_price=0,
            nearest_support=None, nearest_resistance=None,
            strongest_support=None, strongest_resistance=None
        )

    current_price = float(daily_data['Close'].iloc[-1])

    # Analyze each timeframe
    weekly_zones = []
    daily_zones = []
    h4_zones = []

    if "weekly" in data_dict and not data_dict["weekly"].empty:
        weekly_zones = detect_enhanced_levels(
            data_dict["weekly"], "weekly", pip_step, lookback_config.get("weekly")
        )

    if not daily_data.empty:
        daily_zones = detect_enhanced_levels(
            daily_data, "daily", pip_step, lookback_config.get("daily")
        )

    if "4h" in data_dict and not data_dict["4h"].empty:
        h4_zones = detect_enhanced_levels(
            data_dict["4h"], "4h", pip_step, lookback_config.get("4h")
        )

    # Combine all zones
    all_zones = weekly_zones + daily_zones + h4_zones

    # Detect MTF confluence (boost score if level exists on multiple TFs)
    for zone in all_zones:
        mtf_count = 0
        for other_zone in all_zones:
            if other_zone is zone:
                continue
            if other_zone.source_timeframe != zone.source_timeframe:
                # Check if levels are close
                if abs(other_zone.center - zone.center) / zone.center < 0.003:
                    mtf_count += 1

        if mtf_count > 0:
            zone.total_score += SCORING_CONFIG["mtf_confluence"] * mtf_count
            zone.score_breakdown["MTF Confluence"] = SCORING_CONFIG["mtf_confluence"] * mtf_count

    # Separate supports and resistances
    supports = [z for z in all_zones if z.zone_type == ZoneType.SUPPORT]
    resistances = [z for z in all_zones if z.zone_type == ZoneType.RESISTANCE]

    # Sort by distance from price
    supports.sort(key=lambda z: current_price - z.center)
    resistances.sort(key=lambda z: z.center - current_price)

    # Find key levels
    nearest_support = supports[0] if supports else None
    nearest_resistance = resistances[0] if resistances else None

    # Sort by score for strongest
    supports_by_score = sorted(supports, key=lambda z: z.total_score, reverse=True)
    resistances_by_score = sorted(resistances, key=lambda z: z.total_score, reverse=True)

    strongest_support = supports_by_score[0] if supports_by_score else None
    strongest_resistance = resistances_by_score[0] if resistances_by_score else None

    # Check for compression alerts
    compression_into_res = any(
        z.approach_pattern == ApproachPattern.COMPRESSION
        for z in resistances[:3] if z
    )
    compression_into_sup = any(
        z.approach_pattern == ApproachPattern.COMPRESSION
        for z in supports[:3] if z
    )

    return SRAnalysisResult(
        supports=supports[:10],  # Top 10 levels
        resistances=resistances[:10],
        current_price=current_price,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        strongest_support=strongest_support,
        strongest_resistance=strongest_resistance,
        compression_into_resistance=compression_into_res,
        compression_into_support=compression_into_sup,
        weekly_levels=weekly_zones,
        daily_levels=daily_zones,
        h4_levels=h4_zones,
    )


def get_zone_trading_notes(zone: EnhancedZone, current_price: float) -> List[str]:
    """
    Generate trading notes/warnings for a zone.
    """
    notes = []

    # Distance note
    distance_pct = abs(current_price - zone.center) / zone.center * 100
    if distance_pct < 0.3:
        notes.append(f"Price very close to level ({distance_pct:.2f}%)")

    # Liquidity phase warnings
    if zone.liquidity_phase == LiquidityPhase.EXHAUSTED:
        notes.append(f"HIGH BREAK RISK: {zone.recent_cluster_size} rapid tests in current cluster")
    elif zone.liquidity_phase == LiquidityPhase.DEPLETION:
        notes.append(f"Caution: {zone.recent_cluster_size} tests consuming liquidity, watch for diminishing bounces")
    elif zone.liquidity_phase == LiquidityPhase.DISCOVERY:
        notes.append("Fresh level: Limited test history, potential for strong reaction")
    elif zone.liquidity_phase == LiquidityPhase.ESTABLISHED:
        if zone.has_replenished:
            notes.append(f"Replenished level: {zone.total_clusters} test clusters over time, liquidity restored")

    # Compression warning
    if zone.approach_pattern == ApproachPattern.COMPRESSION:
        if zone.zone_type == ZoneType.RESISTANCE:
            notes.append("COMPRESSION: Higher lows forming - elevated break probability")
        else:
            notes.append("COMPRESSION: Lower highs forming - elevated break probability")

    # Diminishing bounces
    if zone.diminishing_bounces:
        notes.append("WARNING: Bounces getting smaller - break may be imminent")

    # Role flip
    if zone.is_role_flip:
        notes.append("ROLE FLIP: Level has acted as both support and resistance")

    # Round number
    if zone.is_round_number:
        notes.append("Psychological round number - expect order clustering")

    # Timeframe significance
    if zone.source_timeframe == "weekly":
        notes.append("Weekly level - highest significance")

    return notes
