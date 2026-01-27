"""Support and Resistance zone detection for FX Trading Dashboard."""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN

from config import SR_CONFIG, CONFLUENCE_WEIGHTS


class ZoneType(Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"


@dataclass
class PriceZone:
    """Represents a support or resistance zone."""
    zone_type: ZoneType
    center: float  # Central price level
    upper_bound: float  # Zone upper edge
    lower_bound: float  # Zone lower edge
    strength: float  # Confluence score (higher = stronger)
    touches: int  # Number of price touches
    sources: List[str] = field(default_factory=list)  # Detection sources

    @property
    def width(self) -> float:
        return self.upper_bound - self.lower_bound


@dataclass
class SRResult:
    """Container for support/resistance analysis."""
    supports: List[PriceZone]
    resistances: List[PriceZone]
    current_price: float
    next_support: Optional[PriceZone]
    next_resistance: Optional[PriceZone]


def detect_swing_levels(
    data: pd.DataFrame,
    window: int = 5,
    lookback: int = 100
) -> List[Tuple[float, str]]:
    """
    Detect swing highs and lows using local extrema.

    Uses scipy.argrelextrema to find local maxima/minima.
    A swing high requires `window` lower highs on each side.
    A swing low requires `window` higher lows on each side.

    Args:
        data: OHLCV DataFrame
        window: Number of bars on each side to confirm swing
        lookback: How many recent bars to analyze

    Returns:
        List of (price_level, "swing_high" | "swing_low")
    """
    df = data.tail(lookback).copy()

    if len(df) < window * 2 + 1:
        return []

    high = df['High'].values
    low = df['Low'].values

    # Find local maxima (swing highs)
    swing_high_indices = argrelextrema(high, np.greater_equal, order=window)[0]
    swing_highs = [(high[i], "swing_high") for i in swing_high_indices]

    # Find local minima (swing lows)
    swing_low_indices = argrelextrema(low, np.less_equal, order=window)[0]
    swing_lows = [(low[i], "swing_low") for i in swing_low_indices]

    return swing_highs + swing_lows


def detect_volume_weighted_levels(
    data: pd.DataFrame,
    num_bins: int = 50,
    top_n: int = 5,
    lookback: int = 100
) -> List[float]:
    """
    Detect price levels with highest volume accumulation.

    Creates price bins across trading range and sums volume at each bin.
    This is a simplified Volume Profile approach.

    Args:
        data: OHLCV DataFrame
        num_bins: Number of price bins
        top_n: Number of top levels to return
        lookback: How many recent bars to analyze

    Returns:
        List of price levels with high volume
    """
    df = data.tail(lookback).copy()

    if len(df) < 10 or 'Volume' not in df.columns:
        return []

    # Skip if volume data is all zeros (common in FX)
    if df['Volume'].sum() == 0:
        return []

    # Calculate typical price for each bar
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3

    # Create price bins
    price_min = df['Low'].min()
    price_max = df['High'].max()

    if price_min == price_max:
        return []

    bins = np.linspace(price_min, price_max, num_bins + 1)

    # Assign each bar to a bin and sum volume
    df['price_bin'] = pd.cut(df['typical_price'], bins=bins, labels=bins[:-1])
    volume_profile = df.groupby('price_bin', observed=True)['Volume'].sum()

    if volume_profile.empty:
        return []

    # Get top N levels by volume
    top_levels = volume_profile.nlargest(top_n).index.astype(float).tolist()

    return top_levels


def detect_bollinger_touches(
    data: pd.DataFrame,
    threshold: float = 0.0005,
    lookback: int = 100
) -> List[Tuple[float, str]]:
    """
    Detect levels where price repeatedly touches Bollinger Bands.

    Args:
        data: DataFrame with bb_upper and bb_lower columns
        threshold: How close to band counts as a "touch" (as fraction)
        lookback: How many recent bars to analyze

    Returns:
        List of (price_level, "bb_upper" | "bb_lower")
    """
    if 'bb_upper' not in data.columns or 'bb_lower' not in data.columns:
        return []

    df = data.tail(lookback).copy()
    touches = []

    for idx, row in df.iterrows():
        close = row['Close']
        upper = row.get('bb_upper')
        lower = row.get('bb_lower')

        if pd.isna(upper) or pd.isna(lower):
            continue

        if upper != 0 and abs(close - upper) / upper < threshold:
            touches.append((float(upper), "bb_upper"))
        if lower != 0 and abs(close - lower) / lower < threshold:
            touches.append((float(lower), "bb_lower"))

    return touches


def detect_round_numbers(
    current_price: float,
    pip_step: float = 0.01,
    num_levels: int = 5
) -> List[float]:
    """
    Generate psychological round number levels.

    For EURUSD: 1.0800, 1.0900, 1.1000, etc.
    For USDJPY: 150.00, 151.00, 152.00, etc.

    Args:
        current_price: Current market price
        pip_step: Step size for round numbers (0.01 for most, 1.0 for JPY)
        num_levels: Number of levels above and below

    Returns:
        List of round number price levels
    """
    # Find the nearest round number
    base = np.floor(current_price / pip_step) * pip_step

    levels = []
    for i in range(-num_levels, num_levels + 1):
        level = base + (i * pip_step)
        levels.append(level)

    return levels


def cluster_and_merge_levels(
    levels: List[Tuple[float, str]],
    threshold_pct: float = 0.002
) -> List[PriceZone]:
    """
    Cluster nearby price levels into zones using DBSCAN.

    Args:
        levels: List of (price, source) tuples
        threshold_pct: Maximum distance (as % of price) to cluster

    Returns:
        List of PriceZone objects (zone_type not yet set)
    """
    if not levels:
        return []

    prices = np.array([l[0] for l in levels]).reshape(-1, 1)
    sources = [l[1] for l in levels]

    # DBSCAN clustering
    avg_price = np.mean(prices)
    eps = avg_price * threshold_pct

    clustering = DBSCAN(eps=eps, min_samples=1).fit(prices)
    labels = clustering.labels_

    zones = []
    for label in set(labels):
        if label == -1:  # Noise points
            continue

        mask = labels == label
        cluster_prices = prices[mask].flatten()
        cluster_sources = [sources[i] for i, m in enumerate(mask) if m]

        center = float(np.mean(cluster_prices))
        zone = PriceZone(
            zone_type=ZoneType.SUPPORT,  # Will be updated later
            center=center,
            upper_bound=float(np.max(cluster_prices)),
            lower_bound=float(np.min(cluster_prices)),
            strength=0.0,  # Will be calculated
            touches=len(cluster_prices),
            sources=list(set(cluster_sources))
        )
        zones.append(zone)

    return zones


def calculate_zone_strength(zone: PriceZone, weights: dict = None) -> float:
    """
    Calculate confluence-based strength score for a zone.

    Args:
        zone: PriceZone object
        weights: Dict of source weights

    Returns:
        Strength score
    """
    if weights is None:
        weights = CONFLUENCE_WEIGHTS

    score = 0.0

    if "swing_high" in zone.sources or "swing_low" in zone.sources:
        score += weights.get("swing_level", 2.0)
    if "volume" in zone.sources:
        score += weights.get("volume_level", 1.5)
    if "bb_upper" in zone.sources or "bb_lower" in zone.sources:
        score += weights.get("bollinger_touch", 1.0)
    if "round" in zone.sources:
        score += weights.get("round_number", 1.0)

    # Bonus for multiple touches
    score += min(zone.touches * 0.2, 2.0)

    return score


def analyze_support_resistance(
    data: pd.DataFrame,
    pip_step: float = 0.01,
    config: dict = None
) -> SRResult:
    """
    Main function to detect all support and resistance zones.

    Combines:
    1. Swing high/low detection
    2. Volume-weighted levels
    3. Bollinger Band touches
    4. Round number levels
    5. Clusters and merges nearby levels
    6. Classifies as support or resistance relative to current price

    Args:
        data: DataFrame with OHLCV and indicators
        pip_step: Round number step (0.01 for most pairs, 1.0 for JPY)
        config: Optional config dict

    Returns:
        SRResult with supports, resistances, and next levels
    """
    if config is None:
        config = SR_CONFIG

    if data.empty:
        return SRResult([], [], 0.0, None, None)

    current_price = float(data['Close'].iloc[-1])
    all_levels = []

    # 1. Swing levels
    swing_window = config.get("swing_window", 5)
    swings = detect_swing_levels(data, window=swing_window)
    all_levels.extend(swings)

    # 2. Volume levels
    volume_lookback = config.get("volume_lookback", 100)
    volume_levels = detect_volume_weighted_levels(data, lookback=volume_lookback)
    all_levels.extend([(v, "volume") for v in volume_levels])

    # 3. Bollinger touches (requires indicators calculated)
    bb_threshold = config.get("bollinger_touch_threshold", 0.0005)
    bb_touches = detect_bollinger_touches(data, threshold=bb_threshold)
    all_levels.extend(bb_touches)

    # 4. Round numbers
    round_levels = detect_round_numbers(current_price, pip_step)
    all_levels.extend([(r, "round") for r in round_levels])

    # 5. Cluster and merge
    merge_threshold = config.get("zone_merge_threshold", 0.002)
    zones = cluster_and_merge_levels(all_levels, merge_threshold)

    # 6. Classify as support or resistance and calculate strength
    supports = []
    resistances = []

    for zone in zones:
        zone.strength = calculate_zone_strength(zone)

        if zone.center < current_price:
            zone.zone_type = ZoneType.SUPPORT
            supports.append(zone)
        else:
            zone.zone_type = ZoneType.RESISTANCE
            resistances.append(zone)

    # Sort by distance from current price (nearest first)
    supports.sort(key=lambda z: current_price - z.center)
    resistances.sort(key=lambda z: z.center - current_price)

    # Limit to max levels
    max_levels = config.get("max_levels", 3)
    supports = supports[:max_levels]
    resistances = resistances[:max_levels]

    return SRResult(
        supports=supports,
        resistances=resistances,
        current_price=current_price,
        next_support=supports[0] if supports else None,
        next_resistance=resistances[0] if resistances else None,
    )
