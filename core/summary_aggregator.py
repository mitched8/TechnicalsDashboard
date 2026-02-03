"""Summary aggregator for multi-pair FX dashboard overview.

Aggregates key statistics across all tracked currency pairs for the Summary page.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st

from config import CURRENCY_PAIRS, StrategySettings, get_default_settings
from core.data_fetcher import fetch_multi_timeframe_data, validate_data
from core.indicators import calculate_all_indicators, IndicatorValues
from core.regime import (
    detect_regime, analyze_mtf_context, analyze_adx,
    MarketRegime, MultiTimeframeContext, ADXAnalysis, ADXRegime, TrendDirection
)
from core.signals import (
    Signal, calculate_confluence_score, summarize_signals,
    generate_signals_with_settings
)
from core.volatility import analyze_volatility, VolatilityAnalysis, VolRegime
from core.support_resistance import analyze_support_resistance, SRResult, PriceZone
from core.trade_setup import generate_trade_setup, TradeSetup, Bias


@dataclass
class PairSummary:
    """Summary data for a single currency pair."""
    symbol: str
    display_name: str  # Without =X suffix
    price: float
    pip_decimal: int
    
    # Price changes
    pct_change_1d: float
    pct_change_1w: float
    pct_change_1m: float
    
    # ADX analysis
    adx: float
    adx_regime: ADXRegime
    adx_slope: str  # "rising", "falling", "flat"
    plus_di: float
    minus_di: float
    
    # Trend direction
    trend_direction: TrendDirection
    trend_strength: float
    
    # MTF alignment
    mtf_alignment: str  # "aligned_bullish", "aligned_bearish", "mixed"
    weekly_trend: str
    daily_trend: str
    four_hour_trend: str
    
    # Volatility
    vol_regime: VolRegime
    rv_1m: float
    rv_percentile: float
    is_squeeze: bool
    vol_trend: str
    
    # Signals
    bullish_count: int
    bearish_count: int
    neutral_count: int
    confluence_score: float
    bias: str  # "LONG", "SHORT", "NEUTRAL"
    
    # Trade setup (if valid)
    has_setup: bool = False
    setup_bias: str = ""
    setup_entry_low: float = 0.0
    setup_entry_high: float = 0.0
    setup_stop: float = 0.0
    setup_target1: float = 0.0
    setup_rr: float = 0.0
    
    # S/R proximity
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    support_distance_pips: float = 0.0
    resistance_distance_pips: float = 0.0


@dataclass
class MarketSummary:
    """Aggregated summary across all pairs."""
    pairs: List[PairSummary]
    timestamp: datetime
    
    # Aggregate metrics
    trending_count: int = 0  # ADX > 25
    bullish_aligned_count: int = 0
    bearish_aligned_count: int = 0
    high_vol_count: int = 0
    squeeze_count: int = 0
    
    # Regime distribution
    range_count: int = 0
    transition_count: int = 0
    trend_count: int = 0
    
    # Signal momentum
    total_bullish_signals: int = 0
    total_bearish_signals: int = 0
    total_neutral_signals: int = 0
    net_signal_bias: int = 0
    
    # Valid setups
    valid_setup_count: int = 0
    
    # Health index (0-10)
    health_index: float = 5.0
    health_components: Dict[str, float] = field(default_factory=dict)
    
    # Alerts
    alerts: List[str] = field(default_factory=list)


def calculate_pct_changes(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate percentage changes over multiple timeframes.
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        Dict with keys '1d', '1w', '1m'
    """
    if df.empty or len(df) < 2:
        return {'1d': 0.0, '1w': 0.0, '1m': 0.0}
    
    current_price = float(df['Close'].iloc[-1])
    
    # 1-day change
    pct_1d = 0.0
    if len(df) >= 2:
        prev_close = float(df['Close'].iloc[-2])
        if prev_close != 0:
            pct_1d = ((current_price - prev_close) / prev_close) * 100
    
    # 1-week change (5 trading days)
    pct_1w = 0.0
    if len(df) >= 6:
        week_ago = float(df['Close'].iloc[-6])
        if week_ago != 0:
            pct_1w = ((current_price - week_ago) / week_ago) * 100
    
    # 1-month change (21 trading days)
    pct_1m = 0.0
    if len(df) >= 22:
        month_ago = float(df['Close'].iloc[-22])
        if month_ago != 0:
            pct_1m = ((current_price - month_ago) / month_ago) * 100
    
    return {'1d': pct_1d, '1w': pct_1w, '1m': pct_1m}


def get_trend_label(regime: MarketRegime) -> str:
    """Get string label for trend direction."""
    if regime.trend == TrendDirection.UP:
        return "Bullish"
    elif regime.trend == TrendDirection.DOWN:
        return "Bearish"
    return "Neutral"


def calculate_pair_summary(
    symbol: str,
    settings: StrategySettings = None
) -> Optional[PairSummary]:
    """
    Calculate summary for a single currency pair.
    
    Args:
        symbol: Currency pair symbol (e.g., "EURUSD=X")
        settings: Strategy settings
        
    Returns:
        PairSummary or None if data fetch fails
    """
    if settings is None:
        settings = get_default_settings()
    
    pair_config = CURRENCY_PAIRS.get(symbol)
    if not pair_config:
        return None
    
    # Fetch data
    mtf_data = fetch_multi_timeframe_data(symbol)
    daily_data = mtf_data.get('daily', pd.DataFrame())
    
    if not validate_data(daily_data, min_bars=30):
        return None
    
    # Calculate indicators for each timeframe
    for tf in ['weekly', 'daily', '4h']:
        if not mtf_data[tf].empty:
            mtf_data[tf], _ = calculate_all_indicators(mtf_data[tf])
    
    daily_data = mtf_data['daily']
    current_price = float(daily_data['Close'].iloc[-1])
    pip_decimal = pair_config.pip_decimal
    
    # Price changes
    pct_changes = calculate_pct_changes(daily_data)
    
    # Regimes
    daily_regime = detect_regime(daily_data)
    mtf_context = analyze_mtf_context(
        mtf_data['weekly'],
        mtf_data['daily'],
        mtf_data['4h']
    )
    adx_analysis = analyze_adx(daily_data)
    
    # Volatility
    try:
        vol_indicators = {
            'adx': adx_analysis.adx,
            'rsi': float(daily_data['rsi'].iloc[-1]) if 'rsi' in daily_data.columns else 50,
        }
        vol_analysis = analyze_volatility(daily_data, vol_indicators)
        vol_regime = vol_analysis.metrics.regime
        rv_1m = vol_analysis.metrics.rv_1m
        rv_percentile = vol_analysis.metrics.rv_percentile
        is_squeeze = vol_analysis.metrics.is_squeeze
        vol_trend = vol_analysis.metrics.vol_trend
    except Exception:
        vol_regime = VolRegime.NORMAL
        rv_1m = 0.0
        rv_percentile = 50.0
        is_squeeze = False
        vol_trend = "flat"
    
    # Signals
    # Get indicator dict from daily data
    latest = daily_data.iloc[-1]
    indicators_dict = {
        'rsi': float(latest.get('rsi', 50)),
        'macd': float(latest.get('macd', 0)),
        'macd_signal': float(latest.get('macd_signal', 0)),
        'macd_histogram': float(latest.get('macd_histogram', 0)),
        'stoch_k': float(latest.get('stoch_k', 50)),
        'stoch_d': float(latest.get('stoch_d', 50)),
        'sma_20': float(latest.get('sma_20', current_price)),
        'sma_50': float(latest.get('sma_50', current_price)),
        'sma_200': float(latest.get('sma_200', current_price)),
        'adx': adx_analysis.adx,
        'plus_di': adx_analysis.plus_di,
        'minus_di': adx_analysis.minus_di,
    }
    
    # Add EMA 200 if available
    if 'ema_200' in latest:
        indicators_dict['ema_200'] = float(latest.get('ema_200', current_price))
    
    signals = generate_signals_with_settings(
        daily_data, current_price, indicators_dict, settings.entry_signals
    )
    signal_summary = summarize_signals(signals)
    
    # Confluence score
    mtf_aligned = mtf_context.alignment in ["aligned_bullish", "aligned_bearish"]
    regime_favorable = adx_analysis.regime != ADXRegime.RANGE_MEAN_REVERSION
    confluence = calculate_confluence_score(
        signals, mtf_aligned=mtf_aligned, regime_favorable=regime_favorable
    )
    
    # Determine bias
    if signal_summary['bullish'] > signal_summary['bearish'] + 2:
        bias = "LONG"
    elif signal_summary['bearish'] > signal_summary['bullish'] + 2:
        bias = "SHORT"
    else:
        bias = "NEUTRAL"
    
    # S/R analysis
    round_step = pair_config.round_number_step
    sr_result = analyze_support_resistance(daily_data, pip_step=round_step)
    
    nearest_support = None
    nearest_resistance = None
    support_distance = 0.0
    resistance_distance = 0.0
    pip_multiplier = 10 ** pip_decimal
    
    if sr_result.next_support:
        nearest_support = sr_result.next_support.center
        support_distance = (current_price - nearest_support) * pip_multiplier
    
    if sr_result.next_resistance:
        nearest_resistance = sr_result.next_resistance.center
        resistance_distance = (nearest_resistance - current_price) * pip_multiplier
    
    # Trade setup
    has_setup = False
    setup_bias = ""
    setup_entry_low = 0.0
    setup_entry_high = 0.0
    setup_stop = 0.0
    setup_target1 = 0.0
    setup_rr = 0.0
    
    atr = float(daily_data['atr'].iloc[-1]) if 'atr' in daily_data.columns else 0.001
    trade_setup = generate_trade_setup(
        sr_result, daily_regime, mtf_context,
        atr=atr, pip_decimal=pip_decimal
    )
    
    if trade_setup.bias != Bias.NEUTRAL:
        has_setup = True
        setup_bias = "LONG" if trade_setup.bias == Bias.LONG else "SHORT"
        setup_entry_low = trade_setup.entry_zone_low
        setup_entry_high = trade_setup.entry_zone_high
        setup_stop = trade_setup.stop_loss
        setup_target1 = trade_setup.target_1
        setup_rr = trade_setup.risk_reward_1
    
    # MTF trend labels
    weekly_trend = get_trend_label(mtf_context.weekly_regime)
    daily_trend_label = get_trend_label(mtf_context.daily_regime)
    four_hour_trend = get_trend_label(mtf_context.four_hour_regime)
    
    return PairSummary(
        symbol=symbol,
        display_name=symbol.replace("=X", ""),
        price=current_price,
        pip_decimal=pip_decimal,
        pct_change_1d=pct_changes['1d'],
        pct_change_1w=pct_changes['1w'],
        pct_change_1m=pct_changes['1m'],
        adx=adx_analysis.adx,
        adx_regime=adx_analysis.regime,
        adx_slope=adx_analysis.slope.value,
        plus_di=adx_analysis.plus_di,
        minus_di=adx_analysis.minus_di,
        trend_direction=daily_regime.trend,
        trend_strength=daily_regime.trend_strength,
        mtf_alignment=mtf_context.alignment,
        weekly_trend=weekly_trend,
        daily_trend=daily_trend_label,
        four_hour_trend=four_hour_trend,
        vol_regime=vol_regime,
        rv_1m=rv_1m,
        rv_percentile=rv_percentile,
        is_squeeze=is_squeeze,
        vol_trend=vol_trend,
        bullish_count=signal_summary['bullish'],
        bearish_count=signal_summary['bearish'],
        neutral_count=signal_summary['neutral'],
        confluence_score=confluence,
        bias=bias,
        has_setup=has_setup,
        setup_bias=setup_bias,
        setup_entry_low=setup_entry_low,
        setup_entry_high=setup_entry_high,
        setup_stop=setup_stop,
        setup_target1=setup_target1,
        setup_rr=setup_rr,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        support_distance_pips=support_distance,
        resistance_distance_pips=resistance_distance,
    )


def calculate_health_index(pairs: List[PairSummary]) -> Tuple[float, Dict[str, float]]:
    """
    Calculate composite market health index (0-10).
    
    Components:
    - Trending strength (ADX avg): 30%
    - Signal agreement: 25%
    - MTF alignment: 25%
    - Volatility stability: 20%
    
    Returns:
        (health_index, component_breakdown)
    """
    if not pairs:
        return 5.0, {}
    
    n = len(pairs)
    
    # Trending strength (higher ADX = better for trend traders)
    avg_adx = sum(p.adx for p in pairs) / n
    trending_strength = min(avg_adx / 40 * 10, 10)  # Scale to 0-10
    
    # Signal agreement (consensus across pairs)
    total_bullish = sum(p.bullish_count for p in pairs)
    total_bearish = sum(p.bearish_count for p in pairs)
    total_signals = total_bullish + total_bearish
    if total_signals > 0:
        agreement = abs(total_bullish - total_bearish) / total_signals
        signal_agreement = agreement * 10
    else:
        signal_agreement = 5.0
    
    # MTF alignment (how many pairs have aligned timeframes)
    aligned_count = sum(1 for p in pairs if p.mtf_alignment in ["aligned_bullish", "aligned_bearish"])
    mtf_alignment = (aligned_count / n) * 10
    
    # Volatility stability (inverse of extreme vol)
    high_vol_count = sum(1 for p in pairs if p.vol_regime == VolRegime.HIGH)
    squeeze_count = sum(1 for p in pairs if p.is_squeeze)
    vol_stability = 10 - ((high_vol_count + squeeze_count) / n * 5)
    vol_stability = max(0, vol_stability)
    
    # Weighted composite
    health_index = (
        0.30 * trending_strength +
        0.25 * signal_agreement +
        0.25 * mtf_alignment +
        0.20 * vol_stability
    )
    
    components = {
        'Trending Strength': trending_strength,
        'Signal Agreement': signal_agreement,
        'MTF Alignment': mtf_alignment,
        'Volatility Stability': vol_stability,
    }
    
    return round(health_index, 1), components


def detect_alerts(pairs: List[PairSummary]) -> List[str]:
    """
    Generate alerts for notable market conditions.
    
    Args:
        pairs: List of PairSummary objects
        
    Returns:
        List of alert strings
    """
    alerts = []
    
    for p in pairs:
        # Squeeze alert
        if p.is_squeeze:
            alerts.append(f"🔔 {p.display_name}: Bollinger Squeeze detected - breakout imminent")
        
        # High volatility alert
        if p.vol_regime == VolRegime.HIGH and p.rv_percentile > 80:
            alerts.append(f"⚠️ {p.display_name}: High volatility ({p.rv_1m:.1f}% RV, {p.rv_percentile:.0f}th percentile)")
        
        # Strong trend alert
        if p.adx > 35 and p.adx_slope == "rising":
            direction = "bullish" if p.plus_di > p.minus_di else "bearish"
            alerts.append(f"📈 {p.display_name}: Strong {direction} trend (ADX: {p.adx:.1f} and rising)")
        
        # Near S/R alert
        if p.support_distance_pips > 0 and p.support_distance_pips < 30:
            alerts.append(f"🎯 {p.display_name}: Approaching support ({p.support_distance_pips:.0f} pips away)")
        if p.resistance_distance_pips > 0 and p.resistance_distance_pips < 30:
            alerts.append(f"🎯 {p.display_name}: Approaching resistance ({p.resistance_distance_pips:.0f} pips away)")
        
        # Valid trade setup alert
        if p.has_setup and p.setup_rr >= 2.0:
            alerts.append(f"✅ {p.display_name}: Valid {p.setup_bias} setup (R:R {p.setup_rr:.1f})")
    
    return alerts[:10]  # Limit to 10 alerts


@st.cache_data(ttl=300, show_spinner=False)
def generate_market_summary(
    symbols: List[str] = None,
    settings: StrategySettings = None
) -> MarketSummary:
    """
    Generate complete market summary for all pairs.
    
    Args:
        symbols: List of currency pair symbols (defaults to all pairs)
        settings: Strategy settings
        
    Returns:
        MarketSummary with all aggregated data
    """
    if symbols is None:
        symbols = list(CURRENCY_PAIRS.keys())
    
    if settings is None:
        settings = get_default_settings()
    
    # Fetch data for all pairs (could parallelize in future)
    pairs: List[PairSummary] = []
    
    for symbol in symbols:
        summary = calculate_pair_summary(symbol, settings)
        if summary:
            pairs.append(summary)
    
    if not pairs:
        return MarketSummary(
            pairs=[],
            timestamp=datetime.now(),
        )
    
    # Aggregate metrics
    trending_count = sum(1 for p in pairs if p.adx > 25)
    bullish_aligned = sum(1 for p in pairs if p.mtf_alignment == "aligned_bullish")
    bearish_aligned = sum(1 for p in pairs if p.mtf_alignment == "aligned_bearish")
    high_vol = sum(1 for p in pairs if p.vol_regime == VolRegime.HIGH)
    squeeze = sum(1 for p in pairs if p.is_squeeze)
    
    # Regime distribution
    range_count = sum(1 for p in pairs if p.adx_regime == ADXRegime.RANGE_MEAN_REVERSION)
    transition_count = sum(1 for p in pairs if p.adx_regime == ADXRegime.TRANSITION_BREAKOUT)
    trend_count = sum(1 for p in pairs if p.adx_regime == ADXRegime.TREND_CONTINUATION)
    
    # Signal momentum
    total_bullish = sum(p.bullish_count for p in pairs)
    total_bearish = sum(p.bearish_count for p in pairs)
    total_neutral = sum(p.neutral_count for p in pairs)
    
    # Valid setups
    valid_setups = sum(1 for p in pairs if p.has_setup and p.setup_rr >= 1.5)
    
    # Health index
    health_index, health_components = calculate_health_index(pairs)
    
    # Alerts
    alerts = detect_alerts(pairs)
    
    return MarketSummary(
        pairs=pairs,
        timestamp=datetime.now(),
        trending_count=trending_count,
        bullish_aligned_count=bullish_aligned,
        bearish_aligned_count=bearish_aligned,
        high_vol_count=high_vol,
        squeeze_count=squeeze,
        range_count=range_count,
        transition_count=transition_count,
        trend_count=trend_count,
        total_bullish_signals=total_bullish,
        total_bearish_signals=total_bearish,
        total_neutral_signals=total_neutral,
        net_signal_bias=total_bullish - total_bearish,
        valid_setup_count=valid_setups,
        health_index=health_index,
        health_components=health_components,
        alerts=alerts,
    )


def pairs_to_dataframe(pairs: List[PairSummary]) -> pd.DataFrame:
    """
    Convert list of PairSummary to DataFrame for display.
    
    Args:
        pairs: List of PairSummary objects
        
    Returns:
        DataFrame suitable for st.dataframe()
    """
    rows = []
    for p in pairs:
        regime_labels = {
            ADXRegime.RANGE_MEAN_REVERSION: "Range",
            ADXRegime.TRANSITION_BREAKOUT: "Transition",
            ADXRegime.TREND_CONTINUATION: "Trend",
        }
        
        mtf_labels = {
            "aligned_bullish": "Aligned Bullish",
            "aligned_bearish": "Aligned Bearish",
            "mixed": "Mixed",
        }
        
        vol_labels = {
            VolRegime.HIGH: "High",
            VolRegime.NORMAL: "Normal",
            VolRegime.LOW: "Low",
            VolRegime.SQUEEZE: "Squeeze",
        }
        
        rows.append({
            'Pair': p.display_name,
            'Price': p.price,
            'Chg 1D': p.pct_change_1d,
            'Chg 1W': p.pct_change_1w,
            'Chg 1M': p.pct_change_1m,
            'ADX': p.adx,
            'Regime': regime_labels.get(p.adx_regime, "Unknown"),
            'MTF Align': mtf_labels.get(p.mtf_alignment, "Mixed"),
            'Vol State': vol_labels.get(p.vol_regime, "Normal"),
            'Bias': p.bias,
            'Confluence': p.confluence_score,
        })
    
    return pd.DataFrame(rows)


def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Apply filters to the pair comparison DataFrame.
    
    Args:
        df: DataFrame from pairs_to_dataframe()
        filters: Dict of filter values
        
    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()
    
    # Currency search - matches base or quote currency
    if filters.get('currency_search'):
        search_term = filters['currency_search'].upper()
        filtered = filtered[filtered['Pair'].str.upper().str.contains(search_term)]
    
    # Specific pair selection
    if filters.get('pairs'):
        filtered = filtered[filtered['Pair'].isin(filters['pairs'])]
    
    if filters.get('regime'):
        filtered = filtered[filtered['Regime'].isin(filters['regime'])]
    
    if filters.get('mtf_align'):
        filtered = filtered[filtered['MTF Align'].isin(filters['mtf_align'])]
    
    if filters.get('vol_state'):
        filtered = filtered[filtered['Vol State'].isin(filters['vol_state'])]
    
    if filters.get('bias'):
        filtered = filtered[filtered['Bias'].isin(filters['bias'])]
    
    if filters.get('min_adx'):
        filtered = filtered[filtered['ADX'] >= filters['min_adx']]
    
    if filters.get('min_confluence'):
        filtered = filtered[filtered['Confluence'] >= filters['min_confluence']]
    
    if filters.get('pct_direction') == 'positive':
        filtered = filtered[filtered['Chg 1D'] > 0]
    elif filters.get('pct_direction') == 'negative':
        filtered = filtered[filtered['Chg 1D'] < 0]
    
    return filtered
