"""Configuration constants for FX Trading Dashboard."""

from dataclasses import dataclass
from typing import Dict

@dataclass
class CurrencyPairConfig:
    """Configuration for a currency pair."""
    symbol: str
    pip_decimal: int  # 4 for most pairs, 2 for JPY pairs
    round_number_step: float  # 0.01 for most, 1.0 for JPY

CURRENCY_PAIRS: Dict[str, CurrencyPairConfig] = {
    "EURUSD=X": CurrencyPairConfig("EURUSD=X", 4, 0.01),
    "GBPUSD=X": CurrencyPairConfig("GBPUSD=X", 4, 0.01),
    "USDJPY=X": CurrencyPairConfig("USDJPY=X", 2, 1.0),
    "AUDUSD=X": CurrencyPairConfig("AUDUSD=X", 4, 0.01),
    "USDCNH=X": CurrencyPairConfig("USDCNH=X", 4, 0.01),
}

# Timeframe configurations
TIMEFRAMES = {
    "weekly": {"period": "1y", "interval": "1wk"},
    "daily": {"period": "6mo", "interval": "1d"},
    "4h": {"period": "60d", "interval": "1h"},  # Fetch 1h, resample to 4h
}

# Support/Resistance Detection Parameters
SR_CONFIG = {
    "swing_window": 5,              # Bars on each side to confirm swing
    "volume_lookback": 100,         # Bars for volume profile analysis
    "zone_merge_threshold": 0.002,  # 0.2% to merge nearby levels
    "min_touches": 2,               # Minimum touches to confirm level
    "max_levels": 3,                # Max levels to display per direction
    "bollinger_touch_threshold": 0.0005,  # How close to band = "touch"
}

# Regime Detection Parameters
REGIME_CONFIG = {
    "adx_trending_threshold": 25,   # ADX above this = trending
    "atr_high_percentile": 75,      # ATR percentile for high volatility
    "atr_low_percentile": 25,       # ATR percentile for low volatility
    "atr_lookback": 100,            # Bars for ATR percentile calculation
}

# Confluence Weights for S/R Strength
CONFLUENCE_WEIGHTS = {
    "swing_level": 2.0,
    "volume_level": 1.5,
    "bollinger_touch": 1.0,
    "round_number": 1.0,
    "mtf_alignment": 2.0,
    "regime_favorable": 1.5,
}

# Trade Setup Parameters
TRADE_CONFIG = {
    "atr_buffer_multiplier": 0.5,   # Half ATR for entry zone buffer
    "min_rr_ratio": 1.5,            # Minimum R:R to show setup
}
