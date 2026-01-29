"""Implied volatility data handling for FX options analysis."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path


# Available tenors in the IV data
IV_TENORS = ['1W', '1M', '3M', '6M', '1Y']

# Tenor to days mapping (approximate)
TENOR_DAYS = {
    '1W': 7,
    '1M': 30,
    '3M': 90,
    '6M': 180,
    '1Y': 365,
}


@dataclass
class ImpliedVolData:
    """Container for implied volatility data."""
    symbol: str
    date: str
    iv_1w: float
    iv_1m: float
    iv_3m: float
    iv_6m: float
    iv_1y: float

    def get_iv_by_tenor(self, tenor: str) -> float:
        """Get IV for a specific tenor."""
        tenor_map = {
            '1W': self.iv_1w,
            '1M': self.iv_1m,
            '3M': self.iv_3m,
            '6M': self.iv_6m,
            '1Y': self.iv_1y,
        }
        return tenor_map.get(tenor, self.iv_1m)

    def get_term_structure(self) -> Dict[str, float]:
        """Get full term structure as dict."""
        return {
            '1W': self.iv_1w,
            '1M': self.iv_1m,
            '3M': self.iv_3m,
            '6M': self.iv_6m,
            '1Y': self.iv_1y,
        }

    def is_inverted(self) -> bool:
        """Check if term structure is inverted (short > long)."""
        return self.iv_1m > self.iv_1y

    def get_slope(self) -> float:
        """Get term structure slope (1Y - 1M)."""
        return self.iv_1y - self.iv_1m


@dataclass
class IVAnalysis:
    """Analysis of implied vol vs realized vol."""
    current_iv: float  # Selected tenor IV
    current_rv: float  # Realized vol (1M)
    iv_rv_spread: float  # IV - RV (vol risk premium)
    iv_percentile: float  # IV percentile (0-100)
    term_structure: Dict[str, float]
    is_inverted: bool
    term_slope: float
    recommendation: str


def load_iv_data(
    symbol: str,
    data_path: str = "data/fx_implied_vol.csv"
) -> Optional[pd.DataFrame]:
    """
    Load implied volatility data for a currency pair.

    Args:
        symbol: Currency pair symbol (e.g., "EURUSD=X")
        data_path: Path to IV CSV file

    Returns:
        DataFrame with IV data or None if not found
    """
    # Convert symbol format (EURUSD=X -> EURUSD)
    clean_symbol = symbol.replace("=X", "")

    try:
        # Try relative path first
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        # Try absolute path
        try:
            abs_path = Path(__file__).parent.parent / data_path
            df = pd.read_csv(abs_path)
        except FileNotFoundError:
            return None

    # Filter for symbol
    df_symbol = df[df['symbol'] == clean_symbol].copy()

    if df_symbol.empty:
        return None

    # Convert date column
    df_symbol['date'] = pd.to_datetime(df_symbol['date'])
    df_symbol = df_symbol.set_index('date').sort_index()

    return df_symbol


def get_latest_iv(
    symbol: str,
    data_path: str = "data/fx_implied_vol.csv"
) -> Optional[ImpliedVolData]:
    """
    Get the latest IV data for a currency pair.

    Args:
        symbol: Currency pair symbol
        data_path: Path to IV CSV file

    Returns:
        ImpliedVolData object or None
    """
    df = load_iv_data(symbol, data_path)

    if df is None or df.empty:
        return None

    latest = df.iloc[-1]

    return ImpliedVolData(
        symbol=symbol.replace("=X", ""),
        date=str(latest.name.date()),
        iv_1w=float(latest.get('iv_1W', 0)),
        iv_1m=float(latest.get('iv_1M', 0)),
        iv_3m=float(latest.get('iv_3M', 0)),
        iv_6m=float(latest.get('iv_6M', 0)),
        iv_1y=float(latest.get('iv_1Y', 0)),
    )


def get_iv_time_series(
    symbol: str,
    tenor: str = '1M',
    data_path: str = "data/fx_implied_vol.csv"
) -> Optional[pd.Series]:
    """
    Get IV time series for a specific tenor.

    Args:
        symbol: Currency pair symbol
        tenor: Vol tenor ('1W', '1M', '3M', '6M', '1Y')
        data_path: Path to IV CSV file

    Returns:
        Series of IV values indexed by date
    """
    df = load_iv_data(symbol, data_path)

    if df is None or df.empty:
        return None

    col_name = f'iv_{tenor}'
    if col_name not in df.columns:
        return None

    return df[col_name]


def merge_iv_with_price_data(
    price_df: pd.DataFrame,
    symbol: str,
    data_path: str = "data/fx_implied_vol.csv"
) -> pd.DataFrame:
    """
    Merge IV data with price DataFrame.

    Args:
        price_df: DataFrame with price data (indexed by date)
        symbol: Currency pair symbol
        data_path: Path to IV CSV file

    Returns:
        DataFrame with IV columns added
    """
    iv_df = load_iv_data(symbol, data_path)

    if iv_df is None or iv_df.empty:
        # Return original with NaN IV columns
        result = price_df.copy()
        for tenor in IV_TENORS:
            result[f'iv_{tenor}'] = np.nan
        return result

    # Merge on date index
    # Forward fill IV data to handle weekends/holidays
    result = price_df.copy()

    for tenor in IV_TENORS:
        col_name = f'iv_{tenor}'
        if col_name in iv_df.columns:
            # Reindex IV to match price dates and forward fill
            iv_series = iv_df[col_name].reindex(result.index, method='ffill')
            result[col_name] = iv_series

    return result


def calculate_iv_rv_spread(
    iv: float,
    rv: float
) -> float:
    """
    Calculate IV-RV spread (vol risk premium).

    Positive spread = IV > RV (vol is "expensive")
    Negative spread = IV < RV (vol is "cheap")

    Args:
        iv: Implied volatility
        rv: Realized volatility

    Returns:
        Spread in vol points
    """
    return iv - rv


def analyze_iv_vs_rv(
    iv_data: ImpliedVolData,
    rv_1m: float,
    selected_tenor: str = '1M',
    iv_history: Optional[pd.Series] = None
) -> IVAnalysis:
    """
    Analyze implied vol vs realized vol relationship.

    Args:
        iv_data: Current IV data
        rv_1m: 1-month realized volatility
        selected_tenor: Tenor to analyze
        iv_history: Historical IV series for percentile calculation

    Returns:
        IVAnalysis object
    """
    current_iv = iv_data.get_iv_by_tenor(selected_tenor)
    spread = calculate_iv_rv_spread(current_iv, rv_1m)
    term_structure = iv_data.get_term_structure()

    # Calculate IV percentile if history available
    if iv_history is not None and len(iv_history) > 0:
        iv_percentile = (iv_history < current_iv).sum() / len(iv_history) * 100
    else:
        iv_percentile = 50.0

    # Generate recommendation
    if spread > 2.0:  # IV much higher than RV
        if iv_percentile > 75:
            recommendation = "SELL VOL: IV expensive vs RV and at high percentile - consider selling premium"
        else:
            recommendation = "SELL VOL BIAS: IV trading above RV - selling strategies may be favorable"
    elif spread < -2.0:  # IV much lower than RV
        if iv_percentile < 25:
            recommendation = "BUY VOL: IV cheap vs RV and at low percentile - consider buying options"
        else:
            recommendation = "BUY VOL BIAS: IV trading below RV - buying strategies may be favorable"
    else:
        if iv_data.is_inverted():
            recommendation = "CAUTION: Inverted term structure suggests near-term event risk"
        else:
            recommendation = "NEUTRAL: IV fairly priced vs RV - no strong directional vol view"

    return IVAnalysis(
        current_iv=current_iv,
        current_rv=rv_1m,
        iv_rv_spread=spread,
        iv_percentile=iv_percentile,
        term_structure=term_structure,
        is_inverted=iv_data.is_inverted(),
        term_slope=iv_data.get_slope(),
        recommendation=recommendation
    )


def get_all_tenors_df(
    symbol: str,
    data_path: str = "data/fx_implied_vol.csv"
) -> Optional[pd.DataFrame]:
    """
    Get DataFrame with all tenor IVs for charting.

    Args:
        symbol: Currency pair symbol
        data_path: Path to IV CSV file

    Returns:
        DataFrame with columns for each tenor
    """
    df = load_iv_data(symbol, data_path)

    if df is None:
        return None

    # Rename columns for clarity
    rename_map = {
        'iv_1W': 'IV 1W',
        'iv_1M': 'IV 1M',
        'iv_3M': 'IV 3M',
        'iv_6M': 'IV 6M',
        'iv_1Y': 'IV 1Y',
    }

    result = df.rename(columns=rename_map)

    # Keep only IV columns
    iv_cols = [col for col in result.columns if col.startswith('IV')]
    result = result[iv_cols]

    return result
