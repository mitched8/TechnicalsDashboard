"""Data fetching module with caching for FX Trading Dashboard."""

import pandas as pd
import yfinance as yf
import streamlit as st
from typing import Dict
from pathlib import Path
import os


# Map symbols to CSV filenames
CSV_FILES = {
    "EURUSD=X": "eurusd_daily.csv",
    "GBPUSD=X": "gbpusd_daily.csv",
    "USDJPY=X": "usdjpy_daily.csv",
    "AUDUSD=X": "eurusd_daily.csv",  # Fallback to EURUSD for demo
    "USDCNH=X": "eurusd_daily.csv",  # Fallback to EURUSD for demo
}


def get_data_dir() -> Path:
    """Get the data directory path."""
    # Try relative to this file first
    module_dir = Path(__file__).parent.parent
    data_dir = module_dir / "data"
    if data_dir.exists():
        return data_dir
    # Fallback to current working directory
    return Path.cwd() / "data"


def load_csv_data(symbol: str) -> pd.DataFrame:
    """
    Load data from CSV file as fallback.

    Args:
        symbol: Currency pair symbol

    Returns:
        DataFrame with OHLCV data
    """
    data_dir = get_data_dir()
    csv_file = CSV_FILES.get(symbol, "eurusd_daily.csv")
    csv_path = data_dir / csv_file

    if not csv_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
        return df
    except Exception as e:
        st.warning(f"Could not load CSV: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_currency_data(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance with CSV fallback.

    Args:
        symbol: Currency pair symbol (e.g., "EURUSD=X")
        period: Data period ("1y", "6mo", "60d")
        interval: Candle interval ("1wk", "1d", "1h")

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex
    """
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)

        # Flatten MultiIndex columns if present (yfinance quirk)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.dropna()

        # If yfinance returned data, use it
        if not data.empty and len(data) >= 20:
            return data

    except Exception as e:
        pass  # Fall through to CSV fallback

    # Fallback to CSV data
    csv_data = load_csv_data(symbol)
    if not csv_data.empty:
        # Resample if needed for different intervals
        if interval == "1wk":
            return csv_data.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        return csv_data

    return pd.DataFrame()


def fetch_multi_timeframe_data(symbol: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for all required timeframes (weekly, daily, 4h).

    Args:
        symbol: Currency pair symbol

    Returns:
        Dict with keys: 'weekly', 'daily', '4h'
    """
    daily = fetch_currency_data(symbol, period="6mo", interval="1d")

    # Try to get weekly from API, fallback to resampling daily
    weekly = fetch_currency_data(symbol, period="1y", interval="1wk")
    if weekly.empty and not daily.empty:
        weekly = daily.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

    # Try to get hourly, fallback to simulating 4h from daily
    hourly = fetch_currency_data(symbol, period="60d", interval="1h")
    four_hour = resample_to_4h(hourly) if not hourly.empty else daily.copy()

    return {
        "weekly": weekly,
        "daily": daily,
        "4h": four_hour,
    }


def resample_to_4h(hourly_data: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-hour data to 4-hour candles.

    Args:
        hourly_data: DataFrame with 1-hour OHLCV data

    Returns:
        DataFrame with 4-hour OHLCV data
    """
    if hourly_data.empty:
        return hourly_data

    return hourly_data.resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()


def validate_data(data: pd.DataFrame, min_bars: int = 50) -> bool:
    """
    Validate that data has sufficient bars for analysis.

    Args:
        data: OHLCV DataFrame
        min_bars: Minimum required bars

    Returns:
        True if data is valid, False otherwise
    """
    if data is None or data.empty:
        return False
    if len(data) < min_bars:
        return False
    return True
