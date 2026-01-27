"""Forex utility functions for pip calculations and formatting."""

from typing import Tuple


def get_pip_info(symbol: str) -> Tuple[int, float]:
    """
    Get pip decimal places and round number step for a currency pair.

    Args:
        symbol: Currency pair symbol (e.g., "EURUSD=X")

    Returns:
        Tuple of (pip_decimal, round_step)
    """
    jpy_pairs = ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY', 'NZDJPY']
    base_symbol = symbol.replace('=X', '').upper()

    if any(jpy in base_symbol for jpy in jpy_pairs):
        return (2, 1.0)
    else:
        return (4, 0.01)


def calculate_pips(price1: float, price2: float, pip_decimal: int) -> float:
    """
    Calculate pip distance between two prices.

    Args:
        price1: First price
        price2: Second price
        pip_decimal: Number of decimal places for pips (4 or 2)

    Returns:
        Pip distance (positive or negative)
    """
    multiplier = 10 ** pip_decimal
    return (price2 - price1) * multiplier


def format_price(price: float, pip_decimal: int) -> str:
    """
    Format price with appropriate decimal places.

    Args:
        price: Price to format
        pip_decimal: Number of decimal places for pips

    Returns:
        Formatted price string
    """
    decimals = pip_decimal + 1  # One more decimal than pip precision
    return f"{price:.{decimals}f}"


def format_pip_distance(price1: float, price2: float, pip_decimal: int) -> str:
    """
    Format pip distance with direction sign.

    Args:
        price1: Starting price (current price)
        price2: Target price
        pip_decimal: Number of decimal places for pips

    Returns:
        Formatted string like "+92 pips" or "-45 pips"
    """
    pips = calculate_pips(price1, price2, pip_decimal)
    sign = "+" if pips >= 0 else ""
    return f"{sign}{pips:.0f} pips"


def format_percentage_move(price1: float, price2: float) -> str:
    """
    Format percentage move between two prices.

    Args:
        price1: Starting price
        price2: Target price

    Returns:
        Formatted string like "+0.85%" or "-1.23%"
    """
    if price1 == 0:
        return "N/A"
    pct = ((price2 - price1) / price1) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def format_level_display(
    current_price: float,
    target_price: float,
    pip_decimal: int
) -> str:
    """
    Format a complete level display with price, pips, and percentage.

    Args:
        current_price: Current market price
        target_price: Target price level
        pip_decimal: Number of decimal places for pips

    Returns:
        Formatted string like "1.0892 (+50 pips, +0.46%)"
    """
    price_str = format_price(target_price, pip_decimal)
    pip_str = format_pip_distance(current_price, target_price, pip_decimal)
    pct_str = format_percentage_move(current_price, target_price)
    return f"{price_str} ({pip_str}, {pct_str})"
