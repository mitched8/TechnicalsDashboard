"""OHLC data loading and validation — the engine's single input seam.

Input contract everywhere in vol_engine:
    DataFrame, DatetimeIndex ascending, columns Open, High, Low, Close (floats).

CSV schema (what to export from the work PC's data source):
    Date,Open,High,Low,Close        # ISO dates; extra columns are ignored
"""

from pathlib import Path
from typing import Union

import pandas as pd

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close")


class DataContractError(ValueError):
    """Raised when input data violates the OHLC contract."""


def validate_ohlc(df: pd.DataFrame, min_bars: int = 100) -> pd.DataFrame:
    """Validate and normalize an OHLC frame. Raises DataContractError on violation."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataContractError(f"Missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataContractError("Index must be a DatetimeIndex")

    out = df.loc[:, list(REQUIRED_COLUMNS)].astype(float).sort_index()
    out = out.dropna()
    if len(out) < min_bars:
        raise DataContractError(
            f"Only {len(out)} valid bars; need at least {min_bars}. "
            "Short histories produce unreliable percentiles and event studies."
        )

    bad = (
        (out["High"] < out[["Open", "Close", "Low"]].max(axis=1))
        | (out["Low"] > out[["Open", "Close", "High"]].min(axis=1))
        | (out[list(REQUIRED_COLUMNS)] <= 0).any(axis=1)
    )
    n_bad = int(bad.sum())
    if n_bad:
        if n_bad > 0.01 * len(out):
            raise DataContractError(f"{n_bad} bars violate OHLC consistency (H>=O,C,L etc.)")
        out = out[~bad]

    out.attrs.setdefault("synthetic", False)
    return out


def load_ohlc_csv(path: Union[str, Path], min_bars: int = 100) -> pd.DataFrame:
    """Load an OHLC CSV (Date column or first column as dates)."""
    path = Path(path)
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    out = validate_ohlc(df, min_bars=min_bars)
    out.attrs["provenance"] = f"CSV: {path.name} ({out.index.min().date()} to {out.index.max().date()})"
    return out
