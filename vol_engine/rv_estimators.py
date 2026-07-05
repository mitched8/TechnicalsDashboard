"""Realized volatility estimators from daily OHLC.

All rolling estimators return annualized vol in percentage points (e.g. 8.5 = 8.5%),
matching the convention used elsewhere in this project. Per-bar variance proxies
(suffix `_var`) return *daily* variance of log returns, un-annualized — these are the
building blocks for HAR and for forward-RV measurement in the event-study harness.

Estimator notes (window n, annualization factor A=252):
- close_to_close: std of log returns. Unbiased but least efficient.
- parkinson:      (1/(4 ln2)) * mean[(ln H/L)^2]. Assumes no drift, no gaps.
- garman_klass:   mean[0.5 (ln H/L)^2 - (2 ln2 - 1)(ln C/O)^2]. More efficient; the
                  project default for FX daily bars (overnight gaps are small).
- rogers_satchell: drift-independent range estimator.
- yang_zhang:     combines overnight, open-close and RS terms; handles drift and gaps.
"""

import numpy as np
import pandas as pd

ANNUALIZATION = 252
DEFAULT_ESTIMATOR = "garman_klass"

_LN2 = np.log(2.0)


def _validate_ohlc(df: pd.DataFrame, cols=("Open", "High", "Low", "Close")) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"OHLC frame missing columns: {missing}")


def log_returns(close: pd.Series) -> pd.Series:
    """Log returns of a close series."""
    close = close.astype(float)
    return np.log(close / close.shift(1))


# ---------------------------------------------------------------------------
# Per-bar daily variance proxies (un-annualized)
# ---------------------------------------------------------------------------

def cc_var(df: pd.DataFrame) -> pd.Series:
    """Squared log return — the close-to-close daily variance proxy."""
    return log_returns(df["Close"]) ** 2


def parkinson_var(df: pd.DataFrame) -> pd.Series:
    _validate_ohlc(df, ("High", "Low"))
    hl = np.log(df["High"].astype(float) / df["Low"].astype(float))
    return hl**2 / (4.0 * _LN2)


def garman_klass_var(df: pd.DataFrame) -> pd.Series:
    _validate_ohlc(df)
    hl = np.log(df["High"].astype(float) / df["Low"].astype(float))
    co = np.log(df["Close"].astype(float) / df["Open"].astype(float))
    return 0.5 * hl**2 - (2.0 * _LN2 - 1.0) * co**2


def rogers_satchell_var(df: pd.DataFrame) -> pd.Series:
    _validate_ohlc(df)
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


PER_BAR_VAR = {
    "close_to_close": cc_var,
    "parkinson": parkinson_var,
    "garman_klass": garman_klass_var,
    "rogers_satchell": rogers_satchell_var,
}


# ---------------------------------------------------------------------------
# Rolling annualized vol estimators
# ---------------------------------------------------------------------------

def _annualize_var(mean_daily_var: pd.Series) -> pd.Series:
    return np.sqrt(mean_daily_var.clip(lower=0.0) * ANNUALIZATION) * 100.0


def close_to_close_rv(df: pd.DataFrame, window: int) -> pd.Series:
    r = log_returns(df["Close"])
    return r.rolling(window).std() * np.sqrt(ANNUALIZATION) * 100.0


def parkinson_rv(df: pd.DataFrame, window: int) -> pd.Series:
    return _annualize_var(parkinson_var(df).rolling(window).mean())


def garman_klass_rv(df: pd.DataFrame, window: int) -> pd.Series:
    return _annualize_var(garman_klass_var(df).rolling(window).mean())


def rogers_satchell_rv(df: pd.DataFrame, window: int) -> pd.Series:
    return _annualize_var(rogers_satchell_var(df).rolling(window).mean())


def yang_zhang_rv(df: pd.DataFrame, window: int) -> pd.Series:
    """Yang-Zhang: sigma^2 = sigma_o^2 + k*sigma_c^2 + (1-k)*sigma_rs^2."""
    _validate_ohlc(df)
    o = df["Open"].astype(float)
    c = df["Close"].astype(float)

    overnight = np.log(o / c.shift(1))
    open_close = np.log(c / o)

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    var_o = overnight.rolling(n).var()
    var_c = open_close.rolling(n).var()
    var_rs = rogers_satchell_var(df).rolling(n).mean()

    return _annualize_var(var_o + k * var_c + (1.0 - k) * var_rs)


ESTIMATORS = {
    "close_to_close": close_to_close_rv,
    "parkinson": parkinson_rv,
    "garman_klass": garman_klass_rv,
    "rogers_satchell": rogers_satchell_rv,
    "yang_zhang": yang_zhang_rv,
}


def realized_vol(
    df: pd.DataFrame,
    window: int = 21,
    estimator: str = DEFAULT_ESTIMATOR,
) -> pd.Series:
    """Rolling annualized realized vol (%) using the named estimator."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown estimator '{estimator}'. Choose from {sorted(ESTIMATORS)}")
    return ESTIMATORS[estimator](df, window)


def rv_panel(
    df: pd.DataFrame,
    windows=(5, 10, 21, 42, 63),
    estimator: str = DEFAULT_ESTIMATOR,
) -> pd.DataFrame:
    """RV at several windows as columns rv_5, rv_10, ... aligned to df.index."""
    out = pd.DataFrame(index=df.index)
    for w in windows:
        out[f"rv_{w}"] = realized_vol(df, w, estimator)
    return out


def rv_term_ratios(panel: pd.DataFrame) -> pd.DataFrame:
    """Short/long RV ratios from an rv_panel frame. Ratio < 1 = realized vol in
    backwardation (recent vol below longer-run vol), a compression signature."""
    out = pd.DataFrame(index=panel.index)
    if {"rv_5", "rv_21"}.issubset(panel.columns):
        out["rv_5_over_21"] = panel["rv_5"] / panel["rv_21"]
    if {"rv_21", "rv_63"}.issubset(panel.columns):
        out["rv_21_over_63"] = panel["rv_21"] / panel["rv_63"]
    return out
