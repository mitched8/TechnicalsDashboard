"""Vol cones: the distribution of realized vol by measurement window.

The cone answers "is current vol high or low *for this horizon*?" and is the natural
chart to overlay implied vol on once real IV data is available.
"""

import numpy as np
import pandas as pd

from vol_engine.rv_estimators import realized_vol, DEFAULT_ESTIMATOR


def vol_cone(
    df: pd.DataFrame,
    windows=(5, 10, 21, 42, 63, 126),
    percentiles=(5, 25, 50, 75, 95),
    estimator: str = DEFAULT_ESTIMATOR,
    min_obs: int = 60,
) -> pd.DataFrame:
    """Build a vol cone table.

    Returns a DataFrame indexed by window with columns p5, p25, ..., current, n_obs.
    Windows without at least `min_obs` historical observations are dropped.
    """
    rows = {}
    for w in windows:
        rv = realized_vol(df, w, estimator).dropna()
        if len(rv) < min_obs:
            continue
        row = {f"p{p}": float(np.percentile(rv.values, p)) for p in percentiles}
        row["current"] = float(rv.iloc[-1])
        row["n_obs"] = int(len(rv))
        rows[w] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("window")


def rolling_rv_percentile(
    rv: pd.Series,
    lookback: int = 252,
    min_periods: int = 120,
) -> pd.Series:
    """Causal percentile rank of current RV vs its own trailing history.

    NaN (not a fabricated 50) when there is insufficient history.
    """

    def pct_rank(x: np.ndarray) -> float:
        return float((x[:-1] <= x[-1]).mean() * 100.0)

    return rv.rolling(lookback, min_periods=min_periods).apply(pct_rank, raw=True)
