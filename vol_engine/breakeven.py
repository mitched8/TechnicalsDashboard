"""Straddle-breakeven scoreboard: did owning gamma pay?

Compares the daily move implied by IV (the delta-hedged straddle's daily breakeven,
|move| ~ S * IV/sqrt(252)) against realized daily moves. With no real IV available,
a clearly-labeled proxy (trailing RV + a premium) can be used to exercise the
machinery; conclusions only become tradeable once real IV is plugged in.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from vol_engine.rv_estimators import log_returns, ANNUALIZATION
from vol_engine.event_study import trailing_rv


@dataclass
class BreakevenResult:
    frame: pd.DataFrame        # daily_move_pct, breakeven_pct, excess, rolling stats
    iv_source: str             # "real" or description of the proxy
    hit_rate_21d: Optional[float]      # latest rolling share of days beating breakeven
    cum_excess_21d: Optional[float]    # latest rolling sum of (|move| - breakeven), pct pts


def breakeven_scoreboard(
    df: pd.DataFrame,
    iv: Optional[pd.Series] = None,
    proxy_premium_pts: float = 1.0,
    proxy_window: int = 21,
    roll: int = 21,
) -> BreakevenResult:
    """Build the gamma scoreboard.

    Args:
        iv: annualized IV in % aligned (or alignable) to df.index. If None, a proxy
            of trailing RV + proxy_premium_pts is used and labeled as such.
        roll: rolling window (bars) for hit rate and cumulative excess.
    """
    out = pd.DataFrame(index=df.index)
    out["daily_move_pct"] = log_returns(df["Close"]).abs() * 100.0

    if iv is not None:
        iv_aligned = iv.reindex(df.index).ffill()
        iv_source = "real"
    else:
        iv_aligned = trailing_rv(df, proxy_window) + proxy_premium_pts
        iv_source = (
            f"PROXY: trailing {proxy_window}d RV + {proxy_premium_pts} vol pts "
            "(not real IV — for machinery testing only)"
        )

    out["iv_pct"] = iv_aligned
    # Daily breakeven for a delta-hedged straddle, in % of spot. IV is quoted for
    # the day AHEAD, so shift(1): today's move is judged against yesterday's IV.
    out["breakeven_pct"] = out["iv_pct"].shift(1) / np.sqrt(ANNUALIZATION)
    out["excess_pct"] = out["daily_move_pct"] - out["breakeven_pct"]
    out["beat_breakeven"] = out["excess_pct"] > 0

    valid = out["excess_pct"].notna()
    out["hit_rate_roll"] = (
        out["beat_breakeven"].where(valid).rolling(roll, min_periods=roll).mean()
    )
    out["cum_excess_roll"] = out["excess_pct"].rolling(roll, min_periods=roll).sum()

    hit = out["hit_rate_roll"].dropna()
    cum = out["cum_excess_roll"].dropna()
    return BreakevenResult(
        frame=out,
        iv_source=iv_source,
        hit_rate_21d=float(hit.iloc[-1]) if len(hit) else None,
        cum_excess_21d=float(cum.iloc[-1]) if len(cum) else None,
    )
