"""Causal S/R level tracking for conditioning break events.

This is the causal analogue of the dashboard's S/R liquidity model (touch counts,
age, depletion): a swing high/low at bar i needs `window` bars each side, so it
becomes KNOWN only at bar i+window. Levels accumulate touches as later swings form
at the same price. Tagging a break event uses only level information known before
the event — no lookahead — so tags are valid event-study conditioners.

The liquidity hypothesis this enables testing: breaks of well-tested (depleted)
levels release more vol than drifts through untouched prices.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from vol_engine.regime import atr as wilder_atr


@dataclass
class LevelRecord:
    price: float               # running mean of merged swing prices
    kind: str                  # "high" or "low" (side of first formation)
    touch_known_bars: List[int] = field(default_factory=list)  # confirmation bars
    _prices: List[float] = field(default_factory=list)

    @property
    def first_known_bar(self) -> int:
        return self.touch_known_bars[0]

    @property
    def touches(self) -> int:
        return len(self.touch_known_bars)

    def touches_before(self, bar: int) -> int:
        """Touches confirmed strictly before `bar` (causally known at `bar`)."""
        return sum(1 for b in self.touch_known_bars if b < bar)

    def merge_touch(self, price: float, known_bar: int) -> None:
        self._prices.append(price)
        self.price = float(np.mean(self._prices))
        self.touch_known_bars.append(known_bar)


def build_level_history(
    df: pd.DataFrame,
    window: int = 5,
    merge_tol_atr: float = 0.5,
    atr_window: int = 20,
) -> List[LevelRecord]:
    """Track swing-based levels causally.

    A swing high (low) at bar i is confirmed at bar i+window. A confirmed swing
    within merge_tol_atr * ATR of an existing same-side level counts as a re-test
    (touch) of that level; otherwise it opens a new level.
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    span = 2 * window + 1

    is_swing_high = (high == high.rolling(span, center=True).max()).values
    is_swing_low = (low == low.rolling(span, center=True).min()).values

    atr_vals = wilder_atr(df, atr_window).values
    n = len(df)

    # (known_bar, price, kind) in confirmation order
    confirmations: List[Tuple[int, float, str]] = []
    for i in range(window, n - window):
        known = i + window
        if is_swing_high[i]:
            confirmations.append((known, float(high.iloc[i]), "high"))
        if is_swing_low[i]:
            confirmations.append((known, float(low.iloc[i]), "low"))
    confirmations.sort(key=lambda t: t[0])

    levels: List[LevelRecord] = []
    for known, price, kind in confirmations:
        a = atr_vals[known] if known < n and not np.isnan(atr_vals[known]) else np.nan
        tol = merge_tol_atr * a if not np.isnan(a) else 0.0
        merged = False
        if tol > 0:
            candidates = [
                lv for lv in levels
                if lv.kind == kind and abs(lv.price - price) <= tol
            ]
            if candidates:
                nearest = min(candidates, key=lambda lv: abs(lv.price - price))
                nearest.merge_touch(price, known)
                merged = True
        if not merged:
            levels.append(LevelRecord(
                price=price, kind=kind,
                touch_known_bars=[known], _prices=[price],
            ))
    return levels


def tag_break_events(
    events_frame: pd.DataFrame,
    df: pd.DataFrame,
    features: pd.DataFrame,
    window: int = 5,
    match_tol_atr: float = 0.75,
    levels: Optional[List[LevelRecord]] = None,
) -> pd.DataFrame:
    """Annotate break events with the S/R history of the broken level.

    Adds columns:
        sr_matched      a tracked level (known before the event) sat within
                        match_tol_atr * ATR of the broken channel level
        sr_touches      touches of the matched level as of the event bar
        sr_age_bars     bars since the level first became known
        sr_phase        "none" | "fresh" (1 touch) | "tested" (2-3) | "depleted" (4+)

    Only break_attempt / break_confirmed / break_failed rows get non-default tags.
    Among matching levels, the most-touched one is used.
    """
    out = events_frame.copy()
    out["sr_matched"] = False
    out["sr_touches"] = 0
    out["sr_age_bars"] = -1
    out["sr_phase"] = "none"

    if out.empty:
        return out

    lvls = levels if levels is not None else build_level_history(df, window=window)
    atr_vals = features["atr_slow"].values

    break_kinds = {"break_attempt", "break_confirmed", "break_failed"}
    for row in out.itertuples():
        if row.kind not in break_kinds or np.isnan(row.level):
            continue
        t = row.bar_index
        a = atr_vals[t] if t < len(atr_vals) and not np.isnan(atr_vals[t]) else np.nan
        if np.isnan(a):
            continue
        tol = match_tol_atr * a

        best_level, best_touches = None, 0
        for lv in lvls:
            if abs(lv.price - row.level) > tol:
                continue
            touches_now = lv.touches_before(t)
            if touches_now > best_touches:
                best_level, best_touches = lv, touches_now

        if best_level is None:
            continue
        out.loc[row.Index, "sr_matched"] = True
        out.loc[row.Index, "sr_touches"] = best_touches
        out.loc[row.Index, "sr_age_bars"] = t - best_level.first_known_bar
        out.loc[row.Index, "sr_phase"] = (
            "fresh" if best_touches == 1 else "tested" if best_touches <= 3 else "depleted"
        )

    return out


def break_event_masks(
    tagged_events: pd.DataFrame,
    index: pd.Index,
    kind: str = "break_confirmed",
    min_touches_tested: int = 2,
) -> Tuple[pd.Series, pd.Series]:
    """Boolean masks over `index` splitting breaks by S/R context.

    Returns (at_tested_level, elsewhere): breaks of levels with at least
    min_touches_tested causally-known touches vs all other breaks of the kind.
    """
    at_tested = pd.Series(False, index=index)
    elsewhere = pd.Series(False, index=index)
    sub = tagged_events[tagged_events["kind"] == kind]
    for row in sub.itertuples():
        if row.sr_matched and row.sr_touches >= min_touches_tested:
            at_tested.iloc[row.bar_index] = True
        else:
            elsewhere.iloc[row.bar_index] = True
    return at_tested, elsewhere
