"""Vol-regime features and state machine.

States are vol states, not trend states:

    RANGE -> COMPRESSION -> BREAK_ATTEMPT -> TREND (confirmed) or back to RANGE (failed)

Causality contract: every feature and state at bar t is computed from data at bars
<= t only. Donchian levels use the *prior* N bars; percentile ranks compare the current
value to trailing history only. Appending new data never changes past states (tested).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

from vol_engine.rv_estimators import rv_panel, rv_term_ratios, DEFAULT_ESTIMATOR


class VolState(Enum):
    RANGE = "range"
    COMPRESSION = "compression"
    BREAK_ATTEMPT = "break_attempt"
    TREND = "trend"


class EventKind(Enum):
    COMPRESSION_START = "compression_start"
    COMPRESSION_END = "compression_end"
    BREAK_ATTEMPT = "break_attempt"
    BREAK_CONFIRMED = "break_confirmed"
    BREAK_FAILED = "break_failed"
    TREND_START = "trend_start"
    TREND_END = "trend_end"


@dataclass
class RegimeEvent:
    date: pd.Timestamp
    kind: EventKind
    direction: str = ""        # "up" / "down" / ""
    level: float = float("nan")  # broken channel level, where applicable
    bar_index: int = -1


@dataclass
class RegimeConfig:
    """All tunable thresholds. Defaults are starting points — the event-study harness
    is the tool for tuning them per pair against measured outcomes."""
    donchian_n: int = 20
    width_pctile_lookback: int = 252
    width_pctile_min_periods: int = 120

    # Compression
    compression_width_pctile_max: float = 25.0
    compression_score_min: float = 0.6

    # Break acceptance / failure
    confirm_window: int = 5        # bars allowed to confirm after an attempt
    confirm_closes: int = 2        # closes beyond the level required to confirm
    confirm_buffer_atr: float = 0.25  # a confirming close must clear the level by
                                      # this many ATRs — acceptance, not a poke
    # failure = close back through the broken level during the confirm window

    # Trend maintenance
    er_window: int = 10
    er_trend_exit: float = 0.25
    er_exit_patience: int = 5      # consecutive weak-ER bars before trend ends

    # Feature windows
    atr_fast: int = 5
    atr_slow: int = 20
    expansion_atr_mult: float = 2.0
    estimator: str = DEFAULT_ESTIMATOR


# ---------------------------------------------------------------------------
# Features (all causal)
# ---------------------------------------------------------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c_prev = df["High"], df["Low"], df["Close"].shift(1)
    return pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Wilder-smoothed ATR (pure pandas — no `ta` dependency)."""
    return true_range(df).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def efficiency_ratio(close: pd.Series, window: int = 10) -> pd.Series:
    """Kaufman efficiency ratio: |net move| / sum(|bar moves|) over `window` bars.
    ~1 = clean trend, ~0 = churn. Near-lag-free trend/chop discriminator."""
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return net / path.replace(0.0, np.nan)


def _causal_pct_rank(s: pd.Series, lookback: int, min_periods: int) -> pd.Series:
    def rank(x: np.ndarray) -> float:
        return float((x[:-1] <= x[-1]).mean() * 100.0)

    return s.rolling(lookback, min_periods=min_periods).apply(rank, raw=True)


def compute_features(df: pd.DataFrame, config: Optional[RegimeConfig] = None) -> pd.DataFrame:
    """Feature frame consumed by the state machine (and useful standalone).

    Columns: don_high, don_low, don_width_pct, width_pctile, atr_fast, atr_slow,
    atr_ratio, er, nr7, inside_day, expansion_bar, rv_5, rv_21, rv_63,
    rv_5_over_21, rv_21_over_63, compression_score.
    """
    cfg = config or RegimeConfig()
    out = pd.DataFrame(index=df.index)
    close = df["Close"].astype(float)

    # Donchian channel from the PRIOR n bars (shift(1) keeps today's bar out,
    # so "close breaks the channel" is well-defined and causal).
    out["don_high"] = df["High"].rolling(cfg.donchian_n).max().shift(1)
    out["don_low"] = df["Low"].rolling(cfg.donchian_n).min().shift(1)
    out["don_width_pct"] = (out["don_high"] - out["don_low"]) / close * 100.0
    out["width_pctile"] = _causal_pct_rank(
        out["don_width_pct"], cfg.width_pctile_lookback, cfg.width_pctile_min_periods
    )

    out["atr_fast"] = atr(df, cfg.atr_fast)
    out["atr_slow"] = atr(df, cfg.atr_slow)
    out["atr_ratio"] = out["atr_fast"] / out["atr_slow"]

    out["er"] = efficiency_ratio(close, cfg.er_window)

    tr = true_range(df)
    out["nr7"] = (tr <= tr.rolling(7).min()) & tr.notna()
    prev_h, prev_l = df["High"].shift(1), df["Low"].shift(1)
    out["inside_day"] = (df["High"] < prev_h) & (df["Low"] > prev_l)
    out["expansion_bar"] = tr > cfg.expansion_atr_mult * out["atr_slow"].shift(1)

    panel = rv_panel(df, windows=(5, 21, 63), estimator=cfg.estimator)
    out = out.join(panel)
    out = out.join(rv_term_ratios(panel))

    # Composite compression score in [0, 1]: narrow channel + fast ATR below slow
    # + realized vol in backwardation + quiet-bar clustering.
    c_width = (1.0 - out["width_pctile"] / 100.0).clip(0.0, 1.0)
    c_atr = ((1.0 - out["atr_ratio"]) / 0.4).clip(0.0, 1.0)
    c_rv = ((1.0 - out["rv_5_over_21"]) / 0.4).clip(0.0, 1.0)
    quiet = (out["nr7"].astype(float) + out["inside_day"].astype(float)).rolling(5).sum()
    c_quiet = (quiet / 4.0).clip(0.0, 1.0)
    out["compression_score"] = pd.concat([c_width, c_atr, c_rv, c_quiet], axis=1).mean(axis=1)
    # Not meaningful until the width percentile exists
    out.loc[out["width_pctile"].isna(), "compression_score"] = np.nan

    return out


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    states: pd.Series                 # VolState.value per bar
    events: List[RegimeEvent] = field(default_factory=list)
    features: Optional[pd.DataFrame] = None

    def events_frame(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(columns=["date", "kind", "direction", "level", "bar_index"])
        return pd.DataFrame(
            [
                {
                    "date": e.date,
                    "kind": e.kind.value,
                    "direction": e.direction,
                    "level": e.level,
                    "bar_index": e.bar_index,
                }
                for e in self.events
            ]
        )

    def event_mask(self, kind: EventKind) -> pd.Series:
        mask = pd.Series(False, index=self.states.index)
        for e in self.events:
            if e.kind == kind:
                mask.iloc[e.bar_index] = True
        return mask


def classify_states(
    df: pd.DataFrame,
    config: Optional[RegimeConfig] = None,
    features: Optional[pd.DataFrame] = None,
) -> RegimeResult:
    """Run the causal state machine over the whole series.

    Single forward pass; the state at bar t depends only on bars <= t.
    """
    cfg = config or RegimeConfig()
    feats = features if features is not None else compute_features(df, cfg)
    close = df["Close"].astype(float).values

    don_high = feats["don_high"].values
    don_low = feats["don_low"].values
    width_pctile = feats["width_pctile"].values
    score = feats["compression_score"].values
    er = feats["er"].values
    atr_slow = feats["atr_slow"].values

    n = len(df)
    states = np.array([VolState.RANGE.value] * n, dtype=object)
    events: List[RegimeEvent] = []

    state = VolState.RANGE
    in_compression = False

    # Break-attempt bookkeeping
    attempt_dir = ""
    attempt_level = np.nan
    attempt_start = -1
    closes_beyond = 0

    # Trend bookkeeping
    trend_dir = ""
    weak_er_run = 0

    def emit(i: int, kind: EventKind, direction: str = "", level: float = float("nan")):
        events.append(RegimeEvent(date=df.index[i], kind=kind, direction=direction,
                                  level=level, bar_index=i))

    for i in range(n):
        c = close[i]
        hi, lo = don_high[i], don_low[i]
        channel_ready = not (np.isnan(hi) or np.isnan(lo))

        if state in (VolState.RANGE, VolState.COMPRESSION):
            # Channel break starts an attempt (checked before compression upkeep).
            if channel_ready and (c > hi or c < lo):
                if in_compression:
                    emit(i, EventKind.COMPRESSION_END)
                    in_compression = False
                attempt_dir = "up" if c > hi else "down"
                attempt_level = hi if attempt_dir == "up" else lo
                attempt_start = i
                buf0 = cfg.confirm_buffer_atr * (atr_slow[i] if not np.isnan(atr_slow[i]) else 0.0)
                cleared = c > attempt_level + buf0 if attempt_dir == "up" else c < attempt_level - buf0
                closes_beyond = 1 if cleared else 0
                emit(i, EventKind.BREAK_ATTEMPT, attempt_dir, attempt_level)
                state = VolState.BREAK_ATTEMPT
            else:
                compressed = (
                    channel_ready
                    and not np.isnan(width_pctile[i])
                    and not np.isnan(score[i])
                    and width_pctile[i] <= cfg.compression_width_pctile_max
                    and score[i] >= cfg.compression_score_min
                )
                if compressed and not in_compression:
                    in_compression = True
                    emit(i, EventKind.COMPRESSION_START)
                elif not compressed and in_compression:
                    in_compression = False
                    emit(i, EventKind.COMPRESSION_END)
                state = VolState.COMPRESSION if in_compression else VolState.RANGE

        elif state == VolState.BREAK_ATTEMPT:
            beyond = c > attempt_level if attempt_dir == "up" else c < attempt_level
            buf = cfg.confirm_buffer_atr * (atr_slow[i] if not np.isnan(atr_slow[i]) else 0.0)
            accepted = (
                c > attempt_level + buf if attempt_dir == "up" else c < attempt_level - buf
            )
            if accepted:
                closes_beyond += 1
            failed = not beyond  # close back through the broken level
            window_up = i - attempt_start >= cfg.confirm_window

            if closes_beyond >= cfg.confirm_closes and accepted:
                emit(i, EventKind.BREAK_CONFIRMED, attempt_dir, attempt_level)
                emit(i, EventKind.TREND_START, attempt_dir)
                trend_dir = attempt_dir
                weak_er_run = 0
                state = VolState.TREND
            elif failed or window_up:
                emit(i, EventKind.BREAK_FAILED, attempt_dir, attempt_level)
                state = VolState.RANGE

        elif state == VolState.TREND:
            e = er[i]
            if not np.isnan(e) and e < cfg.er_trend_exit:
                weak_er_run += 1
            else:
                weak_er_run = 0
            if weak_er_run >= cfg.er_exit_patience:
                emit(i, EventKind.TREND_END, trend_dir)
                trend_dir = ""
                state = VolState.RANGE
            else:
                # A fresh break in the opposite direction restarts the cycle.
                if channel_ready and trend_dir == "up" and c < lo:
                    emit(i, EventKind.TREND_END, trend_dir)
                    attempt_dir, attempt_level, attempt_start = "down", lo, i
                    closes_beyond = 1
                    emit(i, EventKind.BREAK_ATTEMPT, attempt_dir, attempt_level)
                    state = VolState.BREAK_ATTEMPT
                elif channel_ready and trend_dir == "down" and c > hi:
                    emit(i, EventKind.TREND_END, trend_dir)
                    attempt_dir, attempt_level, attempt_start = "up", hi, i
                    closes_beyond = 1
                    emit(i, EventKind.BREAK_ATTEMPT, attempt_dir, attempt_level)
                    state = VolState.BREAK_ATTEMPT

        states[i] = state.value if not (state in (VolState.RANGE, VolState.COMPRESSION)) else (
            VolState.COMPRESSION.value if in_compression else VolState.RANGE.value
        )

    return RegimeResult(
        states=pd.Series(states, index=df.index, name="vol_state"),
        events=events,
        features=feats,
    )
