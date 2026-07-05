"""Event-study harness: turns regime definitions into measured claims.

For any boolean signal series (e.g. "compression started", "break confirmed"), this
module computes the conditional distribution of FORWARD realized vol at several
horizons versus the unconditional baseline, with Monte-Carlo p-values, plus
event-time average paths and time-to-resolution statistics.

Forward-looking data is used here deliberately: this is measurement of outcomes, not
signal construction. Signals passed in must themselves be causal.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from vol_engine.rv_estimators import PER_BAR_VAR, ANNUALIZATION, DEFAULT_ESTIMATOR

MIN_EVENTS_WARN = 15


def forward_rv(
    df: pd.DataFrame,
    horizon: int,
    estimator: str = DEFAULT_ESTIMATOR,
) -> pd.Series:
    """Annualized RV (%) realized over the NEXT `horizon` bars (excluding today).

    Intentionally forward-looking (measurement, not signal).
    """
    var_fn = PER_BAR_VAR.get(estimator, PER_BAR_VAR["garman_klass"])
    daily_var = var_fn(df)
    # Reversed rolling mean gives, at t, the mean over [t, t+h-1]; shifting by -1
    # makes it the mean over the NEXT h bars [t+1, t+h].
    fwd_mean_var = daily_var[::-1].rolling(horizon).mean()[::-1].shift(-1)
    return np.sqrt(fwd_mean_var.clip(lower=0.0) * ANNUALIZATION) * 100.0


def trailing_rv(
    df: pd.DataFrame,
    horizon: int,
    estimator: str = DEFAULT_ESTIMATOR,
) -> pd.Series:
    """Annualized RV (%) over the PAST `horizon` bars (inclusive of today)."""
    var_fn = PER_BAR_VAR.get(estimator, PER_BAR_VAR["garman_klass"])
    return np.sqrt(var_fn(df).rolling(horizon).mean().clip(lower=0.0) * ANNUALIZATION) * 100.0


def _decluster(event_idx: np.ndarray, min_separation: int) -> np.ndarray:
    """Keep the first event of any cluster; drop events within min_separation bars."""
    kept: List[int] = []
    last = -10**9
    for i in event_idx:
        if i - last >= min_separation:
            kept.append(i)
            last = i
    return np.array(kept, dtype=int)


@dataclass
class HorizonStats:
    horizon: int
    n_events: int
    event_median: float
    event_q25: float
    event_q75: float
    baseline_median: float
    ratio_vs_baseline: float          # event median / baseline median
    expansion_prob: float             # P(fwd RV > trailing RV) among events
    baseline_expansion_prob: float
    p_value: float                    # MC: P(random median >= event median)


@dataclass
class EventStudyReport:
    signal_name: str
    horizons: List[HorizonStats] = field(default_factory=list)
    n_events_raw: int = 0
    n_events_used: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for h in self.horizons:
            rows.append({
                "horizon": h.horizon,
                "n_events": h.n_events,
                "event_median_rv": round(h.event_median, 2),
                "event_iqr": f"{h.event_q25:.2f}-{h.event_q75:.2f}",
                "baseline_median_rv": round(h.baseline_median, 2),
                "ratio": round(h.ratio_vs_baseline, 3),
                "expansion_prob": round(h.expansion_prob, 3),
                "baseline_expansion_prob": round(h.baseline_expansion_prob, 3),
                "p_value": round(h.p_value, 4),
            })
        return pd.DataFrame(rows)


def event_study(
    df: pd.DataFrame,
    signal: pd.Series,
    signal_name: str = "signal",
    horizons: Sequence[int] = (5, 10, 21, 42),
    estimator: str = DEFAULT_ESTIMATOR,
    min_separation: int = 5,
    n_boot: int = 2000,
    seed: int = 7,
) -> EventStudyReport:
    """Conditional forward-RV distributions for a boolean event signal.

    Args:
        df: OHLC frame.
        signal: boolean Series aligned to df.index; True marks an event bar.
        horizons: forward horizons in bars.
        min_separation: de-cluster events closer than this many bars.
        n_boot: Monte-Carlo draws for the p-value (random same-size date samples).
    """
    signal = signal.reindex(df.index).fillna(False).astype(bool)
    raw_idx = np.flatnonzero(signal.values)
    used_idx = _decluster(raw_idx, min_separation)

    report = EventStudyReport(signal_name=signal_name,
                              n_events_raw=int(len(raw_idx)),
                              n_events_used=int(len(used_idx)))

    if len(used_idx) == 0:
        report.warnings.append("No events found for this signal.")
        return report
    if len(used_idx) < MIN_EVENTS_WARN:
        report.warnings.append(
            f"Only {len(used_idx)} events — distributions are indicative, not reliable. "
            f"Longer history recommended (>= {MIN_EVENTS_WARN} events)."
        )

    rng = np.random.default_rng(seed)

    for h in horizons:
        fwd = forward_rv(df, h, estimator)
        trail = trailing_rv(df, h, estimator)
        valid = fwd.notna().values & trail.notna().values

        ev = used_idx[valid[used_idx]]
        if len(ev) == 0:
            continue

        fwd_vals = fwd.values
        trail_vals = trail.values
        event_fwd = fwd_vals[ev]
        event_exp = (fwd_vals[ev] > trail_vals[ev]).mean()

        pool = np.flatnonzero(valid)
        base_fwd = fwd_vals[pool]
        base_exp = (fwd_vals[pool] > trail_vals[pool]).mean()

        # Monte-Carlo p-value: how often does a random same-size sample of dates
        # produce a median forward RV at least as high as the events did?
        ev_median = float(np.median(event_fwd))
        boot_medians = np.empty(n_boot)
        for b in range(n_boot):
            sample = rng.choice(pool, size=len(ev), replace=False)
            boot_medians[b] = np.median(fwd_vals[sample])
        p_val = float((boot_medians >= ev_median).mean())

        base_median = float(np.median(base_fwd))
        report.horizons.append(HorizonStats(
            horizon=int(h),
            n_events=int(len(ev)),
            event_median=ev_median,
            event_q25=float(np.percentile(event_fwd, 25)),
            event_q75=float(np.percentile(event_fwd, 75)),
            baseline_median=base_median,
            ratio_vs_baseline=ev_median / base_median if base_median > 0 else float("nan"),
            expansion_prob=float(event_exp),
            baseline_expansion_prob=float(base_exp),
            p_value=p_val,
        ))

    return report


def event_time_path(
    series: pd.Series,
    event_indices: Sequence[int],
    pre: int = 20,
    post: int = 20,
) -> pd.DataFrame:
    """Average/median path of `series` around events, in event time.

    Returns a frame indexed by relative bar (-pre..+post) with mean, median, n columns.
    """
    vals = series.values.astype(float)
    n = len(vals)
    rel = np.arange(-pre, post + 1)
    rows = []
    for r in rel:
        samples = []
        for i in event_indices:
            j = i + r
            if 0 <= j < n and not np.isnan(vals[j]):
                samples.append(vals[j])
        if samples:
            arr = np.array(samples)
            rows.append({"rel_bar": r, "mean": arr.mean(), "median": np.median(arr), "n": len(arr)})
        else:
            rows.append({"rel_bar": r, "mean": np.nan, "median": np.nan, "n": 0})
    return pd.DataFrame(rows).set_index("rel_bar")


def time_to_resolution(
    start_indices: Sequence[int],
    resolution_mask: pd.Series,
    max_bars: int = 60,
) -> Dict[str, float]:
    """Bars from each start event until the next True in resolution_mask.

    Used e.g. for "bars from compression start to first expansion bar or break" —
    which maps signal horizon to option tenor. Unresolved events are censored at
    max_bars and reported separately.
    """
    res = resolution_mask.values.astype(bool)
    n = len(res)
    durations: List[int] = []
    censored = 0
    for i in start_indices:
        d = None
        for j in range(i + 1, min(i + 1 + max_bars, n)):
            if res[j]:
                d = j - i
                break
        if d is None:
            censored += 1
        else:
            durations.append(d)
    out: Dict[str, float] = {
        "n": float(len(start_indices)),
        "n_resolved": float(len(durations)),
        "n_censored": float(censored),
    }
    if durations:
        arr = np.array(durations, dtype=float)
        out.update({
            "median_bars": float(np.median(arr)),
            "q25_bars": float(np.percentile(arr, 25)),
            "q75_bars": float(np.percentile(arr, 75)),
        })
    return out
