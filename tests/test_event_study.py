"""Event-study harness: power (a planted vol-doubling signal is detected) and
calibration (a random signal reports no effect)."""

import numpy as np
import pandas as pd

from vol_engine.event_study import (
    event_study, event_time_path, time_to_resolution, forward_rv, trailing_rv,
)


def make_planted_vol_jump(n=3000, seed=11, n_events=25):
    """Low-vol series where each planted event is followed by 21 bars of 3x vol."""
    rng = np.random.default_rng(seed)
    base_sigma = 0.004
    sigma = np.full(n, base_sigma)
    event_idx = np.sort(rng.choice(np.arange(300, n - 100), size=n_events, replace=False))
    # enforce separation
    keep = [event_idx[0]]
    for i in event_idx[1:]:
        if i - keep[-1] > 60:
            keep.append(i)
    event_idx = np.array(keep)
    for i in event_idx:
        sigma[i + 1 : i + 22] = base_sigma * 3.0

    r = rng.normal(0, sigma)
    c = 1.2 * np.exp(np.cumsum(r))
    o = np.roll(c, 1); o[0] = 1.2
    span = np.abs(rng.normal(0, sigma)) + sigma * 0.5
    df = pd.DataFrame(
        {"Open": o, "High": np.maximum(o, c) * (1 + span),
         "Low": np.minimum(o, c) * (1 - span), "Close": c},
        index=pd.bdate_range("2012-01-02", periods=n),
    )
    signal = pd.Series(False, index=df.index)
    signal.iloc[event_idx] = True
    return df, signal


def test_planted_signal_detected():
    df, signal = make_planted_vol_jump()
    report = event_study(df, signal, "planted", horizons=(5, 21), n_boot=500)
    assert report.n_events_used >= 10
    h21 = next(h for h in report.horizons if h.horizon == 21)
    assert h21.ratio_vs_baseline > 1.5, f"ratio {h21.ratio_vs_baseline}"
    assert h21.expansion_prob > 0.8
    assert h21.p_value < 0.05


def test_random_signal_calibrated():
    df, _ = make_planted_vol_jump(seed=13)
    rng = np.random.default_rng(99)
    idx = rng.choice(np.arange(300, len(df) - 100), size=40, replace=False)
    signal = pd.Series(False, index=df.index)
    signal.iloc[np.sort(idx)] = True
    report = event_study(df, signal, "random", horizons=(21,), n_boot=500)
    h21 = report.horizons[0]
    assert 0.75 < h21.ratio_vs_baseline < 1.35
    assert h21.p_value > 0.05


def test_forward_rv_is_forward():
    """Forward RV at t must reflect bars AFTER t: check on a hand-built step change."""
    n = 400
    sigma = np.full(n, 0.003)
    sigma[200:] = 0.012
    rng = np.random.default_rng(5)
    r = rng.normal(0, sigma)
    c = np.exp(np.cumsum(r))
    o = np.roll(c, 1); o[0] = 1.0
    df = pd.DataFrame(
        {"Open": o, "High": np.maximum(o, c) * 1.001,
         "Low": np.minimum(o, c) * 0.999, "Close": c},
        index=pd.bdate_range("2020-01-01", periods=n),
    )
    fwd = forward_rv(df, 21, "close_to_close")
    trail = trailing_rv(df, 21, "close_to_close")
    # Just before the step, forward RV sees the jump; trailing RV does not.
    assert fwd.iloc[198] > trail.iloc[198] * 1.8


def test_event_time_path_and_resolution():
    df, signal = make_planted_vol_jump()
    rv = trailing_rv(df, 5)
    idx = np.flatnonzero(signal.values)
    path = event_time_path(rv, idx, pre=10, post=25)
    # Trailing 5d RV after the event should exceed before
    before = path.loc[-10:-1, "median"].mean()
    after = path.loc[10:25, "median"].mean()
    assert after > before * 1.5

    # time_to_resolution: expansion bars follow events quickly
    from vol_engine.regime import compute_features
    feats = compute_features(df)
    stats = time_to_resolution(idx, feats["expansion_bar"], max_bars=40)
    assert stats["n_resolved"] >= stats["n"] * 0.6
    assert stats["median_bars"] <= 20
