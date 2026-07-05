"""Regime detection: planted structure is found, and detection is causal
(prefix-stable — appending future data never changes past states)."""

import numpy as np
import pandas as pd

from vol_engine.regime import (
    RegimeConfig, VolState, EventKind, compute_features, classify_states,
)
from vol_engine.synthetic import generate_ohlc, default_script, Segment


def test_planted_compression_detected():
    """Segment-level detection: the Donchian width can only reflect a compression
    segment once narrow bars dominate the window, so entry lag is inherent. What
    matters is that (a) most planted segments are detected before they resolve and
    (b) detected compression concentrates in truly-compressed periods (precision)."""
    df, truth = generate_ohlc(seed=42)
    result = classify_states(df)
    feats = result.features
    detected = result.states.values

    seg_id = (truth != truth.shift()).cumsum()
    positions = np.arange(len(truth))
    n_seg = 0
    n_hit = 0
    for _, seg in truth.groupby(seg_id):
        if seg.iloc[0] != "compression":
            continue
        idx = positions[(seg_id == seg_id.loc[seg.index[0]]).values]
        if feats["width_pctile"].iloc[idx].isna().all():
            continue  # percentile warmup — features not yet available
        n_seg += 1
        ext = np.arange(idx[0], min(idx[-1] + 11, len(truth)))
        if any(detected[i] == VolState.COMPRESSION.value for i in ext):
            n_hit += 1
    assert n_seg >= 5
    assert n_hit / n_seg >= 0.7, f"only {n_hit}/{n_seg} compression segments detected"

    # Bar-level precision against planted labels
    ready = feats["width_pctile"].notna()
    comp_truth = truth == "compression"
    comp_detected = result.states == VolState.COMPRESSION.value
    detected_bars = (comp_detected & ready).sum()
    assert detected_bars > 0
    precision = (comp_truth & comp_detected & ready).sum() / detected_bars
    assert precision > 0.5, f"compression precision {precision:.2f}"


def test_planted_trends_produce_confirmed_breaks():
    """Primary requirement: every planted trend is announced by a confirmed break
    (recall). Some range noise also confirms before failing — real ranges do that
    too; precision only gets a sanity floor here, because measuring which breaks
    were worth trading is the event-study harness's job, not the detector's."""
    df, truth = generate_ohlc(seed=42)
    result = classify_states(df)
    confirmed = [e for e in result.events if e.kind == EventKind.BREAK_CONFIRMED]
    assert len(confirmed) >= 5

    truth_vals = truth.values
    positions = np.arange(len(truth))
    seg_id = (truth != truth.shift()).cumsum()

    # Recall: a confirmed break within [segment_start - 5, segment_end] of each trend
    caught = 0
    trend_segs = [seg for _, seg in truth.groupby(seg_id) if seg.iloc[0] == "trend"]
    for seg in trend_segs:
        lo = positions[truth.index.get_loc(seg.index[0])] - 5
        hi = positions[truth.index.get_loc(seg.index[-1])]
        if any(lo <= e.bar_index <= hi for e in confirmed):
            caught += 1
    assert caught >= len(trend_segs) - 1, f"trends caught: {caught}/{len(trend_segs)}"

    # Precision sanity floor
    near_trend = sum(
        1 for e in confirmed
        if (truth_vals[e.bar_index : e.bar_index + 15] == "trend").any()
    )
    assert near_trend / len(confirmed) > 0.35

    # And the machine must not be gullible: failures should be well represented.
    failed = [e for e in result.events if e.kind == EventKind.BREAK_FAILED]
    assert len(failed) >= len(confirmed) * 0.5


def test_no_lookahead_prefix_stability():
    df, _ = generate_ohlc(seed=7)
    full = classify_states(df).states
    k = len(df) - 150
    prefix = classify_states(df.iloc[:k]).states
    # States on the prefix must equal the same rows of the full computation
    pd.testing.assert_series_equal(full.iloc[:k], prefix, check_names=False)


def test_features_causal_prefix_stability():
    df, _ = generate_ohlc(seed=9)
    cfg = RegimeConfig()
    full = compute_features(df, cfg)
    k = len(df) - 100
    prefix = compute_features(df.iloc[:k], cfg)
    for col in ["don_high", "don_low", "width_pctile", "er", "compression_score"]:
        pd.testing.assert_series_equal(
            full[col].iloc[:k], prefix[col], check_names=False, atol=1e-12,
        )


def test_pure_noise_yields_few_confirmed_breaks():
    rng = np.random.default_rng(3)
    n = 2000
    r = rng.normal(0, 0.005, n)
    c = 1.1 * np.exp(np.cumsum(r))
    o = np.roll(c, 1); o[0] = 1.1
    span = np.abs(rng.normal(0, 0.003, n)) + 0.001
    df = pd.DataFrame(
        {"Open": o, "High": np.maximum(o, c) * (1 + span), "Low": np.minimum(o, c) * (1 - span), "Close": c},
        index=pd.bdate_range("2015-01-01", periods=n),
    )
    result = classify_states(df)
    confirmed = [e for e in result.events if e.kind == EventKind.BREAK_CONFIRMED]
    failed = [e for e in result.events if e.kind == EventKind.BREAK_FAILED]
    # Random walks do break channels sometimes, but failures should be well
    # represented — the machine must not confirm everything.
    assert len(failed) > 0
    assert len(confirmed) < len(failed) * 3
