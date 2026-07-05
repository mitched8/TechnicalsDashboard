"""Causal S/R level tracker: touch counting, causality, event tagging."""

import numpy as np
import pandas as pd

from vol_engine.levels import (
    build_level_history, tag_break_events, break_event_masks,
)
from vol_engine.regime import classify_states
from vol_engine.synthetic import generate_ohlc


def make_double_top(n=260, seed=4):
    """Flat noisy series with two clear swing highs at the same price (a tested
    level), then a break above it."""
    rng = np.random.default_rng(seed)
    c = 1.10 + rng.normal(0, 0.0006, n).cumsum() * 0.1
    h = c + np.abs(rng.normal(0, 0.0008, n))
    l = c - np.abs(rng.normal(0, 0.0008, n))
    # two spikes to the same level, well-separated
    for peak in (80, 140):
        h[peak] = 1.115
        c[peak] = 1.113
    # breakout at 210: close through and beyond
    h[210:220] += 0.02
    c[210:220] += 0.02
    l[210:220] += 0.015
    o = np.roll(c, 1); o[0] = c[0]
    return pd.DataFrame(
        {"Open": o, "High": h, "Low": l, "Close": c},
        index=pd.bdate_range("2021-01-01", periods=n),
    )


def test_double_top_counted_as_two_touches():
    df = make_double_top()
    levels = build_level_history(df, window=5)
    highs = [lv for lv in levels if lv.kind == "high" and abs(lv.price - 1.115) < 0.002]
    assert highs, "double-top level not tracked"
    lv = max(highs, key=lambda x: x.touches)
    assert lv.touches >= 2
    # First confirmation known at peak+window
    assert lv.first_known_bar == 85


def test_level_history_is_causal_prefix_stable():
    df, _ = generate_ohlc(seed=42)
    full = build_level_history(df)
    k = len(df) - 200
    prefix = build_level_history(df.iloc[:k])
    # Every level fully formed before k (all touches known before k) must exist
    # identically in the prefix computation.
    full_early = [
        (round(lv.price, 6), lv.kind, tuple(lv.touch_known_bars))
        for lv in full
        if lv.touch_known_bars[-1] < k - 1
    ]
    prefix_set = {
        (round(lv.price, 6), lv.kind, tuple(lv.touch_known_bars)) for lv in prefix
    }
    missing = [x for x in full_early if x not in prefix_set]
    # Levels whose touch list extends past k will differ (their running-mean price
    # moved); levels complete before k must match exactly.
    assert not missing, f"{len(missing)} levels changed by future data"


def test_tagging_only_uses_prior_knowledge():
    df = make_double_top()
    result = classify_states(df)
    events = result.events_frame()
    tagged = tag_break_events(events, df, result.features)
    breaks = tagged[tagged["kind"].isin(["break_attempt", "break_confirmed"])]
    if breaks.empty:
        # The channel break must at least have been attempted
        raise AssertionError("no break events detected in constructed breakout")
    # The breakout at bar ~210 crosses the double-top: it should match the level
    # with 2 causally-known touches.
    late = breaks[breaks["bar_index"] >= 205]
    assert (late["sr_matched"]).any()
    assert late.loc[late["sr_matched"], "sr_touches"].max() >= 2
    assert set(late.loc[late["sr_matched"], "sr_phase"]) <= {"tested", "depleted", "fresh"}


def test_masks_partition_break_events():
    df, _ = generate_ohlc(seed=42)
    result = classify_states(df)
    events = result.events_frame()
    tagged = tag_break_events(events, df, result.features)
    at_tested, elsewhere = break_event_masks(tagged, df.index, "break_confirmed")
    n_conf = (tagged["kind"] == "break_confirmed").sum()
    assert int(at_tested.sum() + elsewhere.sum()) == int(n_conf)
    assert not (at_tested & elsewhere).any()
