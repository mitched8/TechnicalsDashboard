"""Breakeven scoreboard, breadth, playbook, data IO contract."""

import numpy as np
import pandas as pd
import pytest

from vol_engine.breakeven import breakeven_scoreboard
from vol_engine.breadth import compression_breadth
from vol_engine.playbook import get_playbook, PLAYBOOK, FAILED_BREAK_ENTRY
from vol_engine.data_io import validate_ohlc, load_ohlc_csv, DataContractError
from vol_engine.regime import classify_states, VolState
from vol_engine.synthetic import generate_ohlc


@pytest.fixture(scope="module")
def demo():
    df, truth = generate_ohlc(seed=42)
    return df


def test_breakeven_proxy_labeled(demo):
    result = breakeven_scoreboard(demo)
    assert "PROXY" in result.iv_source
    assert result.hit_rate_21d is not None
    assert 0.0 <= result.hit_rate_21d <= 1.0
    # With IV = RV + premium, moves should beat breakeven less than half the time
    valid = result.frame["beat_breakeven"][result.frame["excess_pct"].notna()]
    assert valid.mean() < 0.5


def test_breakeven_with_real_iv(demo):
    iv = pd.Series(5.0, index=demo.index)  # constant 5% IV: cheap vs realized
    result = breakeven_scoreboard(demo, iv=iv)
    assert result.iv_source == "real"
    valid = result.frame["beat_breakeven"][result.frame["excess_pct"].notna()]
    assert valid.mean() > 0.3  # cheap IV gets beaten often


def test_breadth(demo):
    r1 = classify_states(demo)
    r2 = classify_states(demo.iloc[:-100])
    breadth = compression_breadth({"A": r1, "B": r2})
    assert {"fraction_compressed", "fraction_trending", "n_pairs"} <= set(breadth.columns)
    assert breadth["fraction_compressed"].dropna().between(0, 1).all()


def test_playbook_states_covered():
    for state in [s.value for s in VolState]:
        entry = get_playbook(state)
        assert entry.stance and entry.rationale

    fade = get_playbook(VolState.RANGE.value, recent_break_failed=True)
    assert fade is FAILED_BREAK_ENTRY

    measured = get_playbook(
        VolState.COMPRESSION.value,
        resolution_stats={"median_bars": 9.0, "q75_bars": 15.0},
    )
    assert "9" in measured.tenor_guidance


def test_data_contract(tmp_path, demo):
    ok = validate_ohlc(demo)
    assert len(ok) > 0

    with pytest.raises(DataContractError):
        validate_ohlc(demo.rename(columns={"Close": "close"}))

    with pytest.raises(DataContractError):
        validate_ohlc(demo.iloc[:50])  # too short

    p = tmp_path / "demo.csv"
    demo.rename_axis("Date").to_csv(p)
    loaded = load_ohlc_csv(p)
    assert len(loaded) == len(demo)
    assert "CSV" in loaded.attrs["provenance"]
