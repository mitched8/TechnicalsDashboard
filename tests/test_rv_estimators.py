"""Estimator correctness: convergence to true sigma on simulated GBM, and the
efficiency ordering (range-based estimators less noisy than close-to-close)."""

import numpy as np
import pandas as pd
import pytest

from vol_engine.rv_estimators import (
    ESTIMATORS, realized_vol, rv_panel, rv_term_ratios, ANNUALIZATION,
)


def make_gbm_ohlc(n=4000, sigma_daily=0.006, seed=1, steps_per_day=400):
    # steps_per_day matters: discretely-sampled highs/lows understate the
    # continuous extremes, biasing range-based estimators low. 400 steps keeps
    # that discretization bias inside the test tolerance.
    """GBM path sampled intraday so H/L are meaningful."""
    rng = np.random.default_rng(seed)
    step_sigma = sigma_daily / np.sqrt(steps_per_day)
    opens, highs, lows, closes = [], [], [], []
    p = 1.0
    for _ in range(n):
        o = p
        path = o * np.exp(np.cumsum(rng.normal(0, step_sigma, steps_per_day)))
        c = path[-1]
        highs.append(max(o, path.max()))
        lows.append(min(o, path.min()))
        opens.append(o)
        closes.append(c)
        p = c
    idx = pd.bdate_range("2005-01-03", periods=n)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


TRUE_VOL_PCT = 0.006 * np.sqrt(ANNUALIZATION) * 100  # ~9.5%


@pytest.fixture(scope="module")
def gbm():
    return make_gbm_ohlc()


@pytest.mark.parametrize("name", sorted(ESTIMATORS))
def test_estimators_converge_to_true_sigma(gbm, name):
    rv = realized_vol(gbm, window=63, estimator=name).dropna()
    # Long-run average within 10% of truth
    assert abs(rv.mean() - TRUE_VOL_PCT) / TRUE_VOL_PCT < 0.10, (
        f"{name}: mean {rv.mean():.2f} vs true {TRUE_VOL_PCT:.2f}"
    )


def test_range_estimators_more_efficient_than_cc(gbm):
    cc = realized_vol(gbm, 21, "close_to_close").dropna()
    gk = realized_vol(gbm, 21, "garman_klass").dropna()
    pk = realized_vol(gbm, 21, "parkinson").dropna()
    assert gk.std() < cc.std()
    assert pk.std() < cc.std()


def test_rv_panel_and_term_ratios(gbm):
    panel = rv_panel(gbm, windows=(5, 21, 63))
    assert {"rv_5", "rv_21", "rv_63"} <= set(panel.columns)
    ratios = rv_term_ratios(panel).dropna()
    # Constant-vol world: ratios hover near 1
    assert abs(ratios["rv_5_over_21"].median() - 1.0) < 0.15


def test_unknown_estimator_raises(gbm):
    with pytest.raises(ValueError):
        realized_vol(gbm, 21, "nope")
