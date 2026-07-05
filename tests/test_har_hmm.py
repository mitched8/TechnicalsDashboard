"""HAR: beats the naive trailing-RV forecast OOS on vol-clustered data.
HMM: recovers planted two-regime structure."""

import numpy as np
import pandas as pd

from vol_engine.har import fit_har, har_forecast_report
from vol_engine.hmm import fit_hmm_2state
from vol_engine.rv_estimators import log_returns


def make_vol_clustered(n=2500, seed=21):
    """Two-state regime-switching vol with persistence (vol clustering)."""
    rng = np.random.default_rng(seed)
    state = 0
    sigmas = [0.004, 0.012]
    states = np.zeros(n, dtype=int)
    for i in range(1, n):
        if rng.random() < 0.02:  # persistent regimes
            state = 1 - state
        states[i] = state
    sigma = np.array([sigmas[s] for s in states])
    r = rng.normal(0, sigma)
    c = 1.3 * np.exp(np.cumsum(r))
    o = np.roll(c, 1); o[0] = 1.3
    span = np.abs(rng.normal(0, sigma)) + sigma * 0.5
    df = pd.DataFrame(
        {"Open": o, "High": np.maximum(o, c) * (1 + span),
         "Low": np.minimum(o, c) * (1 - span), "Close": c},
        index=pd.bdate_range("2010-01-04", periods=n),
    )
    return df, states


def test_har_fits_and_beats_unconditional():
    df, _ = make_vol_clustered()
    coeffs = fit_har(df, horizon=21)
    assert coeffs["n"] > 1000

    report = har_forecast_report(df, horizon=21, min_train=400, refit_every=42)
    assert report.n_oos > 500
    # HAR should have positive OOS R^2 (beats unconditional mean) on clustered vol
    assert report.oos_r2_har > 0.1
    # And be at least competitive with the naive random-walk forecast
    assert report.oos_mae_har < report.oos_mae_naive * 1.1


def test_har_augmented_runs():
    df, _ = make_vol_clustered(seed=22)
    extra = pd.DataFrame(index=df.index)
    # A useless extra regressor must not crash or catastrophically hurt
    extra["noise"] = np.random.default_rng(1).normal(0, 1, len(df))
    report = har_forecast_report(df, horizon=21, min_train=400, extra_features=extra)
    assert report.oos_mae_augmented is not None
    assert report.oos_mae_augmented < report.oos_mae_naive * 1.3


def test_hmm_recovers_planted_regimes():
    df, states = make_vol_clustered(seed=23)
    returns = log_returns(df["Close"]).dropna()
    result = fit_hmm_2state(returns, max_iter=100)
    assert result is not None
    # State vols recovered in the right ballpark
    assert result.sigmas[0] < result.sigmas[1]
    assert 0.002 < result.sigmas[0] < 0.007
    assert 0.008 < result.sigmas[1] < 0.018

    truth = pd.Series(states, index=df.index).reindex(result.viterbi.index)
    accuracy = (result.viterbi.values == truth.values).mean()
    assert accuracy > 0.85, f"HMM state accuracy {accuracy:.2f}"


def test_hmm_short_series_returns_none():
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 50))
    assert fit_hmm_2state(r) is None
