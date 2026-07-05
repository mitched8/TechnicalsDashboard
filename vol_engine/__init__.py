"""Vol Regime Engine — standalone, data-source-agnostic FX vol analytics.

Pure pandas/numpy. No Streamlit, no `ta` dependency. Input contract everywhere:
a DataFrame with a DatetimeIndex and columns Open, High, Low, Close.
"""

from vol_engine.rv_estimators import (
    log_returns, realized_vol, rv_panel, rv_term_ratios,
    ESTIMATORS, DEFAULT_ESTIMATOR,
)
from vol_engine.cones import vol_cone, rolling_rv_percentile
from vol_engine.regime import (
    RegimeConfig, VolState, RegimeEvent, compute_features, classify_states,
)
from vol_engine.event_study import (
    event_study, event_time_path, time_to_resolution, forward_rv,
)
from vol_engine.har import fit_har, har_forecast_report
from vol_engine.hmm import fit_hmm_2state
from vol_engine.breakeven import breakeven_scoreboard
from vol_engine.breadth import compression_breadth
from vol_engine.playbook import get_playbook, PLAYBOOK
