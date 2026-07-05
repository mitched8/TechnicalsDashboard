"""HAR-RV forecaster (Corsi) with walk-forward out-of-sample evaluation.

Model: log(fwd RV over h bars) ~ b0 + bd*log(RV_1d) + bw*log(RV_5d) + bm*log(RV_21d)

Fitting on log-RV stabilizes the regression; forecasts are exponentiated back
(small retransformation bias accepted — this is a relative-skill tool, and the
benchmark comparison uses the same transform).

The regime-augmented variant adds the compression score and a trend dummy; its
*incremental* out-of-sample skill measures whether the regime layer adds forecast
power beyond vanilla HAR — the honest test of the whole regime framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from vol_engine.rv_estimators import PER_BAR_VAR, ANNUALIZATION, DEFAULT_ESTIMATOR
from vol_engine.event_study import forward_rv

EPS = 1e-12


def _har_features(df: pd.DataFrame, estimator: str = DEFAULT_ESTIMATOR) -> pd.DataFrame:
    var_fn = PER_BAR_VAR.get(estimator, PER_BAR_VAR["garman_klass"])
    daily_var = var_fn(df).clip(lower=0.0)

    def ann_vol(mean_var: pd.Series) -> pd.Series:
        return np.sqrt(mean_var * ANNUALIZATION) * 100.0

    out = pd.DataFrame(index=df.index)
    out["rv_d"] = ann_vol(daily_var)
    out["rv_w"] = ann_vol(daily_var.rolling(5).mean())
    out["rv_m"] = ann_vol(daily_var.rolling(21).mean())
    return out


@dataclass
class HARReport:
    horizon: int
    n_train_min: int
    n_oos: int
    coefficients: Dict[str, float]
    oos_mae_har: float
    oos_mae_naive: float           # naive = trailing RV of same horizon
    oos_r2_har: float              # OOS R^2 vs unconditional mean benchmark
    oos_r2_naive: float
    oos_mae_augmented: Optional[float] = None
    oos_r2_augmented: Optional[float] = None
    forecast: Optional[pd.Series] = None   # latest walk-forward forecasts (vol %)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        s = {
            "horizon": self.horizon,
            "n_oos": self.n_oos,
            "oos_mae_har": round(self.oos_mae_har, 3),
            "oos_mae_naive": round(self.oos_mae_naive, 3),
            "oos_r2_har": round(self.oos_r2_har, 3),
            "oos_r2_naive": round(self.oos_r2_naive, 3),
        }
        if self.oos_mae_augmented is not None:
            s["oos_mae_augmented"] = round(self.oos_mae_augmented, 3)
            s["oos_r2_augmented"] = round(self.oos_r2_augmented, 3)
        return s


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta


def _predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return beta[0] + X @ beta[1:]


def fit_har(
    df: pd.DataFrame,
    horizon: int = 21,
    estimator: str = DEFAULT_ESTIMATOR,
) -> Dict[str, float]:
    """Full-sample HAR fit; returns coefficients (for inspection, not evaluation)."""
    feats = _har_features(df, estimator)
    target = forward_rv(df, horizon, estimator)
    data = feats.join(target.rename("y")).dropna()
    X = np.log(data[["rv_d", "rv_w", "rv_m"]].values + EPS)
    y = np.log(data["y"].values + EPS)
    beta = _ols(X, y)
    return {"const": beta[0], "log_rv_d": beta[1], "log_rv_w": beta[2], "log_rv_m": beta[3],
            "n": len(data)}


def har_forecast_report(
    df: pd.DataFrame,
    horizon: int = 21,
    estimator: str = DEFAULT_ESTIMATOR,
    min_train: int = 252,
    refit_every: int = 21,
    extra_features: Optional[pd.DataFrame] = None,
) -> HARReport:
    """Walk-forward HAR evaluation.

    At each refit point, fit on all data whose *target* is fully realized (no
    lookahead into the evaluation bar), then forecast forward until the next refit.

    Args:
        extra_features: optional additional causal regressors aligned to df.index
            (e.g. compression score, trend dummy) for the augmented variant.
    """
    feats = _har_features(df, estimator)
    target = forward_rv(df, horizon, estimator)

    base_cols = ["rv_d", "rv_w", "rv_m"]
    X_base = np.log(feats[base_cols].values + EPS)
    y_log = np.log(target.values + EPS)
    y_raw = target.values

    if extra_features is not None:
        extra = extra_features.reindex(df.index).values.astype(float)
    else:
        extra = None

    n = len(df)
    pred_har = np.full(n, np.nan)
    pred_aug = np.full(n, np.nan)

    valid_feat = ~np.isnan(X_base).any(axis=1)
    if extra is not None:
        valid_extra = ~np.isnan(extra).any(axis=1)
    else:
        valid_extra = np.ones(n, dtype=bool)
    valid_target = ~np.isnan(y_log)

    beta_har = None
    beta_aug = None
    coeffs: Dict[str, float] = {}

    for t in range(min_train, n):
        if (t - min_train) % refit_every == 0:
            # Training rows: features known at s, target realized by s + horizon <= t.
            train = np.arange(0, t - horizon)
            train = train[valid_feat[train] & valid_target[train]]
            if len(train) >= 60:
                beta_har = _ols(X_base[train], y_log[train])
                coeffs = {"const": beta_har[0], "log_rv_d": beta_har[1],
                          "log_rv_w": beta_har[2], "log_rv_m": beta_har[3]}
                if extra is not None:
                    train_aug = train[valid_extra[train]]
                    if len(train_aug) >= 60:
                        X_aug_train = np.column_stack([X_base[train_aug], extra[train_aug]])
                        beta_aug = _ols(X_aug_train, y_log[train_aug])
        if beta_har is not None and valid_feat[t]:
            pred_har[t] = np.exp(_predict(beta_har, X_base[t][None, :])[0])
            if beta_aug is not None and valid_extra[t]:
                x_aug = np.concatenate([X_base[t], extra[t]])[None, :]
                pred_aug[t] = np.exp(_predict(beta_aug, x_aug)[0])

    # Naive benchmark: trailing RV at the same horizon (random-walk forecast).
    from vol_engine.event_study import trailing_rv
    naive = trailing_rv(df, horizon, estimator).values

    eval_mask = ~np.isnan(pred_har) & ~np.isnan(y_raw) & ~np.isnan(naive)
    n_oos = int(eval_mask.sum())

    report = HARReport(
        horizon=horizon, n_train_min=min_train, n_oos=n_oos, coefficients=coeffs,
        oos_mae_har=float("nan"), oos_mae_naive=float("nan"),
        oos_r2_har=float("nan"), oos_r2_naive=float("nan"),
    )
    if n_oos < 30:
        report.warnings.append(
            f"Only {n_oos} out-of-sample points — evaluation not meaningful. "
            "Provide more history."
        )
        return report

    y_e = y_raw[eval_mask]
    mean_bench = np.full_like(y_e, y_e.mean())

    def mae(p):
        return float(np.mean(np.abs(p - y_e)))

    def r2(p):
        ss_res = np.sum((p - y_e) ** 2)
        ss_tot = np.sum((y_e - mean_bench) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    report.oos_mae_har = mae(pred_har[eval_mask])
    report.oos_mae_naive = mae(naive[eval_mask])
    report.oos_r2_har = r2(pred_har[eval_mask])
    report.oos_r2_naive = r2(naive[eval_mask])

    aug_mask = eval_mask & ~np.isnan(pred_aug)
    if extra is not None and aug_mask.sum() >= 30:
        y_a = y_raw[aug_mask]
        report.oos_mae_augmented = float(np.mean(np.abs(pred_aug[aug_mask] - y_a)))
        ss_res = np.sum((pred_aug[aug_mask] - y_a) ** 2)
        ss_tot = np.sum((y_a - y_a.mean()) ** 2)
        report.oos_r2_augmented = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    report.forecast = pd.Series(pred_har, index=df.index, name=f"har_rv_{horizon}")
    return report
