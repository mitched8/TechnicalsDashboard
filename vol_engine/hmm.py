"""Lightweight 2-state Gaussian HMM on returns — statistical cross-check for the
rule-based regime states.

Numpy-only Baum-Welch (EM) with zero-mean Gaussian emissions parameterized by
state volatility. Deterministic initialization (low-vol/high-vol split), capped
iterations, forward-backward smoothing and Viterbi decoding.

This is intentionally minimal: the rule-based state machine is the primary regime
model (transparent, debuggable); the HMM answers "does an agnostic statistical model
see the same high-vol/low-vol episodes?"
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class HMMResult:
    sigmas: np.ndarray             # state vols (daily, in return units), low first
    transition: np.ndarray         # 2x2 transition matrix
    smoothed_p_high: pd.Series     # P(high-vol state | all data)
    viterbi: pd.Series             # hard state path: 0 = low vol, 1 = high vol
    log_likelihood: float
    n_iter: int
    converged: bool


def _gaussian_loglik(x: np.ndarray, sigma: float) -> np.ndarray:
    return -0.5 * np.log(2.0 * np.pi * sigma**2) - 0.5 * (x / sigma) ** 2


def fit_hmm_2state(
    returns: pd.Series,
    max_iter: int = 200,
    tol: float = 1e-6,
    min_sigma: float = 1e-6,
) -> Optional[HMMResult]:
    """Fit a 2-state zero-mean Gaussian HMM to a return series.

    Returns None if the series is too short (< 100 finite observations).
    """
    r = returns.dropna()
    x = r.values.astype(float)
    n = len(x)
    if n < 100:
        return None

    # Deterministic init: split by median absolute return.
    med = np.median(np.abs(x))
    lo = np.abs(x) <= med
    sigmas = np.array([
        max(x[lo].std(), min_sigma),
        max(x[~lo].std(), min_sigma),
    ])
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi = np.array([0.5, 0.5])

    prev_ll = -np.inf
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        # E-step (log-space forward-backward)
        logB = np.column_stack([_gaussian_loglik(x, s) for s in sigmas])  # n x 2
        logA = np.log(A)

        log_alpha = np.zeros((n, 2))
        log_alpha[0] = np.log(pi) + logB[0]
        for t in range(1, n):
            m = log_alpha[t - 1][:, None] + logA
            log_alpha[t] = logB[t] + np.logaddexp(m[0], m[1])

        log_beta = np.zeros((n, 2))
        for t in range(n - 2, -1, -1):
            m = logA + logB[t + 1][None, :] + log_beta[t + 1][None, :]
            log_beta[t] = np.logaddexp(m[:, 0], m[:, 1])

        ll = np.logaddexp(log_alpha[-1, 0], log_alpha[-1, 1])

        log_gamma = log_alpha + log_beta - ll
        gamma = np.exp(log_gamma)

        # xi sums for transition update
        xi_sum = np.zeros((2, 2))
        for t in range(n - 1):
            log_xi = (log_alpha[t][:, None] + logA
                      + logB[t + 1][None, :] + log_beta[t + 1][None, :] - ll)
            xi_sum += np.exp(log_xi)

        # M-step
        pi = gamma[0] / gamma[0].sum()
        A = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        for k in range(2):
            w = gamma[:, k]
            sigmas[k] = max(np.sqrt((w * x**2).sum() / w.sum()), min_sigma)

        if abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
            converged = True
            break
        prev_ll = ll

    # Order states: index 0 = low vol.
    if sigmas[0] > sigmas[1]:
        sigmas = sigmas[::-1]
        A = A[::-1, ::-1]
        gamma = gamma[:, ::-1]

    # Viterbi
    logB = np.column_stack([_gaussian_loglik(x, s) for s in sigmas])
    logA = np.log(A)
    delta = np.zeros((n, 2))
    psi = np.zeros((n, 2), dtype=int)
    delta[0] = np.log(np.array([0.5, 0.5])) + logB[0]
    for t in range(1, n):
        m = delta[t - 1][:, None] + logA
        psi[t] = m.argmax(axis=0)
        delta[t] = m.max(axis=0) + logB[t]
    path = np.zeros(n, dtype=int)
    path[-1] = delta[-1].argmax()
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1][path[t + 1]]

    return HMMResult(
        sigmas=sigmas,
        transition=A,
        smoothed_p_high=pd.Series(gamma[:, 1], index=r.index, name="p_high_vol"),
        viterbi=pd.Series(path, index=r.index, name="hmm_state"),
        log_likelihood=float(ll),
        n_iter=it,
        converged=converged,
    )
