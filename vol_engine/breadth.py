"""Cross-pair breadth: market-wide compression/expansion context.

A break confirmed while most of the complex is also expanding is more likely genuine
than a single-pair move; a lone-pair "breakout" against a quiet board deserves
suspicion. These helpers aggregate per-pair regime output into breadth measures.
"""

from typing import Dict

import pandas as pd

from vol_engine.regime import RegimeResult, VolState


def compression_breadth(results: Dict[str, RegimeResult]) -> pd.DataFrame:
    """Daily breadth across pairs.

    Args:
        results: mapping pair name -> RegimeResult (states aligned on dates).

    Returns:
        DataFrame with fraction_compressed, fraction_trending, fraction_break,
        avg_compression_score, n_pairs per date (union of indices, NaN-safe).
    """
    if not results:
        return pd.DataFrame()

    states = pd.DataFrame({name: r.states for name, r in results.items()})
    scores = pd.DataFrame({
        name: r.features["compression_score"]
        for name, r in results.items()
        if r.features is not None and "compression_score" in r.features
    })

    n_pairs = states.notna().sum(axis=1)
    out = pd.DataFrame(index=states.index)
    out["fraction_compressed"] = (states == VolState.COMPRESSION.value).sum(axis=1) / n_pairs
    out["fraction_trending"] = (states == VolState.TREND.value).sum(axis=1) / n_pairs
    out["fraction_break"] = (states == VolState.BREAK_ATTEMPT.value).sum(axis=1) / n_pairs
    if not scores.empty:
        out["avg_compression_score"] = scores.mean(axis=1)
    out["n_pairs"] = n_pairs
    return out
