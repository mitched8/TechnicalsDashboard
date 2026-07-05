"""Synthetic OHLC generation with planted regime structure.

FOR ENGINE VALIDATION AND CLEARLY-LABELED DEMO USE ONLY. Every frame produced here
carries attrs['synthetic'] = True; UI surfaces must display that provenance. The point
of planted ground truth is to verify the detectors detect (tests), and to demo the
engine's mechanics when no long real history is available on this machine.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Segment:
    """One regime segment of a synthetic path."""
    kind: str          # "range" | "compression" | "trend"
    n_bars: int
    daily_vol: float   # daily log-return sigma (e.g. 0.005 = ~8% annualized)
    drift: float = 0.0  # daily log drift (trends use nonzero)


def default_script(cycles: int = 8) -> List[Segment]:
    """A repeating range -> compression -> trend(break) script, ~15y at cycles=8."""
    script: List[Segment] = []
    rng = np.random.default_rng(123)
    for k in range(cycles):
        direction = 1.0 if k % 2 == 0 else -1.0
        script += [
            Segment("range", int(rng.integers(120, 200)), 0.0050),
            Segment("compression", int(rng.integers(25, 45)), 0.0022),
            Segment("trend", int(rng.integers(40, 80)), 0.0085, drift=direction * 0.0028),
        ]
    return script


def generate_ohlc(
    script: Optional[List[Segment]] = None,
    s0: float = 1.10,
    seed: int = 42,
    start: str = "2010-01-04",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic OHLC from a regime script.

    Returns:
        (ohlc frame, ground-truth label Series aligned to the frame's index).
        Frame carries attrs: synthetic=True, provenance description.
    """
    segs = script or default_script()
    rng = np.random.default_rng(seed)

    closes: List[float] = []
    opens: List[float] = []
    highs: List[float] = []
    lows: List[float] = []
    labels: List[str] = []

    c = s0
    # Range segments mean-revert around an anchor so channels genuinely form.
    for seg in segs:
        anchor = c
        for _ in range(seg.n_bars):
            o = c
            if seg.kind in ("range", "compression"):
                # OU-style pull toward the anchor keeps price inside a channel;
                # strong enough that channel breaks within ranges usually fail,
                # as in real FX ranges.
                shock = rng.normal(0.0, seg.daily_vol)
                theta = 0.12 if seg.kind == "range" else 0.20
                pull = -theta * np.log(c / anchor)
                r = shock + pull
            else:  # trend
                r = rng.normal(seg.drift, seg.daily_vol)
            c = o * np.exp(r)
            # Intrabar range scaled to the segment's vol.
            span = abs(rng.normal(0.0, seg.daily_vol)) + seg.daily_vol * 0.5
            hi = max(o, c) * np.exp(span * rng.uniform(0.3, 0.7))
            lo = min(o, c) * np.exp(-span * rng.uniform(0.3, 0.7))
            opens.append(o)
            closes.append(c)
            highs.append(hi)
            lows.append(lo)
            labels.append(seg.kind)

    idx = pd.bdate_range(start=start, periods=len(closes))
    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes},
        index=idx,
    )
    df.attrs["synthetic"] = True
    df.attrs["provenance"] = (
        "SYNTHETIC demo data (planted regimes, seeded RNG) — for engine "
        "validation/demo only. Never trade off this."
    )
    truth = pd.Series(labels, index=idx, name="true_regime")
    return df, truth
