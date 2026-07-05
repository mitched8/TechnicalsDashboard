"""State -> options-stance playbook.

Declarative mapping from vol state to trading stance, FX-native structures and
caveats. Tenor guidance is derived from *measured* time-to-resolution when event-study
stats are supplied; the static entries are defaults to be overridden by evidence.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from vol_engine.regime import VolState


@dataclass
class PlaybookEntry:
    state: str
    stance: str
    rationale: str
    structures: List[str] = field(default_factory=list)
    tenor_guidance: str = ""
    caveats: List[str] = field(default_factory=list)


PLAYBOOK: Dict[str, PlaybookEntry] = {
    VolState.RANGE.value: PlaybookEntry(
        state="range",
        stance="Harvest premium (short vol), size small",
        rationale=(
            "In an established range RV bleeds lower and IV typically holds a premium "
            "over RV before following it down. Theta collection works until the break."
        ),
        structures=[
            "Short strangle with strikes beyond the range edges",
            "Iron condor if defined risk is required",
            "Short calendar only if the term structure is steep",
        ],
        tenor_guidance="Sell tenors shorter than the typical range residual life",
        caveats=[
            "Track range age / level-test count: many-times-tested levels deplete "
            "(higher break risk) — reduce or exit short vol as tests accumulate",
            "This is the consensus carry trade; the edge is exiting on time, not entering",
        ],
    ),
    VolState.COMPRESSION.value: PlaybookEntry(
        state="compression",
        stance="Flip long vol / own gamma",
        rationale=(
            "Forward RV out of compression is bimodal (keep compressing, or break and "
            "expand) while IV has usually bled to a low percentile — convexity is "
            "cheapest exactly when its value is highest."
        ),
        structures=[
            "Long straddle / strangle in the tenor matching measured resolution time",
            "Calendar (short front, long back) if front IV still holds a premium",
            "Ratio backspread for a directional lean with long vega",
        ],
        tenor_guidance="Buy the tenor bracketing the median time-to-resolution",
        caveats=[
            "Bleed risk: if compression persists, theta hurts — the measured "
            "resolution-time distribution sets the stop-clock",
            "Check the event study's expansion probability for this pair before sizing",
        ],
    ),
    VolState.BREAK_ATTEMPT.value: PlaybookEntry(
        state="break_attempt",
        stance="Hold/add gamma, delta-hedge; do not sell vol",
        rationale=(
            "RV jumps at breaks and IV reprices with it; skew shifts toward the break. "
            "Unconfirmed breaks can still fail — gamma is the position that wins "
            "either way if bought before the attempt."
        ),
        structures=[
            "Keep long straddle/strangle, hedge deltas on the move",
            "If flat: buy short-dated vol only if IV has not yet repriced",
        ],
        tenor_guidance="Short-dated (1-3 weeks) — the resolution is imminent by definition",
        caveats=[
            "Buying vol AFTER the expansion bar usually pays the repricing — check "
            "IV percentile before chasing",
            "A failed break flips the stance entirely (see below)",
        ],
    ),
    VolState.TREND.value: PlaybookEntry(
        state="trend",
        stance="Convert gamma to direction; then fade vol as the trend matures",
        rationale=(
            "RV peaks early after confirmation, then orderly FX trends grind with "
            "DECLINING realized vol while IV often stays bid — the counterintuitive "
            "short-vol entry. Risk-reversal skew richens with the trend."
        ),
        structures=[
            "Risk reversal positioned with the trend (finance the directional wing "
            "by selling the rich counter-trend wing)",
            "Directional call/put spreads instead of naked gamma",
            "Once trend is mature: sell strangles re-centered on the drift",
        ],
        tenor_guidance="Directional structures 1-3M; short vol shorter-dated",
        caveats=[
            "Exit long gamma soon after confirmation — the event-time RV path chart "
            "shows how fast the gamma payoff decays for this pair",
            "Trend-end (efficiency-ratio decay) re-enters range rules",
        ],
    ),
}

FAILED_BREAK_ENTRY = PlaybookEntry(
    state="break_failed",
    stance="Fade: sell the vol spike, mean-revert delta",
    rationale=(
        "A failed break traps breakout positioning: price snaps back into the range "
        "and the IV spike deflates. Historically one of the better short-vol moments "
        "in FX ranges."
    ),
    structures=[
        "Sell short-dated strangle into the spike",
        "Mean-reversion spot/delta back toward mid-range against the trapped move",
    ],
    tenor_guidance="Short-dated (<= 1M) to monetize the IV crush",
    caveats=["Wrong if the 'failure' was only a retest — require the close back inside"],
)


def get_playbook(
    state: str,
    resolution_stats: Optional[dict] = None,
    recent_break_failed: bool = False,
) -> PlaybookEntry:
    """Playbook entry for the current state.

    Args:
        resolution_stats: output of event_study.time_to_resolution for compression
            events — used to replace the generic tenor guidance with a measured one.
        recent_break_failed: True if the latest resolved break failed (overrides
            the range entry with the fade playbook).
    """
    if recent_break_failed and state == VolState.RANGE.value:
        return FAILED_BREAK_ENTRY

    entry = PLAYBOOK.get(state)
    if entry is None:
        return PlaybookEntry(state=state, stance="No guidance", rationale="Unknown state")

    if (
        state == VolState.COMPRESSION.value
        and resolution_stats
        and "median_bars" in resolution_stats
    ):
        med = resolution_stats["median_bars"]
        q75 = resolution_stats.get("q75_bars", med)
        entry = PlaybookEntry(
            state=entry.state,
            stance=entry.stance,
            rationale=entry.rationale,
            structures=entry.structures,
            tenor_guidance=(
                f"Measured: median resolution {med:.0f} bars (75th pct {q75:.0f}) — "
                f"favor expiries covering ~{int(med)}-{int(q75)} trading days"
            ),
            caveats=entry.caveats,
        )
    return entry
