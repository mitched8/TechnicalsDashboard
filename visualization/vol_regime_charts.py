"""Plotly charts for the Vol Regime page.

Palette follows the validated reference set (fixed slot order, one axis per chart,
recessive grid, thin marks). Direction (up/down breaks) uses the diverging
blue/red pair; regime states use low-alpha background washes.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Validated categorical slots (light mode)
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#4a3aa7"

# Diverging pair for direction (polarity)
DIV_UP = "#2a78d6"
DIV_DOWN = "#e34948"

INK_MUTED = "#898781"
GRID = "#e1e0d9"

STATE_WASH = {
    "range": "rgba(0,0,0,0)",
    "compression": "rgba(237,161,0,0.14)",     # yellow wash
    "break_attempt": "rgba(235,104,52,0.18)",  # orange wash
    "trend": "rgba(42,120,214,0.10)",          # blue wash
}
STATE_LABEL = {
    "range": "Range",
    "compression": "Compression",
    "break_attempt": "Break attempt",
    "trend": "Trend",
}


def _base_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID)
    return fig


def state_chart(
    df: pd.DataFrame,
    states: pd.Series,
    features: pd.DataFrame,
    events_frame: pd.DataFrame,
    title: str = "Price with vol-regime states",
    last_n: int = 500,
) -> go.Figure:
    """Close price + Donchian channel, state background washes, break markers."""
    d = df.tail(last_n)
    s = states.reindex(d.index)
    f = features.reindex(d.index)

    fig = go.Figure()

    # State background bands: contiguous runs of the same state
    run_id = (s != s.shift()).cumsum()
    for _, run in s.groupby(run_id):
        state = run.iloc[0]
        wash = STATE_WASH.get(state, "rgba(0,0,0,0)")
        if wash == "rgba(0,0,0,0)":
            continue
        fig.add_vrect(
            x0=run.index[0], x1=run.index[-1],
            fillcolor=wash, line_width=0, layer="below",
        )

    fig.add_trace(go.Scatter(
        x=d.index, y=f["don_high"], name="Channel high",
        line=dict(color=INK_MUTED, width=1, dash="dot"), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=d.index, y=f["don_low"], name="Channel low",
        line=dict(color=INK_MUTED, width=1, dash="dot"), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=d.index, y=d["Close"], name="Close",
        line=dict(color=BLUE, width=2),
    ))

    if not events_frame.empty:
        ev = events_frame[events_frame["date"].isin(d.index)]
        for kind, symbol, name in [
            ("break_confirmed", "triangle-up", "Break confirmed"),
            ("break_failed", "x", "Break failed"),
        ]:
            sub = ev[ev["kind"] == kind]
            if sub.empty:
                continue
            colors = [DIV_UP if dr == "up" else DIV_DOWN for dr in sub["direction"]]
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["level"], mode="markers", name=name,
                marker=dict(symbol=symbol, size=11, color=colors,
                            line=dict(width=1, color="#fcfcfb")),
            ))

    # State legend proxies (washes aren't in the legend otherwise)
    for state in ("compression", "break_attempt", "trend"):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=f"{STATE_LABEL[state]} (shading)",
            marker=dict(size=10, symbol="square",
                        color=STATE_WASH[state].replace("0.14", "0.5").replace("0.18", "0.5").replace("0.10", "0.5")),
        ))

    return _base_layout(fig, title, height=480)


def cone_chart(cone: pd.DataFrame, iv_points: Optional[dict] = None,
               title: str = "Realized vol cone") -> go.Figure:
    """Percentile fan by RV window, with current RV per window. Optionally overlay
    IV per tenor-in-days (iv_points: {days: iv_pct}) once real IV is available."""
    fig = go.Figure()
    x = cone.index.tolist()

    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=cone["p95"].tolist() + cone["p5"].tolist()[::-1],
        fill="toself", fillcolor="rgba(42,120,214,0.10)",
        line=dict(width=0), name="5-95 pct", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=cone["p75"].tolist() + cone["p25"].tolist()[::-1],
        fill="toself", fillcolor="rgba(42,120,214,0.22)",
        line=dict(width=0), name="25-75 pct", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=cone["p50"], name="Median",
        line=dict(color=INK_MUTED, width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=cone["current"], name="Current RV",
        mode="lines+markers", line=dict(color=BLUE, width=2),
        marker=dict(size=8),
    ))
    if iv_points:
        fig.add_trace(go.Scatter(
            x=list(iv_points.keys()), y=list(iv_points.values()),
            name="Implied vol", mode="markers",
            marker=dict(size=10, color=VIOLET, symbol="diamond"),
        ))

    fig = _base_layout(fig, title)
    fig.update_xaxes(title="RV window (trading days)")
    fig.update_yaxes(title="Annualized vol %")
    return fig


def rv_panel_chart(features: pd.DataFrame, last_n: int = 500,
                   title: str = "Realized vol by window") -> go.Figure:
    d = features.tail(last_n)
    fig = go.Figure()
    for col, color, name in [
        ("rv_5", BLUE, "RV 5d"),
        ("rv_21", AQUA, "RV 21d"),
        ("rv_63", YELLOW, "RV 63d"),
    ]:
        if col in d.columns:
            fig.add_trace(go.Scatter(
                x=d.index, y=d[col], name=name, line=dict(color=color, width=2),
            ))
    fig = _base_layout(fig, title)
    fig.update_yaxes(title="Annualized vol %")
    return fig


def compression_chart(features: pd.DataFrame, threshold: float, last_n: int = 500,
                      title: str = "Compression score") -> go.Figure:
    d = features.tail(last_n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.index, y=d["compression_score"], name="Compression score",
        line=dict(color=YELLOW, width=2),
    ))
    fig.add_hline(y=threshold, line=dict(color=INK_MUTED, width=1, dash="dash"),
                  annotation_text="threshold", annotation_font_color=INK_MUTED)
    fig = _base_layout(fig, title, height=300)
    fig.update_yaxes(title="Score", range=[0, 1])
    return fig


def event_path_chart(path: pd.DataFrame, title: str,
                     y_title: str = "RV (annualized %)") -> go.Figure:
    """Median/mean of a series in event time (bar 0 = event)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=path.index, y=path["median"], name="Median",
        line=dict(color=BLUE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=path.index, y=path["mean"], name="Mean",
        line=dict(color=INK_MUTED, width=1, dash="dash"),
    ))
    fig.add_vline(x=0, line=dict(color=DIV_DOWN, width=1, dash="dot"),
                  annotation_text="event", annotation_font_color=INK_MUTED)
    fig = _base_layout(fig, title, height=340)
    fig.update_xaxes(title="Bars relative to event")
    fig.update_yaxes(title=y_title)
    return fig


def har_chart(forecast: pd.Series, realized_fwd: pd.Series, last_n: int = 750,
              title: str = "HAR forecast vs subsequently realized vol (21d)") -> go.Figure:
    f = forecast.tail(last_n)
    r = realized_fwd.reindex(f.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=r.index, y=r, name="Realized (next 21d)",
        line=dict(color=AQUA, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=f.index, y=f, name="HAR forecast",
        line=dict(color=VIOLET, width=2),
    ))
    fig = _base_layout(fig, title)
    fig.update_yaxes(title="Annualized vol %")
    return fig


def breadth_chart(breadth: pd.DataFrame, last_n: int = 750,
                  title: str = "Cross-pair regime breadth") -> go.Figure:
    """Fraction of the pair complex in each vol state over time."""
    d = breadth.tail(last_n)
    fig = go.Figure()
    for col, color, name in [
        ("fraction_compressed", YELLOW, "Compressed"),
        ("fraction_trending", BLUE, "Trending"),
        ("fraction_break", "#eb6834", "Break attempt"),
    ]:
        if col in d.columns:
            fig.add_trace(go.Scatter(
                x=d.index, y=d[col], name=name, line=dict(color=color, width=2),
            ))
    fig = _base_layout(fig, title, height=340)
    fig.update_yaxes(title="Fraction of pairs", range=[0, 1])
    return fig


def hmm_chart(p_high: pd.Series, last_n: int = 750,
              title: str = "HMM cross-check: P(high-vol state)") -> go.Figure:
    d = p_high.tail(last_n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.index, y=d, name="P(high vol)",
        line=dict(color=BLUE, width=2),
        fill="tozeroy", fillcolor="rgba(42,120,214,0.15)",
    ))
    fig = _base_layout(fig, title, height=300)
    fig.update_yaxes(title="Probability", range=[0, 1])
    return fig
