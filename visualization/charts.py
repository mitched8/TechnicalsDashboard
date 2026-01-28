"""Plotly chart creation for FX Trading Dashboard."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional

from core.support_resistance import SRResult, ZoneType
from core.trade_setup import TradeSetup, Bias


def create_main_chart(
    data: pd.DataFrame,
    sr_result: SRResult,
    trade_setup: Optional[TradeSetup] = None,
    title: str = "FX Analysis"
) -> go.Figure:
    """
    Create interactive candlestick chart with S/R zones and trade setup.

    Args:
        data: OHLCV DataFrame with indicators
        sr_result: Support/resistance analysis result
        trade_setup: Optional trade setup to overlay
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(title, 'RSI', 'MACD')
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1
    )

    # SMAs
    if 'sma_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['sma_20'],
                mode='lines', name='SMA 20',
                line=dict(color='#ffeb3b', width=1),
            ),
            row=1, col=1
        )
    if 'sma_50' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['sma_50'],
                mode='lines', name='SMA 50',
                line=dict(color='#2196f3', width=1),
            ),
            row=1, col=1
        )

    # Bollinger Bands
    if 'bb_upper' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['bb_upper'],
                mode='lines', name='BB Upper',
                line=dict(color='rgba(128,128,128,0.4)', dash='dash', width=1),
                showlegend=False,
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['bb_lower'],
                mode='lines', name='BB Lower',
                line=dict(color='rgba(128,128,128,0.4)', dash='dash', width=1),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                showlegend=False,
            ),
            row=1, col=1
        )

    # Support zones (green)
    for i, zone in enumerate(sr_result.supports[:3]):
        fig.add_hrect(
            y0=zone.lower_bound, y1=zone.upper_bound,
            fillcolor="rgba(76,175,80,0.15)",
            line=dict(color="rgba(76,175,80,0.6)", width=1),
            row=1, col=1,
        )
        # Add annotation for the zone
        fig.add_annotation(
            x=data.index[-1],
            y=zone.center,
            text=f"S{i+1}: {zone.center:.5f}",
            showarrow=False,
            xanchor="left",
            font=dict(color="green", size=10),
            row=1, col=1,
        )

    # Resistance zones (red)
    for i, zone in enumerate(sr_result.resistances[:3]):
        fig.add_hrect(
            y0=zone.lower_bound, y1=zone.upper_bound,
            fillcolor="rgba(244,67,54,0.15)",
            line=dict(color="rgba(244,67,54,0.6)", width=1),
            row=1, col=1,
        )
        fig.add_annotation(
            x=data.index[-1],
            y=zone.center,
            text=f"R{i+1}: {zone.center:.5f}",
            showarrow=False,
            xanchor="left",
            font=dict(color="red", size=10),
            row=1, col=1,
        )

    # Trade setup visualization
    if trade_setup and trade_setup.bias != Bias.NEUTRAL:
        # Entry zone (blue)
        fig.add_hrect(
            y0=trade_setup.entry_zone_low,
            y1=trade_setup.entry_zone_high,
            fillcolor="rgba(33,150,243,0.2)",
            line=dict(color="rgba(33,150,243,0.8)", width=2),
            row=1, col=1,
        )

        # Stop loss line (red dashed)
        fig.add_hline(
            y=trade_setup.stop_loss,
            line=dict(color="red", width=2, dash="dash"),
            row=1, col=1,
        )
        fig.add_annotation(
            x=data.index[0],
            y=trade_setup.stop_loss,
            text=f"SL: {trade_setup.stop_loss:.5f}",
            showarrow=False,
            xanchor="right",
            font=dict(color="red", size=10),
            row=1, col=1,
        )

        # Target 1 (green dashed)
        fig.add_hline(
            y=trade_setup.target_1,
            line=dict(color="green", width=2, dash="dash"),
            row=1, col=1,
        )
        fig.add_annotation(
            x=data.index[0],
            y=trade_setup.target_1,
            text=f"T1: {trade_setup.target_1:.5f}",
            showarrow=False,
            xanchor="right",
            font=dict(color="green", size=10),
            row=1, col=1,
        )

        # Target 2 (lighter green)
        if trade_setup.target_2:
            fig.add_hline(
                y=trade_setup.target_2,
                line=dict(color="rgba(76,175,80,0.6)", width=1, dash="dot"),
                row=1, col=1,
            )
            fig.add_annotation(
                x=data.index[0],
                y=trade_setup.target_2,
                text=f"T2: {trade_setup.target_2:.5f}",
                showarrow=False,
                xanchor="right",
                font=dict(color="green", size=9),
                row=1, col=1,
            )

    # RSI subplot
    if 'rsi' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['rsi'],
                mode='lines', name='RSI',
                line=dict(color='#9c27b0', width=1),
            ),
            row=2, col=1
        )
        fig.add_hline(y=70, line=dict(color='rgba(244,67,54,0.5)', dash='dash', width=1), row=2, col=1)
        fig.add_hline(y=30, line=dict(color='rgba(76,175,80,0.5)', dash='dash', width=1), row=2, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(128,128,128,0.1)", line_width=0, row=2, col=1)

    # MACD subplot
    if 'macd' in data.columns and 'macd_signal' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['macd'],
                mode='lines', name='MACD',
                line=dict(color='#2196f3', width=1),
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['macd_signal'],
                mode='lines', name='Signal',
                line=dict(color='#ff9800', width=1),
            ),
            row=3, col=1
        )

        if 'macd_histogram' in data.columns:
            colors = ['#26a69a' if v >= 0 else '#ef5350' for v in data['macd_histogram']]
            fig.add_trace(
                go.Bar(
                    x=data.index, y=data['macd_histogram'],
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.5,
                ),
                row=3, col=1
            )

        fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', width=1), row=3, col=1)

    # Layout
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=120, t=60, b=40),
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

    # Update y-axis for RSI
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    return fig


def create_mtf_summary_chart(
    weekly_data: pd.DataFrame,
    daily_data: pd.DataFrame,
    four_hour_data: pd.DataFrame,
    symbol: str
) -> go.Figure:
    """
    Create a mini multi-timeframe overview chart.

    Args:
        weekly_data: Weekly OHLCV data
        daily_data: Daily OHLCV data
        four_hour_data: 4-hour OHLCV data
        symbol: Currency pair symbol

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Weekly', 'Daily', '4-Hour'),
        horizontal_spacing=0.05,
    )

    datasets = [
        ('Weekly', weekly_data.tail(52) if len(weekly_data) >= 52 else weekly_data),
        ('Daily', daily_data.tail(60) if len(daily_data) >= 60 else daily_data),
        ('4H', four_hour_data.tail(100) if len(four_hour_data) >= 100 else four_hour_data),
    ]

    for idx, (name, data) in enumerate(datasets, 1):
        if data.empty:
            continue

        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=name,
                showlegend=False,
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
            ),
            row=1, col=idx
        )

        # Add SMA if available
        if 'sma_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['sma_20'],
                    mode='lines', name='SMA 20',
                    line=dict(color='#ffeb3b', width=1),
                    showlegend=False,
                ),
                row=1, col=idx
            )

    fig.update_layout(
        height=250,
        template='plotly_dark',
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=20),
    )

    # Disable range sliders for all subplots
    fig.update_xaxes(rangeslider_visible=False)

    return fig


def create_regime_indicator(
    regime_description: str,
    trend_emoji: str,
    is_bullish: bool
) -> go.Figure:
    """
    Create a simple regime indicator gauge.

    Args:
        regime_description: Text description of regime
        trend_emoji: Emoji for trend direction
        is_bullish: Whether the regime is bullish

    Returns:
        Plotly Figure object
    """
    color = "#26a69a" if is_bullish else "#ef5350"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        title={'text': regime_description},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
        }
    ))

    fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=40, b=20),
        template='plotly_dark',
    )

    return fig


def create_vol_chart(
    data: pd.DataFrame,
    title: str = "Realized Volatility (1M)"
) -> go.Figure:
    """
    Create volatility time series chart for options traders.

    Shows realized vol with percentile bands and squeeze detection.

    Args:
        data: DataFrame with vol columns (rv_1m, rv_percentile, bb_width)
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, 'Vol Percentile & BB Width')
    )

    # Main volatility time series
    if 'rv_1m' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['rv_1m'],
                mode='lines',
                name='1M Realized Vol',
                line=dict(color='#9c27b0', width=2),
                fill='tozeroy',
                fillcolor='rgba(156, 39, 176, 0.1)',
            ),
            row=1, col=1
        )

    if 'rv_3m' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['rv_3m'],
                mode='lines',
                name='3M Realized Vol',
                line=dict(color='#ff9800', width=1, dash='dash'),
            ),
            row=1, col=1
        )

    # Add mean line
    if 'rv_1m' in data.columns:
        mean_vol = data['rv_1m'].mean()
        fig.add_hline(
            y=mean_vol,
            line=dict(color='rgba(255,255,255,0.5)', dash='dot', width=1),
            annotation_text=f"Mean: {mean_vol:.1f}%",
            annotation_position="right",
            row=1, col=1
        )

        # Add high/low bands (1 std dev)
        std_vol = data['rv_1m'].std()
        fig.add_hline(
            y=mean_vol + std_vol,
            line=dict(color='rgba(244,67,54,0.3)', dash='dot', width=1),
            row=1, col=1
        )
        fig.add_hline(
            y=mean_vol - std_vol,
            line=dict(color='rgba(76,175,80,0.3)', dash='dot', width=1),
            row=1, col=1
        )

    # Vol percentile subplot
    if 'rv_percentile' in data.columns:
        # Color by percentile
        percentile_colors = [
            '#ef5350' if p >= 75 else '#26a69a' if p <= 25 else '#9e9e9e'
            for p in data['rv_percentile'].fillna(50)
        ]

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['rv_percentile'],
                mode='lines',
                name='Vol Percentile',
                line=dict(color='#2196f3', width=1),
            ),
            row=2, col=1
        )

        # Add reference lines
        fig.add_hline(y=75, line=dict(color='rgba(244,67,54,0.5)', dash='dash', width=1), row=2, col=1)
        fig.add_hline(y=25, line=dict(color='rgba(76,175,80,0.5)', dash='dash', width=1), row=2, col=1)
        fig.add_hline(y=50, line=dict(color='rgba(255,255,255,0.3)', dash='dot', width=1), row=2, col=1)

    # BB Width on secondary y-axis (squeeze indicator)
    if 'bb_width' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['bb_width'] * 100,  # Convert to percentage
                mode='lines',
                name='BB Width %',
                line=dict(color='#ffeb3b', width=1),
                yaxis='y4',
            ),
            row=2, col=1
        )

    fig.update_layout(
        height=500,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=60, t=60, b=40),
        yaxis4=dict(
            title='BB Width %',
            overlaying='y3',
            side='right',
            showgrid=False,
        ),
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Volatility %", row=1, col=1)
    fig.update_yaxes(title_text="Percentile", range=[0, 100], row=2, col=1)

    return fig


def create_vol_gauge(
    current_vol: float,
    vol_percentile: float,
    forecast: str,
    forecast_confidence: float
) -> go.Figure:
    """
    Create a volatility gauge for quick visual reference.

    Args:
        current_vol: Current 1M realized vol
        vol_percentile: Current vol percentile (0-100)
        forecast: "expanding", "compressing", or "neutral"
        forecast_confidence: Confidence level (0-1)

    Returns:
        Plotly Figure object
    """
    # Determine color based on percentile
    if vol_percentile >= 75:
        color = "#ef5350"  # Red for high vol
        zone_label = "HIGH"
    elif vol_percentile <= 25:
        color = "#26a69a"  # Green for low vol
        zone_label = "LOW"
    else:
        color = "#9e9e9e"  # Gray for normal
        zone_label = "NORMAL"

    # Forecast indicator
    if forecast == "expanding":
        forecast_symbol = "📈"
    elif forecast == "compressing":
        forecast_symbol = "📉"
    else:
        forecast_symbol = "➡️"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=vol_percentile,
        number={'suffix': '%ile', 'font': {'size': 24}},
        title={'text': f"Vol: {current_vol:.1f}% | {zone_label} | {forecast_symbol}", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': 'rgba(38,166,154,0.3)'},
                {'range': [25, 75], 'color': 'rgba(158,158,158,0.2)'},
                {'range': [75, 100], 'color': 'rgba(239,83,80,0.3)'},
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 2},
                'thickness': 0.75,
                'value': vol_percentile
            }
        }
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=50, b=20),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def create_main_chart_with_vol(
    data: pd.DataFrame,
    sr_result: SRResult,
    trade_setup: Optional[TradeSetup] = None,
    title: str = "FX Analysis",
    show_vol: bool = True
) -> go.Figure:
    """
    Create main chart with optional volatility overlay.

    Same as create_main_chart but with 4 rows (adding vol subplot).

    Args:
        data: OHLCV DataFrame with indicators and vol data
        sr_result: Support/resistance analysis result
        trade_setup: Optional trade setup to overlay
        title: Chart title
        show_vol: Whether to show volatility subplot

    Returns:
        Plotly Figure object
    """
    if show_vol and 'rv_1m' in data.columns:
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(title, 'RSI', 'MACD', '1M Realized Vol')
        )
        vol_row = 4
    else:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(title, 'RSI', 'MACD')
        )
        vol_row = None

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1
    )

    # SMAs
    if 'sma_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['sma_20'],
                mode='lines', name='SMA 20',
                line=dict(color='#ffeb3b', width=1),
            ),
            row=1, col=1
        )
    if 'sma_50' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['sma_50'],
                mode='lines', name='SMA 50',
                line=dict(color='#2196f3', width=1),
            ),
            row=1, col=1
        )

    # EMA 200 (if available)
    if 'ema_200' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['ema_200'],
                mode='lines', name='EMA 200',
                line=dict(color='#e91e63', width=1, dash='dot'),
            ),
            row=1, col=1
        )

    # Bollinger Bands
    if 'bb_upper' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['bb_upper'],
                mode='lines', name='BB Upper',
                line=dict(color='rgba(128,128,128,0.4)', dash='dash', width=1),
                showlegend=False,
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['bb_lower'],
                mode='lines', name='BB Lower',
                line=dict(color='rgba(128,128,128,0.4)', dash='dash', width=1),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                showlegend=False,
            ),
            row=1, col=1
        )

    # Support zones (green)
    for i, zone in enumerate(sr_result.supports[:3]):
        fig.add_hrect(
            y0=zone.lower_bound, y1=zone.upper_bound,
            fillcolor="rgba(76,175,80,0.15)",
            line=dict(color="rgba(76,175,80,0.6)", width=1),
            row=1, col=1,
        )
        fig.add_annotation(
            x=data.index[-1],
            y=zone.center,
            text=f"S{i+1}: {zone.center:.5f}",
            showarrow=False,
            xanchor="left",
            font=dict(color="green", size=10),
            row=1, col=1,
        )

    # Resistance zones (red)
    for i, zone in enumerate(sr_result.resistances[:3]):
        fig.add_hrect(
            y0=zone.lower_bound, y1=zone.upper_bound,
            fillcolor="rgba(244,67,54,0.15)",
            line=dict(color="rgba(244,67,54,0.6)", width=1),
            row=1, col=1,
        )
        fig.add_annotation(
            x=data.index[-1],
            y=zone.center,
            text=f"R{i+1}: {zone.center:.5f}",
            showarrow=False,
            xanchor="left",
            font=dict(color="red", size=10),
            row=1, col=1,
        )

    # Trade setup visualization
    if trade_setup and trade_setup.bias != Bias.NEUTRAL:
        fig.add_hrect(
            y0=trade_setup.entry_zone_low,
            y1=trade_setup.entry_zone_high,
            fillcolor="rgba(33,150,243,0.2)",
            line=dict(color="rgba(33,150,243,0.8)", width=2),
            row=1, col=1,
        )
        fig.add_hline(
            y=trade_setup.stop_loss,
            line=dict(color="red", width=2, dash="dash"),
            row=1, col=1,
        )
        fig.add_hline(
            y=trade_setup.target_1,
            line=dict(color="green", width=2, dash="dash"),
            row=1, col=1,
        )

    # RSI subplot
    if 'rsi' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['rsi'],
                mode='lines', name='RSI',
                line=dict(color='#9c27b0', width=1),
            ),
            row=2, col=1
        )
        fig.add_hline(y=70, line=dict(color='rgba(244,67,54,0.5)', dash='dash', width=1), row=2, col=1)
        fig.add_hline(y=30, line=dict(color='rgba(76,175,80,0.5)', dash='dash', width=1), row=2, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(128,128,128,0.1)", line_width=0, row=2, col=1)

    # MACD subplot
    if 'macd' in data.columns and 'macd_signal' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['macd'],
                mode='lines', name='MACD',
                line=dict(color='#2196f3', width=1),
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['macd_signal'],
                mode='lines', name='Signal',
                line=dict(color='#ff9800', width=1),
            ),
            row=3, col=1
        )

        if 'macd_histogram' in data.columns:
            colors = ['#26a69a' if v >= 0 else '#ef5350' for v in data['macd_histogram']]
            fig.add_trace(
                go.Bar(
                    x=data.index, y=data['macd_histogram'],
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.5,
                ),
                row=3, col=1
            )

        fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', width=1), row=3, col=1)

    # Volatility subplot
    if vol_row and 'rv_1m' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['rv_1m'],
                mode='lines',
                name='1M Vol',
                line=dict(color='#9c27b0', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(156, 39, 176, 0.2)',
            ),
            row=vol_row, col=1
        )

        # Add mean line
        mean_vol = data['rv_1m'].mean()
        fig.add_hline(
            y=mean_vol,
            line=dict(color='rgba(255,255,255,0.5)', dash='dot', width=1),
            row=vol_row, col=1
        )

    # Layout
    height = 900 if vol_row else 800
    fig.update_layout(
        height=height,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=120, t=60, b=40),
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    if vol_row:
        fig.update_yaxes(title_text="Vol %", row=vol_row, col=1)

    return fig
