"""Plotly chart creation for FX Trading Dashboard."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional

from core.support_resistance import SRResult, ZoneType
from core.trade_setup import TradeSetup, Bias
from core.implied_vol import IV_TENORS, TENOR_DAYS


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
    # Check if ADX data is available
    has_adx = all(col in data.columns for col in ['adx', 'plus_di', 'minus_di'])

    if has_adx:
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.45, 0.15, 0.15, 0.25],
            subplot_titles=(title, 'RSI', 'MACD', 'ADX Trend Strength')
        )
    else:
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

    # ADX subplot (row 4)
    if has_adx:
        # ADX line (purple, main indicator)
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['adx'],
                mode='lines', name='ADX',
                line=dict(color='#9c27b0', width=2),
            ),
            row=4, col=1
        )

        # +DI line (green, bullish directional strength)
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['plus_di'],
                mode='lines', name='+DI',
                line=dict(color='#26a69a', width=1.5),
            ),
            row=4, col=1
        )

        # -DI line (red, bearish directional strength)
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['minus_di'],
                mode='lines', name='-DI',
                line=dict(color='#ef5350', width=1.5),
            ),
            row=4, col=1
        )

        # Threshold levels with zone shading
        # Range zone (0-20): light blue/gray - mean reversion favored
        fig.add_hrect(
            y0=0, y1=20,
            fillcolor="rgba(33,150,243,0.1)",
            line_width=0,
            row=4, col=1
        )

        # Transition zone (20-25): light orange
        fig.add_hrect(
            y0=20, y1=25,
            fillcolor="rgba(255,152,0,0.1)",
            line_width=0,
            row=4, col=1
        )

        # Trend zone (25+): light green
        fig.add_hrect(
            y0=25, y1=60,
            fillcolor="rgba(76,175,80,0.05)",
            line_width=0,
            row=4, col=1
        )

        # Threshold lines
        fig.add_hline(y=20, line=dict(color='rgba(33,150,243,0.6)', dash='dash', width=1), row=4, col=1)  # Range threshold
        fig.add_hline(y=25, line=dict(color='rgba(255,152,0,0.8)', dash='dash', width=1), row=4, col=1)   # Transition threshold
        fig.add_hline(y=30, line=dict(color='rgba(76,175,80,0.8)', dash='dash', width=1), row=4, col=1)   # Strong trend threshold

    # Layout
    chart_height = 950 if has_adx else 800
    fig.update_layout(
        height=chart_height,
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

    # Update y-axis for ADX
    if has_adx:
        fig.update_yaxes(range=[0, 60], title_text="ADX", row=4, col=1)

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


def create_iv_chart(
    iv_df: pd.DataFrame,
    selected_tenor: str = '1M',
    title: str = "Implied Volatility"
) -> go.Figure:
    """
    Create implied volatility time series chart for selected tenor.

    Args:
        iv_df: DataFrame with IV columns (IV 1W, IV 1M, etc.)
        selected_tenor: Tenor to highlight ('1W', '1M', '3M', '6M', '1Y')
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Color mapping for tenors
    tenor_colors = {
        '1W': '#ef5350',   # Red - short term
        '1M': '#9c27b0',   # Purple - main focus
        '3M': '#2196f3',   # Blue
        '6M': '#ff9800',   # Orange
        '1Y': '#4caf50',   # Green - long term
    }

    # Plot all tenors, highlighting the selected one
    for tenor in IV_TENORS:
        col_name = f'IV {tenor}'
        if col_name not in iv_df.columns:
            continue

        is_selected = tenor == selected_tenor
        fig.add_trace(
            go.Scatter(
                x=iv_df.index,
                y=iv_df[col_name],
                mode='lines',
                name=tenor,
                line=dict(
                    color=tenor_colors.get(tenor, '#9e9e9e'),
                    width=2.5 if is_selected else 1,
                    dash=None if is_selected else 'dot'
                ),
                opacity=1.0 if is_selected else 0.5,
                fill='tozeroy' if is_selected else None,
                fillcolor=f"rgba{tuple(list(int(tenor_colors.get(tenor, '#9e9e9e').lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}" if is_selected else None,
            )
        )

    # Add mean line for selected tenor
    selected_col = f'IV {selected_tenor}'
    if selected_col in iv_df.columns:
        mean_iv = iv_df[selected_col].mean()
        fig.add_hline(
            y=mean_iv,
            line=dict(color='rgba(255,255,255,0.5)', dash='dot', width=1),
            annotation_text=f"Mean: {mean_iv:.1f}%",
            annotation_position="right"
        )

    fig.update_layout(
        title=title,
        height=400,
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
        yaxis_title="Implied Volatility %",
        xaxis_title="Date",
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

    return fig


def create_iv_rv_comparison_chart(
    data: pd.DataFrame,
    iv_tenor: str = '1M',
    title: str = "IV vs RV Comparison"
) -> go.Figure:
    """
    Create chart comparing implied vs realized volatility.

    Shows IV for selected tenor, RV (1M), and the spread.

    Args:
        data: DataFrame with IV columns and rv_1m column
        iv_tenor: IV tenor to compare ('1W', '1M', '3M', '6M', '1Y')
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, 'IV-RV Spread (Vol Risk Premium)')
    )

    iv_col = f'iv_{iv_tenor}'

    # IV time series
    if iv_col in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[iv_col],
                mode='lines',
                name=f'IV {iv_tenor}',
                line=dict(color='#9c27b0', width=2),
            ),
            row=1, col=1
        )

    # RV time series
    if 'rv_1m' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['rv_1m'],
                mode='lines',
                name='RV 1M',
                line=dict(color='#2196f3', width=2),
            ),
            row=1, col=1
        )

    # Calculate and plot spread
    if iv_col in data.columns and 'rv_1m' in data.columns:
        spread = data[iv_col] - data['rv_1m']

        # Color bars based on positive/negative spread
        colors = ['#26a69a' if s >= 0 else '#ef5350' for s in spread.fillna(0)]

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=spread,
                name='IV-RV Spread',
                marker_color=colors,
                opacity=0.7,
            ),
            row=2, col=1
        )

        # Zero line
        fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.5)', width=1), row=2, col=1)

        # Add annotations for spread interpretation
        mean_spread = spread.mean()
        fig.add_hline(
            y=mean_spread,
            line=dict(color='rgba(255,255,255,0.3)', dash='dot', width=1),
            annotation_text=f"Mean: {mean_spread:.2f}",
            annotation_position="right",
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
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Volatility %", row=1, col=1)
    fig.update_yaxes(title_text="Spread %", row=2, col=1)

    return fig


def create_term_structure_chart(
    term_structure: dict,
    current_rv: Optional[float] = None,
    title: str = "Vol Term Structure"
) -> go.Figure:
    """
    Create volatility term structure chart (IV curve across tenors).

    Args:
        term_structure: Dict mapping tenor -> IV value
        current_rv: Optional current realized vol for comparison
        title: Chart title

    Returns:
        Plotly Figure object
    """
    # Convert tenors to days for x-axis
    tenors = list(term_structure.keys())
    days = [TENOR_DAYS.get(t, 30) for t in tenors]
    ivs = [term_structure[t] for t in tenors]

    fig = go.Figure()

    # IV term structure line
    fig.add_trace(
        go.Scatter(
            x=days,
            y=ivs,
            mode='lines+markers',
            name='Implied Vol',
            line=dict(color='#9c27b0', width=3),
            marker=dict(size=12, color='#9c27b0'),
        )
    )

    # Add labels for each tenor point
    for i, (tenor, iv) in enumerate(zip(tenors, ivs)):
        fig.add_annotation(
            x=days[i],
            y=iv,
            text=f"{tenor}<br>{iv:.1f}%",
            showarrow=False,
            yshift=25,
            font=dict(size=10, color='white'),
        )

    # Add RV reference line if provided
    if current_rv is not None:
        fig.add_hline(
            y=current_rv,
            line=dict(color='#2196f3', width=2, dash='dash'),
            annotation_text=f"RV 1M: {current_rv:.1f}%",
            annotation_position="right"
        )

    # Determine if curve is inverted
    is_inverted = ivs[0] > ivs[-1] if len(ivs) >= 2 else False

    # Add shading between short and long term
    if len(ivs) >= 2:
        fill_color = 'rgba(239,83,80,0.1)' if is_inverted else 'rgba(76,175,80,0.1)'
        fig.add_trace(
            go.Scatter(
                x=days,
                y=ivs,
                mode='none',
                fill='tozeroy',
                fillcolor=fill_color,
                showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(
            text=title + (" (INVERTED)" if is_inverted else " (Normal)"),
            font=dict(color='#ef5350' if is_inverted else 'white')
        ),
        height=300,
        template='plotly_dark',
        showlegend=True,
        margin=dict(l=60, r=60, t=60, b=40),
        xaxis_title="Days to Expiry",
        yaxis_title="Implied Volatility %",
        xaxis=dict(
            tickmode='array',
            tickvals=days,
            ticktext=tenors,
        ),
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

    return fig


def create_iv_rv_overlay_chart(
    data: pd.DataFrame,
    iv_tenor: str = '1M',
    title: str = "Price with IV/RV Overlay"
) -> go.Figure:
    """
    Create price chart with IV and RV overlaid on secondary axis.

    Args:
        data: DataFrame with OHLC, IV, and RV columns
        iv_tenor: IV tenor to display
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=(title, f'IV {iv_tenor} vs RV 1M'),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )

    # Price candlestick
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

    iv_col = f'iv_{iv_tenor}'

    # IV on vol subplot
    if iv_col in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[iv_col],
                mode='lines',
                name=f'IV {iv_tenor}',
                line=dict(color='#9c27b0', width=2),
                fill='tozeroy',
                fillcolor='rgba(156, 39, 176, 0.1)',
            ),
            row=2, col=1
        )

    # RV on same subplot
    if 'rv_1m' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['rv_1m'],
                mode='lines',
                name='RV 1M',
                line=dict(color='#2196f3', width=2),
            ),
            row=2, col=1
        )

    fig.update_layout(
        height=600,
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
        margin=dict(l=60, r=60, t=60, b=40),
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Volatility %", row=2, col=1)

    return fig


def create_spot_iv_rv_aligned_chart(
    data: pd.DataFrame,
    iv_tenor: str = '1M',
    title: str = "Spot Price with Volatility"
) -> go.Figure:
    """
    Create aligned chart showing spot price with IV and RV subplots.

    Three-panel chart with shared x-axis:
    1. Spot price candlestick
    2. Implied volatility for selected tenor
    3. Realized volatility (1M)

    Args:
        data: DataFrame with OHLC, IV, and RV columns
        iv_tenor: IV tenor to display
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(title, f'Implied Vol ({iv_tenor})', 'Realized Vol (1M)')
    )

    # Panel 1: Spot price candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Spot',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1
    )

    # Add SMAs to spot chart
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

    iv_col = f'iv_{iv_tenor}'

    # Panel 2: Implied Volatility
    if iv_col in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[iv_col],
                mode='lines',
                name=f'IV {iv_tenor}',
                line=dict(color='#9c27b0', width=2),
                fill='tozeroy',
                fillcolor='rgba(156, 39, 176, 0.15)',
            ),
            row=2, col=1
        )

        # Add mean line
        mean_iv = data[iv_col].mean()
        fig.add_hline(
            y=mean_iv,
            line=dict(color='rgba(255,255,255,0.4)', dash='dot', width=1),
            row=2, col=1
        )

    # Panel 3: Realized Volatility
    if 'rv_1m' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['rv_1m'],
                mode='lines',
                name='RV 1M',
                line=dict(color='#2196f3', width=2),
                fill='tozeroy',
                fillcolor='rgba(33, 150, 243, 0.15)',
            ),
            row=3, col=1
        )

        # Add mean line
        mean_rv = data['rv_1m'].mean()
        fig.add_hline(
            y=mean_rv,
            line=dict(color='rgba(255,255,255,0.4)', dash='dot', width=1),
            row=3, col=1
        )

    fig.update_layout(
        height=700,
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
        margin=dict(l=60, r=60, t=60, b=40),
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="IV %", row=2, col=1)
    fig.update_yaxes(title_text="RV %", row=3, col=1)

    return fig


def create_vol_with_ta_events_chart(
    data: pd.DataFrame,
    ta_events: list,
    iv_tenor: str = '1M',
    title: str = "Volatility with TA Events"
) -> go.Figure:
    """
    Create IV/RV chart with markers at TA signal event points.

    Shows how volatility behaved around technical analysis events like
    support breaks, resistance tests, MACD crossovers, etc.

    Args:
        data: DataFrame with IV and RV columns
        ta_events: List of dicts with 'date', 'type', 'direction', 'description'
        iv_tenor: IV tenor to display
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=(title, 'IV-RV Spread at Events')
    )

    iv_col = f'iv_{iv_tenor}'

    # IV time series
    if iv_col in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[iv_col],
                mode='lines',
                name=f'IV {iv_tenor}',
                line=dict(color='#9c27b0', width=2),
            ),
            row=1, col=1
        )

    # RV time series
    if 'rv_1m' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['rv_1m'],
                mode='lines',
                name='RV 1M',
                line=dict(color='#2196f3', width=2),
            ),
            row=1, col=1
        )

    # Add TA event markers
    bullish_events = [e for e in ta_events if e.get('direction') == 'bullish']
    bearish_events = [e for e in ta_events if e.get('direction') == 'bearish']
    neutral_events = [e for e in ta_events if e.get('direction') not in ['bullish', 'bearish']]

    # Get IV values at event dates for marker placement
    def get_iv_at_date(date):
        if iv_col in data.columns and date in data.index:
            return data.loc[date, iv_col]
        return None

    # Bullish events (green triangles pointing up)
    if bullish_events:
        dates = [e['date'] for e in bullish_events if e['date'] in data.index]
        ivs = [get_iv_at_date(d) for d in dates]
        texts = [e.get('description', e.get('type', '')) for e in bullish_events if e['date'] in data.index]

        if dates and any(iv is not None for iv in ivs):
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=ivs,
                    mode='markers',
                    name='Bullish Signal',
                    marker=dict(
                        symbol='triangle-up',
                        size=14,
                        color='#26a69a',
                        line=dict(color='white', width=1)
                    ),
                    text=texts,
                    hovertemplate='%{text}<br>IV: %{y:.2f}%<extra></extra>',
                ),
                row=1, col=1
            )

    # Bearish events (red triangles pointing down)
    if bearish_events:
        dates = [e['date'] for e in bearish_events if e['date'] in data.index]
        ivs = [get_iv_at_date(d) for d in dates]
        texts = [e.get('description', e.get('type', '')) for e in bearish_events if e['date'] in data.index]

        if dates and any(iv is not None for iv in ivs):
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=ivs,
                    mode='markers',
                    name='Bearish Signal',
                    marker=dict(
                        symbol='triangle-down',
                        size=14,
                        color='#ef5350',
                        line=dict(color='white', width=1)
                    ),
                    text=texts,
                    hovertemplate='%{text}<br>IV: %{y:.2f}%<extra></extra>',
                ),
                row=1, col=1
            )

    # Neutral events (yellow circles)
    if neutral_events:
        dates = [e['date'] for e in neutral_events if e['date'] in data.index]
        ivs = [get_iv_at_date(d) for d in dates]
        texts = [e.get('description', e.get('type', '')) for e in neutral_events if e['date'] in data.index]

        if dates and any(iv is not None for iv in ivs):
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=ivs,
                    mode='markers',
                    name='Neutral Signal',
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color='#ffeb3b',
                        line=dict(color='white', width=1)
                    ),
                    text=texts,
                    hovertemplate='%{text}<br>IV: %{y:.2f}%<extra></extra>',
                ),
                row=1, col=1
            )

    # IV-RV Spread with event markers
    if iv_col in data.columns and 'rv_1m' in data.columns:
        spread = data[iv_col] - data['rv_1m']
        colors = ['#26a69a' if s >= 0 else '#ef5350' for s in spread.fillna(0)]

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=spread,
                name='IV-RV Spread',
                marker_color=colors,
                opacity=0.6,
            ),
            row=2, col=1
        )

        fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.5)', width=1), row=2, col=1)

        # Add vertical lines at event dates
        for event in ta_events:
            if event['date'] in data.index:
                color = '#26a69a' if event.get('direction') == 'bullish' else \
                        '#ef5350' if event.get('direction') == 'bearish' else '#ffeb3b'
                fig.add_vline(
                    x=event['date'],
                    line=dict(color=color, width=1, dash='dot'),
                    row=2, col=1
                )

    fig.update_layout(
        height=550,
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
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Volatility %", row=1, col=1)
    fig.update_yaxes(title_text="Spread %", row=2, col=1)

    return fig
