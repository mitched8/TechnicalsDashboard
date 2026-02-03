"""FX Market Summary Page - Multi-pair overview dashboard."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from config import CURRENCY_PAIRS, get_default_settings
from core.summary_aggregator import (
    generate_market_summary, pairs_to_dataframe, apply_filters,
    PairSummary, MarketSummary
)
from core.regime import ADXRegime
from core.volatility import VolRegime

st.set_page_config(
    page_title="FX Summary",
    page_icon="📊",
    layout="wide"
)

st.title("📊 FX Market Summary")
st.caption("Multi-pair overview of market conditions, regimes, and opportunities")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("Settings")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    
    # Section visibility toggles
    st.subheader("Show/Hide Sections")
    show_health = st.checkbox("Market Health Dashboard", value=True)
    show_grid = st.checkbox("Pair Comparison Grid", value=True)
    show_regime = st.checkbox("Regime Distribution", value=True)
    show_signals = st.checkbox("Signal Momentum", value=True)
    show_vol = st.checkbox("Volatility Overview", value=True)
    show_setups = st.checkbox("Trade Opportunities", value=True)
    show_mtf = st.checkbox("MTF Alignment Matrix", value=True)
    show_sr = st.checkbox("S/R Proximity Alerts", value=True)
    show_alerts = st.checkbox("Alerts & Notifications", value=True)

# ========== LOAD DATA ==========
with st.spinner("Loading market data for all pairs..."):
    summary = generate_market_summary()

if not summary.pairs:
    st.error("Could not load data for any currency pairs. Please check your connection.")
    st.stop()

st.caption(f"Last updated: {summary.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

# ========== SECTION 1: MARKET HEALTH DASHBOARD ==========
if show_health:
    st.header("Market Health Dashboard")
    
    health_col1, health_col2, health_col3, health_col4 = st.columns(4)
    
    with health_col1:
        st.metric(
            label="📈 Trending Pairs",
            value=f"{summary.trending_count} / {len(summary.pairs)}",
            delta=f"ADX > 25",
            delta_color="off"
        )
    
    with health_col2:
        bullish_pct = (summary.bullish_aligned_count / len(summary.pairs)) * 100
        st.metric(
            label="🟢 Bullish Aligned",
            value=f"{summary.bullish_aligned_count}",
            delta=f"{bullish_pct:.0f}%",
            delta_color="normal"
        )
    
    with health_col3:
        bearish_pct = (summary.bearish_aligned_count / len(summary.pairs)) * 100
        st.metric(
            label="🔴 Bearish Aligned",
            value=f"{summary.bearish_aligned_count}",
            delta=f"{bearish_pct:.0f}%",
            delta_color="inverse"
        )
    
    with health_col4:
        vol_alert = summary.high_vol_count + summary.squeeze_count
        st.metric(
            label="⚠️ Vol Alerts",
            value=vol_alert,
            delta=f"{summary.squeeze_count} squeeze" if summary.squeeze_count > 0 else "normal",
            delta_color="inverse" if vol_alert > 0 else "off"
        )
    
    st.divider()

# ========== SECTION 9: REGIME HEALTH INDEX ==========
if show_health:
    st.subheader(f"Market Health Index: {summary.health_index:.1f} / 10")
    
    # Progress bar
    health_pct = summary.health_index / 10
    if health_pct >= 0.7:
        health_color = "green"
    elif health_pct >= 0.4:
        health_color = "orange"
    else:
        health_color = "red"
    
    st.progress(health_pct)
    
    # Component breakdown
    if summary.health_components:
        comp_cols = st.columns(len(summary.health_components))
        for i, (name, value) in enumerate(summary.health_components.items()):
            with comp_cols[i]:
                st.metric(name, f"{value:.1f}/10")
    
    st.divider()

# ========== SECTION 10: ALERTS & NOTIFICATIONS ==========
if show_alerts and summary.alerts:
    st.header("🔔 Alerts & Notifications")
    
    for alert in summary.alerts:
        if alert.startswith("⚠️"):
            st.warning(alert)
        elif alert.startswith("✅"):
            st.success(alert)
        elif alert.startswith("📈"):
            st.info(alert)
        else:
            st.info(alert)
    
    st.divider()

# ========== SECTION 2: PAIR COMPARISON GRID ==========
if show_grid:
    st.header("Pair Comparison Grid")
    
    # Filters
    with st.expander("🔍 Filters", expanded=False):
        # Row 1: Currency/Pair filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            currency_search = st.text_input(
                "Currency Search",
                placeholder="e.g., USD, EUR, JPY...",
                help="Type a currency code to filter pairs containing that currency"
            )
        with filter_col2:
            pair_filter = st.multiselect(
                "Select Pairs",
                [p.display_name for p in summary.pairs],
                default=[]
            )
        
        st.divider()
        
        # Row 2: Metric filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            regime_filter = st.multiselect("Regime", ["Range", "Transition", "Trend"])
            mtf_filter = st.multiselect("MTF Align", ["Aligned Bullish", "Aligned Bearish", "Mixed"])
        
        with col2:
            vol_filter = st.multiselect("Vol State", ["High", "Normal", "Low", "Squeeze"])
            bias_filter = st.multiselect("Bias", ["LONG", "SHORT", "NEUTRAL"])
        
        with col3:
            min_adx = st.slider("Min ADX", 0, 50, 0)
            min_confluence = st.slider("Min Confluence", 0.0, 10.0, 0.0)
        
        with col4:
            pct_direction = st.radio("1D Change", ["All", "Positive only", "Negative only"], horizontal=True)
            sort_by = st.selectbox("Sort by", ["Confluence", "ADX", "Chg 1D", "Chg 1W", "Chg 1M"])
    
    # Build filters dict
    filters = {
        'currency_search': currency_search,
        'pairs': pair_filter if pair_filter else None,
        'regime': regime_filter if regime_filter else None,
        'mtf_align': mtf_filter if mtf_filter else None,
        'vol_state': vol_filter if vol_filter else None,
        'bias': bias_filter if bias_filter else None,
        'min_adx': min_adx if min_adx > 0 else None,
        'min_confluence': min_confluence if min_confluence > 0 else None,
        'pct_direction': 'positive' if pct_direction == "Positive only" else 'negative' if pct_direction == "Negative only" else None,
    }
    
    # Convert to DataFrame and apply filters
    df = pairs_to_dataframe(summary.pairs)
    df = apply_filters(df, filters)
    
    # Sort
    sort_ascending = sort_by in ["Chg 1D", "Chg 1W", "Chg 1M"]  # Changes: ascending shows negative first
    if sort_by == "Confluence":
        df = df.sort_values("Confluence", ascending=False)
    elif sort_by == "ADX":
        df = df.sort_values("ADX", ascending=False)
    else:
        df = df.sort_values(sort_by, ascending=sort_ascending)
    
    # Style the dataframe
    def style_pct(val):
        if val > 0:
            return f"color: green"
        elif val < 0:
            return f"color: red"
        return ""
    
    def style_adx(val):
        if val > 25:
            return "background-color: rgba(76, 175, 80, 0.3)"
        elif val > 20:
            return "background-color: rgba(255, 193, 7, 0.3)"
        return ""
    
    # Display dataframe
    st.dataframe(
        df.style
        .applymap(style_pct, subset=['Chg 1D', 'Chg 1W', 'Chg 1M'])
        .applymap(style_adx, subset=['ADX'])
        .format({
            'Price': '{:.5f}',
            'Chg 1D': '{:+.2f}%',
            'Chg 1W': '{:+.2f}%',
            'Chg 1M': '{:+.2f}%',
            'ADX': '{:.1f}',
            'Confluence': '{:.1f}',
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()

# ========== SECTION 3: REGIME DISTRIBUTION ==========
if show_regime:
    st.header("ADX Regime Distribution")
    
    regime_col1, regime_col2 = st.columns([2, 1])
    
    with regime_col1:
        # Bar chart
        regime_data = {
            'Regime': ['Range (Mean Reversion)', 'Transition (Breakout)', 'Trend (Continuation)'],
            'Count': [summary.range_count, summary.transition_count, summary.trend_count],
            'Color': ['#2196F3', '#FF9800', '#4CAF50']
        }
        regime_df = pd.DataFrame(regime_data)
        
        fig = px.bar(
            regime_df,
            x='Count',
            y='Regime',
            orientation='h',
            color='Regime',
            color_discrete_map={
                'Range (Mean Reversion)': '#2196F3',
                'Transition (Breakout)': '#FF9800',
                'Trend (Continuation)': '#4CAF50'
            }
        )
        fig.update_layout(
            showlegend=False,
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with regime_col2:
        st.markdown("**Regime Stats**")
        total = len(summary.pairs)
        st.write(f"Range: {summary.range_count} ({summary.range_count/total*100:.0f}%)")
        st.write(f"Transition: {summary.transition_count} ({summary.transition_count/total*100:.0f}%)")
        st.write(f"Trend: {summary.trend_count} ({summary.trend_count/total*100:.0f}%)")
        
        avg_adx = sum(p.adx for p in summary.pairs) / total
        st.write(f"Avg ADX: {avg_adx:.1f}")
    
    st.divider()

# ========== SECTION 4: SIGNAL MOMENTUM SUMMARY ==========
if show_signals:
    st.header("Signal Momentum Summary")
    
    sig_col1, sig_col2, sig_col3, sig_col4 = st.columns(4)
    
    with sig_col1:
        st.metric("🟢 Bullish Signals", summary.total_bullish_signals)
    
    with sig_col2:
        st.metric("🔴 Bearish Signals", summary.total_bearish_signals)
    
    with sig_col3:
        st.metric("⚪ Neutral Signals", summary.total_neutral_signals)
    
    with sig_col4:
        bias_label = "Bullish" if summary.net_signal_bias > 0 else "Bearish" if summary.net_signal_bias < 0 else "Neutral"
        st.metric("Net Bias", f"{bias_label} ({summary.net_signal_bias:+d})")
    
    # Signal breakdown by pair
    st.subheader("Signal Breakdown by Pair")
    
    signal_data = []
    for p in summary.pairs:
        signal_data.append({
            'Pair': p.display_name,
            'Bullish': p.bullish_count,
            'Bearish': -p.bearish_count,  # Negative for stacked bar
            'Net': p.bullish_count - p.bearish_count
        })
    
    signal_df = pd.DataFrame(signal_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Bullish',
        x=signal_df['Pair'],
        y=signal_df['Bullish'],
        marker_color='#4CAF50'
    ))
    fig.add_trace(go.Bar(
        name='Bearish',
        x=signal_df['Pair'],
        y=signal_df['Bearish'],
        marker_color='#F44336'
    ))
    fig.update_layout(
        barmode='relative',
        height=300,
        template='plotly_dark',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()

# ========== SECTION 5: VOLATILITY OVERVIEW ==========
if show_vol:
    st.header("Volatility Overview")
    
    # Aggregate vol metrics
    vol_col1, vol_col2, vol_col3, vol_col4 = st.columns(4)
    
    avg_rv = sum(p.rv_1m for p in summary.pairs) / len(summary.pairs)
    
    with vol_col1:
        st.metric("Avg 1M RV", f"{avg_rv:.1f}%")
    
    with vol_col2:
        st.metric("High Vol Count", summary.high_vol_count)
    
    with vol_col3:
        st.metric("Squeeze Count", summary.squeeze_count)
    
    with vol_col4:
        low_vol = sum(1 for p in summary.pairs if p.vol_regime == VolRegime.LOW)
        st.metric("Low Vol Count", low_vol)
    
    # Vol table
    vol_data = []
    for p in summary.pairs:
        vol_labels = {
            VolRegime.HIGH: "🔴 High",
            VolRegime.NORMAL: "🟡 Normal",
            VolRegime.LOW: "🟢 Low",
            VolRegime.SQUEEZE: "⚡ Squeeze",
        }
        vol_data.append({
            'Pair': p.display_name,
            'RV 1M': f"{p.rv_1m:.1f}%",
            'RV Percentile': f"{p.rv_percentile:.0f}",
            'Vol Regime': vol_labels.get(p.vol_regime, "Normal"),
            'Squeeze': "Yes" if p.is_squeeze else "No",
            'Vol Trend': p.vol_trend.title(),
        })
    
    vol_df = pd.DataFrame(vol_data)
    st.dataframe(vol_df, use_container_width=True, hide_index=True)
    
    st.divider()

# ========== SECTION 6: TRADE OPPORTUNITY BOARD ==========
if show_setups:
    st.header("Trade Opportunity Board")
    
    # Filter pairs with valid setups
    setups = [p for p in summary.pairs if p.has_setup and p.setup_rr >= 1.5]
    setups.sort(key=lambda x: x.confluence_score, reverse=True)
    
    if setups:
        st.caption(f"Showing {len(setups)} valid setups (R:R >= 1.5)")
        
        for i, p in enumerate(setups[:5]):  # Top 5
            with st.expander(f"#{i+1} {p.display_name} - {p.setup_bias} (R:R {p.setup_rr:.1f})", expanded=(i == 0)):
                setup_col1, setup_col2, setup_col3 = st.columns(3)
                
                with setup_col1:
                    st.markdown("**Entry & Risk**")
                    st.write(f"Entry Zone: {p.setup_entry_low:.5f} - {p.setup_entry_high:.5f}")
                    st.write(f"Stop Loss: {p.setup_stop:.5f}")
                
                with setup_col2:
                    st.markdown("**Targets**")
                    st.write(f"Target 1: {p.setup_target1:.5f}")
                    st.write(f"R:R: {p.setup_rr:.1f}")
                
                with setup_col3:
                    st.markdown("**Context**")
                    st.write(f"ADX: {p.adx:.1f} ({p.adx_slope})")
                    st.write(f"MTF: {p.mtf_alignment.replace('_', ' ').title()}")
                    st.write(f"Confluence: {p.confluence_score:.1f}/10")
    else:
        st.info("No valid trade setups at current levels (minimum R:R 1.5)")
    
    st.divider()

# ========== SECTION 7: MTF ALIGNMENT MATRIX ==========
if show_mtf:
    st.header("Multi-Timeframe Alignment Matrix")
    
    # Create heatmap data
    mtf_data = []
    for p in summary.pairs:
        mtf_data.append({
            'Pair': p.display_name,
            'Weekly': p.weekly_trend,
            'Daily': p.daily_trend,
            '4H': p.four_hour_trend,
            'Alignment': p.mtf_alignment.replace('_', ' ').title()
        })
    
    mtf_df = pd.DataFrame(mtf_data)
    
    # Color mapping for heatmap
    def trend_color(val):
        if val == "Bullish":
            return "background-color: rgba(76, 175, 80, 0.5)"
        elif val == "Bearish":
            return "background-color: rgba(244, 67, 54, 0.5)"
        return "background-color: rgba(158, 158, 158, 0.3)"
    
    styled_mtf = mtf_df.style.applymap(
        trend_color,
        subset=['Weekly', 'Daily', '4H']
    )
    
    st.dataframe(styled_mtf, use_container_width=True, hide_index=True)
    
    # Legend
    st.caption("🟢 Bullish | 🔴 Bearish | ⚪ Neutral")
    
    st.divider()

# ========== SECTION 8: S/R PROXIMITY ALERTS ==========
if show_sr:
    st.header("S/R Proximity Alerts")
    
    # Filter pairs near S/R (within 50 pips)
    proximity_threshold = 50
    
    sr_alerts = []
    for p in summary.pairs:
        if p.support_distance_pips > 0 and p.support_distance_pips < proximity_threshold:
            sr_alerts.append({
                'Pair': p.display_name,
                'Level': p.nearest_support,
                'Type': '🟢 Support',
                'Distance (pips)': p.support_distance_pips,
                'ADX': p.adx,
            })
        if p.resistance_distance_pips > 0 and p.resistance_distance_pips < proximity_threshold:
            sr_alerts.append({
                'Pair': p.display_name,
                'Level': p.nearest_resistance,
                'Type': '🔴 Resistance',
                'Distance (pips)': p.resistance_distance_pips,
                'ADX': p.adx,
            })
    
    if sr_alerts:
        sr_alerts.sort(key=lambda x: x['Distance (pips)'])
        sr_df = pd.DataFrame(sr_alerts)
        
        st.dataframe(
            sr_df.style.format({
                'Level': '{:.5f}',
                'Distance (pips)': '{:.0f}',
                'ADX': '{:.1f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.caption(f"Showing levels within {proximity_threshold} pips of current price")
    else:
        st.info(f"No pairs currently within {proximity_threshold} pips of key S/R levels")

# ========== FOOTER ==========
st.divider()
st.caption("Data refreshes every 5 minutes. Click 'Refresh Data' in sidebar for manual refresh.")
