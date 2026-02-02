# Crypto/FX Summary Page Plan

## Overview

A multi-pair summary page that aggregates key statistics across all tracked currency pairs (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCNH), providing a bird's-eye view of market conditions, trading opportunities, and regime health.

---

## Page Structure

### Section 1: Market Health Dashboard (Top Row)
**Purpose:** Quick visual KPIs for overall market conditions

| Metric | Description | Visual |
|--------|-------------|--------|
| **Trending Pairs** | Count of pairs with ADX > 25 | Large number with trend icon |
| **Bullish Alignment** | % of pairs with aligned bullish MTF | Green gauge |
| **Bearish Alignment** | % of pairs with aligned bearish MTF | Red gauge |
| **Vol Alert Count** | Pairs in HIGH vol or SQUEEZE | Warning badge |

**Layout:** 4 columns using `st.columns(4)` with `st.metric()` components

---

### Section 2: Pair Comparison Grid
**Purpose:** Side-by-side comparison of all pairs with key metrics

| Pair | Price | ADX | Regime | MTF Align | Vol State | Bias | Confluence |
|------|-------|-----|--------|-----------|-----------|------|------------|
| EURUSD | 1.0845 | 24.5 | Transition | Mixed | Normal | Neutral | 4.2 |
| GBPUSD | 1.2650 | 32.1 | Trend | Bullish | Normal | Long | 7.8 |
| USDJPY | 149.50 | 18.2 | Range | Mixed | Low | Neutral | 3.1 |
| AUDUSD | 0.6520 | 28.7 | Trend | Bearish | High | Short | 6.5 |
| USDCNH | 7.2450 | 21.3 | Transition | Mixed | Normal | Neutral | 4.8 |

**Key Columns:**
1. **Pair** - Symbol name
2. **Price** - Current close price
3. **ADX** - Current ADX value (color-coded: <20 gray, 20-25 yellow, >25 green)
4. **Regime** - ADXRegime enum (Range/Transition/Trend with icon)
5. **MTF Align** - Multi-timeframe alignment status
6. **Vol State** - Volatility regime (HIGH/NORMAL/LOW/SQUEEZE)
7. **Bias** - Overall directional bias from signals
8. **Confluence** - Average confluence score (0-10)

**Implementation:** `st.dataframe()` with column styling or custom HTML table

---

### Section 3: Regime Distribution
**Purpose:** Visual breakdown of how many pairs are in each trading regime

```
ADX Regime Distribution
================================
Range Mean Reversion  ████████░░  2 pairs (40%)
Transition Breakout   ████░░░░░░  2 pairs (40%)
Trend Continuation    ██░░░░░░░░  1 pair  (20%)
```

**Statistics to Calculate:**
- Count per regime type
- Percentage distribution
- Average ADX per regime group
- Pairs trending UP vs DOWN vs RANGING

**Visual:** Horizontal bar chart or pie chart using Plotly

---

### Section 4: Signal Momentum Summary
**Purpose:** Aggregate directional signals across all pairs

#### 4a. Net Signal Bias
| Category | Count | Pairs |
|----------|-------|-------|
| **Strong Bullish** | 3 signals | GBPUSD, AUDUSD |
| **Moderate Bullish** | 5 signals | EURUSD, GBPUSD |
| **Neutral** | 8 signals | USDJPY, USDCNH |
| **Moderate Bearish** | 2 signals | AUDUSD |
| **Strong Bearish** | 1 signal | USDCNH |

#### 4b. Signal Type Breakdown
- **Trend Signals** (MACD, SMA alignment): X bullish / Y bearish
- **Momentum Signals** (RSI, Stochastic): X bullish / Y bearish
- **Advanced Signals** (Pullback, Histogram): X bullish / Y bearish

**Visual:** Stacked bar chart showing bullish vs bearish per pair

---

### Section 5: Volatility Overview
**Purpose:** Compare volatility conditions across pairs

| Pair | RV 1M | RV Percentile | Vol Regime | BB Squeeze | Vol Trend |
|------|-------|---------------|------------|------------|-----------|
| EURUSD | 8.2% | 45 | Normal | No | Flat |
| GBPUSD | 11.5% | 78 | High | No | Rising |
| USDJPY | 6.1% | 22 | Low | Yes | Falling |
| AUDUSD | 14.2% | 85 | High | No | Rising |
| USDCNH | 7.8% | 55 | Normal | No | Flat |

**Aggregate Metrics:**
- **Average RV:** Mean realized vol across all pairs
- **Squeeze Count:** Number of pairs in Bollinger squeeze
- **High Vol Count:** Pairs with vol in top quartile
- **Vol Expansion Alert:** Pairs likely to see vol increase

**Visual:** Heatmap or radar chart comparing vol profiles

---

### Section 6: Trade Opportunity Board
**Purpose:** Highlight valid trade setups ranked by quality

| Rank | Pair | Bias | Entry Zone | Stop | Target 1 | R:R | Confluence | Reasoning |
|------|------|------|------------|------|----------|-----|------------|-----------|
| 1 | GBPUSD | LONG | 1.2630-1.2650 | 1.2580 | 1.2720 | 2.1 | 7.8 | MTF bullish, near support |
| 2 | AUDUSD | SHORT | 0.6540-0.6560 | 0.6610 | 0.6470 | 1.8 | 6.5 | Trend down, near resistance |

**Filters Applied:**
- Only show setups with R:R >= 1.5
- Sorted by confluence score descending
- Max 5 setups displayed

**Visual:** Expandable cards with full setup details

---

### Section 7: Multi-Timeframe Alignment Matrix
**Purpose:** Show MTF context for each pair across Weekly/Daily/4H

```
              Weekly    Daily     4H
EURUSD        Bearish   Neutral   Bullish   (Mixed)
GBPUSD        Bullish   Bullish   Bullish   (Aligned Bullish)
USDJPY        Neutral   Bearish   Bearish   (Partially Aligned)
AUDUSD        Bearish   Bearish   Bearish   (Aligned Bearish)
USDCNH        Neutral   Neutral   Bullish   (Mixed)
```

**Color Coding:**
- Green: Bullish
- Red: Bearish
- Gray: Neutral/Ranging

**Visual:** Heatmap grid using Plotly or styled DataFrame

---

### Section 8: S/R Proximity Alerts
**Purpose:** Highlight pairs approaching key support/resistance levels

| Pair | Nearest Level | Type | Distance | Strength | Phase |
|------|---------------|------|----------|----------|-------|
| GBPUSD | 1.2600 | Support | 50 pips | Strong | Established |
| USDJPY | 150.00 | Resistance | 50 pips | Very Strong | Discovery |
| AUDUSD | 0.6500 | Support | 20 pips | Moderate | Depletion |

**Filters:**
- Only show levels within X% of current price (configurable)
- Sort by distance ascending (closest first)
- Include liquidity phase for professional context

**Visual:** Table with color-coded distance warnings

---

### Section 9: Regime Health Index (Composite Score)
**Purpose:** Single score (1-10) indicating overall market tradability

```
Market Health Index: 6.8 / 10
================================
[████████████████░░░░░░░░░░░░░░] 68%

Components:
- Trending Strength (ADX avg): 7.2 / 10
- Signal Agreement: 6.5 / 10
- MTF Alignment: 7.0 / 10
- Volatility Stability: 6.5 / 10
```

**Calculation:**
```python
health_score = (
    0.30 * trending_strength +    # ADX-based
    0.25 * signal_agreement +     # Bullish/bearish consensus
    0.25 * mtf_alignment +        # Timeframe alignment %
    0.20 * volatility_stability   # Inverse of extreme vol
)
```

**Visual:** Large gauge or progress bar with component breakdown

---

### Section 10: Alerts & Notifications
**Purpose:** Surface important market events and warnings

**Alert Types:**
1. **Regime Change Alert:** "EURUSD changed from Range to Transition (2 bars ago)"
2. **Compression Breakout:** "USDJPY showing ADX compression breakout pattern"
3. **Vol Spike Alert:** "AUDUSD volatility expanded 25% in 5 days"
4. **S/R Break Alert:** "GBPUSD broke above 1.2700 resistance"
5. **Squeeze Alert:** "USDJPY in Bollinger squeeze - breakout imminent"

**Visual:** Dismissible alert boxes using `st.warning()`, `st.info()`, `st.error()`

---

## Data Sources & Calculations

### Required Data Per Pair
From existing modules, we need to call:

```python
from core.data_fetcher import fetch_forex_data
from core.indicators import calculate_all_indicators
from core.signals import SignalGenerator
from core.regime import (
    calculate_regime,
    calculate_adx_regime_analysis,
    calculate_mtf_context
)
from core.volatility import calculate_volatility_metrics
from core.support_resistance import detect_sr_zones
from core.trade_setup import generate_trade_setup
```

### Aggregation Functions Needed

```python
def aggregate_regime_distribution(pair_data: Dict) -> Dict:
    """Count pairs in each ADX regime"""
    pass

def calculate_signal_momentum(pair_data: Dict) -> Dict:
    """Sum bullish/bearish signals across pairs"""
    pass

def calculate_market_health_index(pair_data: Dict) -> float:
    """Composite score from all metrics"""
    pass

def get_trade_opportunities(pair_data: Dict) -> List[TradeSetup]:
    """Filter and rank valid setups"""
    pass

def get_sr_proximity_alerts(pair_data: Dict, threshold_pct: float) -> List:
    """Find pairs near key S/R levels"""
    pass

def detect_regime_change_alerts(pair_data: Dict) -> List[str]:
    """Identify recent regime changes"""
    pass
```

---

## Implementation Approach

### File Structure
```
pages/
  3_Summary.py          # New summary page
core/
  summary_aggregator.py # New module for cross-pair calculations
```

### Key Functions in summary_aggregator.py

```python
@dataclass
class PairSummary:
    symbol: str
    price: float
    adx: float
    adx_regime: ADXRegime
    trend_direction: TrendDirection
    mtf_alignment: str
    vol_regime: VolRegime
    rv_1m: float
    signal_counts: Dict[str, int]  # bullish, bearish, neutral
    confluence_score: float
    trade_setup: Optional[TradeSetup]
    nearest_sr: Optional[Dict]

@dataclass
class MarketSummary:
    pairs: List[PairSummary]
    regime_distribution: Dict[ADXRegime, int]
    signal_momentum: Dict[str, int]
    health_index: float
    alerts: List[str]
    timestamp: datetime

def generate_market_summary(symbols: List[str]) -> MarketSummary:
    """Main entry point - generates complete summary"""
    pass
```

---

## UI/UX Considerations

### Sidebar Controls
- **Refresh Button:** Manually refresh all data
- **Auto-refresh Toggle:** Refresh every X minutes
- **Alert Threshold:** Configure S/R proximity threshold
- **Show/Hide Sections:** Toggle visibility of each section

### Color Scheme
- **Bullish:** Green (#00C853)
- **Bearish:** Red (#FF1744)
- **Neutral:** Gray (#9E9E9E)
- **Warning:** Orange (#FF9100)
- **Trending:** Blue (#2979FF)
- **Ranging:** Yellow (#FFD600)

### Responsiveness
- Use `st.columns()` with appropriate ratios
- Collapsible sections with `st.expander()` for dense info
- Tabs for related but separate views

---

## Priority Implementation Order

### Phase 1: Core Framework
1. Create `summary_aggregator.py` with basic data collection
2. Create `3_Summary.py` page skeleton
3. Implement Pair Comparison Grid (Section 2)
4. Implement Market Health Dashboard (Section 1)

### Phase 2: Analysis Views
5. Implement Regime Distribution (Section 3)
6. Implement Signal Momentum Summary (Section 4)
7. Implement Volatility Overview (Section 5)

### Phase 3: Trading Features
8. Implement Trade Opportunity Board (Section 6)
9. Implement MTF Alignment Matrix (Section 7)
10. Implement S/R Proximity Alerts (Section 8)

### Phase 4: Advanced Features
11. Implement Regime Health Index (Section 9)
12. Implement Alerts & Notifications (Section 10)
13. Add caching and performance optimization

---

## Performance Considerations

### Caching Strategy
```python
@st.cache_data(ttl=300)  # 5-minute cache
def fetch_all_pair_data(symbols: List[str]) -> Dict:
    """Fetch and cache data for all pairs"""
    pass
```

### Parallel Data Fetching
- Use `concurrent.futures` for parallel API calls
- Batch indicator calculations where possible

### Lazy Loading
- Load detailed data only when sections are expanded
- Use placeholders during data load

---

## Summary Statistics Reference

### Per-Pair Metrics Available
| Metric | Source | Type |
|--------|--------|------|
| ADX Value | `indicators.py` | float |
| ADX Regime | `regime.py` | ADXRegime enum |
| +DI, -DI | `indicators.py` | float |
| Trend Direction | `regime.py` | TrendDirection enum |
| Vol Regime | `regime.py` | VolatilityState enum |
| RV 1M, 3M, 1Y | `volatility.py` | float (%) |
| RV Percentile | `volatility.py` | float (0-100) |
| BB Squeeze | `volatility.py` | bool |
| Vol Trend | `volatility.py` | str |
| MTF Alignment | `regime.py` | str |
| Alignment Score | `regime.py` | float (0-1) |
| Signal Count | `signals.py` | int |
| Confluence Score | `signals.py` | float (0-10) |
| Trade Setup | `trade_setup.py` | TradeSetup |
| S/R Levels | `support_resistance.py` | List[PriceZone] |

### Aggregate Metrics to Calculate
| Metric | Calculation |
|--------|-------------|
| % Trending | count(ADX > 25) / total_pairs |
| % Bullish Aligned | count(MTF == aligned_bullish) / total_pairs |
| % Bearish Aligned | count(MTF == aligned_bearish) / total_pairs |
| Avg ADX | mean(all ADX values) |
| Avg Confluence | mean(all confluence scores) |
| Net Signal Bias | sum(bullish) - sum(bearish) |
| Avg RV | mean(all RV 1M values) |
| High Vol Count | count(vol_regime == HIGH) |
| Squeeze Count | count(is_squeeze == True) |
| Valid Setup Count | count(setups with R:R >= 1.5) |
| Health Index | weighted composite score |
