# FX Trading Dashboard 📈

A professional technical analysis dashboard for forex trading, built with Streamlit. This tool provides comprehensive multi-timeframe analysis, support/resistance detection, market regime identification, and automated trade setup generation.

## Vol Regime Engine (new)

`vol_engine/` is a standalone, data-source-agnostic vol analytics package for FX
options work — regime state machine (range / compression / break / trend with
acceptance and failure logic), OHLC realized-vol estimators (Parkinson,
Garman-Klass, Yang-Zhang), vol cones, an event-study harness that measures each
signal's conditional forward-RV distribution, HAR-RV forecasting with walk-forward
evaluation, an HMM cross-check, and a state→options-structure playbook.

- Design and methodology: `VOL_ENGINE_PLAN.md`
- Dashboard surface: `pages/4_Vol_Regime.py`
- Headless use (work PC, no UI): `python run_vol_analysis.py --csv <ohlc.csv>` or `--demo`
- Tests: `python -m pytest tests/` (see `requirements-dev.txt`)

The package imports neither Streamlit nor `ta`; input is any DataFrame with
`Date`-indexed `Open, High, Low, Close` — see `vol_engine/data_io.py` for the CSV
schema to export from a real data source.

## Features

### 🎯 Core Capabilities

- **Multi-Timeframe Analysis**: Weekly, Daily, and 4-Hour timeframe context
- **Support & Resistance Detection**: Advanced multi-method zone identification
- **Market Regime Detection**: Identifies trending vs. ranging markets
- **Technical Indicators**: Comprehensive indicator suite organized by category
- **Trade Setup Generation**: Automated entry, stop loss, and target calculations
- **Interactive Charts**: Plotly-powered visualizations with S/R zones

### 📊 Technical Indicators

The dashboard calculates and categorizes indicators into different groups:

#### Trend Indicators
- **SMA 20, 50, 200**: Simple Moving Averages
- **MACD**: Moving Average Convergence Divergence (with signal and histogram)
- **ADX**: Average Directional Index (trend strength)
- **+DI / -DI**: Directional Indicators

#### Momentum Indicators
- **RSI**: Relative Strength Index (14-period)
- **Stochastic Oscillator**: %K and %D lines

#### Volatility Indicators
- **Bollinger Bands**: Upper, Middle, Lower bands with width
- **ATR**: Average True Range

### 🔍 Support & Resistance Detection

The system uses a sophisticated multi-method approach:

1. **Swing High/Low Detection**: Identifies local extrema using scipy
2. **Volume-Weighted Levels**: Volume Profile analysis for high-volume zones
3. **Bollinger Band Touches**: Dynamic S/R from band interactions
4. **Round Number Levels**: Psychological price levels (e.g., 1.0800, 1.0900)
5. **DBSCAN Clustering**: Merges nearby levels into zones
6. **Confluence Scoring**: Strength calculation based on multiple confirmations

### 💱 Supported Currency Pairs

- EUR/USD
- GBP/USD
- USD/JPY
- AUD/USD
- USD/CNH

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Technicals_dashboard
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Dashboard

```bash
streamlit run main.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

### Using the Dashboard

1. **Select Currency Pair**: Use the sidebar to choose from available pairs
2. **View Key Metrics**: 
   - Current price
   - Market regime (trending/ranging)
   - Multi-timeframe alignment
   - Trading bias (LONG/SHORT/NEUTRAL)
3. **Analyze Price Levels**: Review support and resistance zones with strength scores
4. **Review Trade Setup**: If a valid setup exists, view entry zones, stop loss, targets, and R:R ratios
5. **Explore Charts**: Interactive multi-timeframe charts with S/R zones highlighted

## Project Structure

```
Technicals_dashboard/
├── main.py                 # Streamlit app entry point
├── config.py              # Configuration and constants
├── requirements.txt        # Python dependencies
├── core/
│   ├── data_fetcher.py    # Market data fetching (yfinance)
│   ├── indicators.py      # Technical indicator calculations
│   ├── support_resistance.py  # S/R zone detection
│   ├── regime.py          # Market regime detection
│   ├── trade_setup.py     # Trade setup generation
│   └── signals.py         # Signal generation
├── visualization/
│   └── charts.py          # Plotly chart creation
├── utils/
│   └── forex_utils.py     # Forex-specific utilities
└── data/                  # Cached market data (optional)
```

## Configuration

Key configuration options in `config.py`:

- **SR_CONFIG**: Support/Resistance detection parameters
- **REGIME_CONFIG**: Market regime detection thresholds
- **CONFLUENCE_WEIGHTS**: Strength scoring weights
- **TRADE_CONFIG**: Trade setup generation parameters

## Technical Details

### Data Source
- Uses `yfinance` for real-time and historical forex data
- Supports multiple timeframes with automatic resampling

### Support/Resistance Algorithm
- Combines 4 detection methods
- Uses DBSCAN clustering to merge nearby levels
- Calculates confluence-based strength scores
- Classifies zones as support (below price) or resistance (above price)

### Market Regime Detection
- Analyzes ADX, ATR, and moving average relationships
- Classifies as: Strong Uptrend, Weak Uptrend, Ranging, Weak Downtrend, Strong Downtrend

### Trade Setup Generation
- Considers S/R zones, regime, and multi-timeframe alignment
- Calculates entry zones with ATR-based buffers
- Sets stop loss at zone invalidation
- Identifies targets at next S/R levels
- Only generates setups with minimum 1.5:1 R:R ratio

## Dependencies

- `streamlit>=1.42.0` - Web framework
- `yfinance>=0.2.53` - Market data
- `ta>=0.11.0` - Technical analysis library
- `pandas>=2.2.3` - Data manipulation
- `numpy>=1.26.4` - Numerical operations
- `scipy>=1.15.1` - Scientific computing (for extrema detection)
- `plotly>=5.18.0` - Interactive charts
- `scikit-learn>=1.3.0` - Machine learning (DBSCAN clustering)

## License

This project is provided as-is for educational and research purposes.

## Disclaimer

This tool is for educational purposes only. Trading forex involves substantial risk of loss. Past performance does not guarantee future results. Always conduct your own research and consider consulting with a financial advisor before making trading decisions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Created for professional forex technical analysis.
