"""Generate sample implied volatility data for testing."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_iv_data():
    """
    Generate realistic FX implied volatility data.

    Creates sample data with:
    - Tenors: 1W, 1M, 3M, 6M, 1Y
    - Currency pairs: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCNH
    - Daily data for ~1 year
    - Realistic vol levels and term structure
    """
    np.random.seed(42)

    # Date range - 1 year of daily data
    end_date = datetime(2024, 1, 15)
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days

    # Currency pair base vol levels (ATM vol approximations)
    base_vols = {
        'EURUSD': 7.5,   # Lower vol, major pair
        'GBPUSD': 9.0,   # Slightly higher
        'USDJPY': 10.5,  # Higher due to BoJ
        'AUDUSD': 11.0,  # Commodity currency
        'USDCNH': 6.5,   # Managed float, lower vol
    }

    # Term structure multipliers (typical upward sloping)
    tenor_multipliers = {
        '1W': 0.95,   # Short-term slightly lower
        '1M': 1.00,   # Base
        '3M': 1.05,   # Slight premium
        '6M': 1.08,   # More premium
        '1Y': 1.12,   # Longest tenor highest
    }

    tenors = ['1W', '1M', '3M', '6M', '1Y']

    all_data = []

    for symbol, base_vol in base_vols.items():
        # Generate a vol regime path (mean-reverting with jumps)
        n_days = len(dates)

        # Base random walk for vol regime
        vol_regime = np.zeros(n_days)
        vol_regime[0] = 0

        for i in range(1, n_days):
            # Mean reversion + random shock + occasional jumps
            mean_reversion = -0.02 * vol_regime[i-1]
            random_shock = np.random.normal(0, 0.15)

            # Occasional vol spike (5% chance)
            if np.random.random() < 0.05:
                random_shock += np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)

            vol_regime[i] = vol_regime[i-1] + mean_reversion + random_shock

        # Clip regime to reasonable range
        vol_regime = np.clip(vol_regime, -3, 5)

        for tenor in tenors:
            multiplier = tenor_multipliers[tenor]

            # Calculate IV for this tenor
            # Base + regime effect + tenor adjustment + daily noise
            iv_values = (
                base_vol +
                vol_regime +
                (multiplier - 1) * base_vol +
                np.random.normal(0, 0.1, n_days)  # Daily noise
            )

            # Ensure positive and reasonable range
            iv_values = np.clip(iv_values, base_vol * 0.5, base_vol * 2.5)

            for i, date in enumerate(dates):
                all_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'tenor': tenor,
                    'iv': round(iv_values[i], 2)
                })

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Pivot to wide format for easier reading
    # Each row is date + symbol, columns are tenor IVs
    df_wide = df.pivot_table(
        index=['date', 'symbol'],
        columns='tenor',
        values='iv'
    ).reset_index()

    # Flatten column names
    df_wide.columns = ['date', 'symbol'] + [f'iv_{t}' for t in tenors]

    return df_wide


if __name__ == '__main__':
    df = generate_iv_data()

    # Save to CSV
    output_path = '/home/user/TechnicalsDashboard/data/fx_implied_vol.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows of IV data")
    print(f"Saved to {output_path}")
    print("\nSample data:")
    print(df.head(10))
    print("\nColumns:", df.columns.tolist())
    print("\nSymbols:", df['symbol'].unique())
    print("\nDate range:", df['date'].min(), "to", df['date'].max())
