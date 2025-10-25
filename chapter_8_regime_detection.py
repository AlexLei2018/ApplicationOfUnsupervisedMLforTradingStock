import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import quant_utils as qtu

def analyze_market_regimes(tickers, start_date, end_date, k_value=3):
    """
    Analyzes market regimes using K-Means clustering and visualizes them
    against the S&P 500 index.

    Args:
        tickers (list): A list of stock tickers to build market features from.
        start_date (str): The start date for historical data (e.g., "YYYY-MM-DD").
        end_date (str): The end date for historical data (e.g., "YYYY-MM-DD").
        k_value (int): The number of clusters (market regimes) to identify.

    Returns:
        pd.DataFrame: A DataFrame containing the market regime features and identified regimes.
    """

    # --- 1. Build market regime feature matrix ---
    print("\n--- 1. Building Market Regime Features ---")
    regime_features = qtu.build_market_regime_features(tickers, start_date, end_date)
    print("Market regime features built successfully.")

    # --- FIX STARTS HERE ---
    # Ensure all column names are strings for scikit-learn compatibility
    if not all(isinstance(col, str) for col in regime_features.columns):
        print("Converting DataFrame column names to string type for scikit-learn compatibility.")
        regime_features.columns = regime_features.columns.astype(str)
    # --- FIX ENDS HERE ---

    # --- 2. Use K-Means to identify market regimes ---
    # get_clusters_snapshot function already includes standardization
    print(f"\n--- 2. Identifying Market Regimes using K-Means (K={k_value}) ---")
    # Make sure qtu.get_clusters_snapshot is correctly defined in your quant_utils.py
    # as discussed in the previous turn.
    regime_labels = qtu.get_clusters_snapshot(regime_features, k=k_value)
    market_regimes_df = regime_features.copy()
    market_regimes_df['Regime'] = regime_labels

    print(f"\n--- Daily Market Regime Labels (K={k_value}) (Preview) ---")
    print(market_regimes_df.tail())

    # --- 3. Interpret and name each regime ---
    print("\n--- 3. Interpreting and Naming Market Regimes ---")
    regime_profiles = market_regimes_df.groupby('Regime').mean()
    print("\n--- Market Regime Characteristics Profile ---")
    print(regime_profiles)

    regime_mapping = {}

    if 'VIX' in regime_profiles.columns:
        bearish_cluster = regime_profiles['VIX'].idxmax()
        regime_mapping[bearish_cluster] = 'Bearish'
    else:
        print("Warning: 'VIX' column not found in regime profiles. Cannot automatically identify 'Bearish' regime.")

    if 'SP500_ROC_63d' in regime_profiles.columns:
        bullish_cluster = regime_profiles['SP500_ROC_63d'].idxmax()
        if bullish_cluster not in regime_mapping:
            regime_mapping[bullish_cluster] = 'Bullish'
        else:
            print("Warning: Bullish cluster is the same as Bearish. Re-evaluating naming logic or increasing K.")
    else:
        print("Warning: 'SP500_ROC_63d' column not found in regime profiles. Cannot automatically identify 'Bullish' regime.")

    identified_clusters = set(regime_mapping.keys())
    all_clusters = set(range(k_value))
    neutral_clusters = list(all_clusters - identified_clusters)

    if neutral_clusters:
        for i, cluster in enumerate(neutral_clusters):
            if len(neutral_clusters) == 1:
                regime_mapping[cluster] = 'Neutral/Volatile'
            else:
                regime_mapping[cluster] = f'Neutral/Volatile_{i+1}'
        if len(neutral_clusters) > 1:
            print("Note: Multiple 'neutral' clusters identified. Consider refining K-Means or naming logic.")
    else:
        print("All clusters have been mapped to Bullish/Bearish.")

    market_regimes_df['Regime_Name'] = market_regimes_df['Regime'].map(regime_mapping)

    print(f"\nState naming mapping: {regime_mapping}")

    # --- 4. Visualize Market Regime Switching ---
    print("\n--- 4. Visualizing Market Regime Switching ---")
    sp500_price = qtu.get_stock_data(['^GSPC'], start_date, end_date)['Close']
    fig, ax = plt.subplots(figsize=(20, 8))
    price_to_plot = sp500_price.reindex(market_regimes_df.index)
    price_to_plot.plot(ax=ax, color='black', label='S&P 500 Index')

    colors = {
        'Bullish': 'lightgreen',
        'Bearish': 'lightcoral',
        'Neutral/Volatile': 'khaki',
        'Neutral/Volatile_1': 'khaki',
        'Neutral/Volatile_2': 'lightgray'
    }
    for regime_name in market_regimes_df['Regime_Name'].unique():
        if regime_name not in colors:
            print(f"Warning: No color defined for regime '{regime_name}'. Using default.")
            colors[regime_name] = 'lightgray'

    for regime_name, color in colors.items():
        regime_periods = market_regimes_df[market_regimes_df['Regime_Name'] == regime_name]
        if not regime_periods.empty:
            for i, block in (regime_periods.index.to_series().diff() > pd.Timedelta(days=1)).cumsum().groupby(level=0):
                start, end = block.index.min(), block.index.max()
                ax.axvspan(start, end, color=color, alpha=0.3)

    legend_elements = [Patch(facecolor=colors.get(name, 'lightgray'), alpha=0.3, label=name)
                       for name in sorted(market_regimes_df['Regime_Name'].unique())]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_title(f'S&P 500 Index and Market Regimes (K={k_value})')
    ax.set_xlabel('Date')
    ax.set_ylabel('S&P 500 Close Price')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    return market_regimes_df

# --- Main Execution ---
if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'BAC', 'V', 'WMT']
    start_date = "2010-01-01"
    end_date = "2025-12-31"
    k_value = 3

    final_market_regimes_df = analyze_market_regimes(tickers, start_date, end_date, k_value)

    print("\n--- Final Market Regimes DataFrame Head ---")
    print(final_market_regimes_df.head())
    print("\n--- Final Market Regimes DataFrame Value Counts ---")
    print(final_market_regimes_df['Regime_Name'].value_counts())