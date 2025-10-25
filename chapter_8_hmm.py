import quant_utils as qtu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def analyze_market_regimes_hmm(tickers, start_date, end_date, n_components=3, min_data_points=50):
    """
    Analyzes market regimes using a 3-state Gaussian HMM and robust naming logic.
    """
    # --- 1. Build and standardize market regime feature matrix ---
    print("\n--- 1. Building Market Regime Features for HMM ---")
    price_data_df = qtu.get_stock_data(['^GSPC'], start_date, end_date)

    try:
        sp500_price_series = price_data_df['Close'].squeeze()
    except KeyError:
        raise KeyError(
            f"CRITICAL ERROR: The loaded DataFrame does not contain a 'Close' column. "
            f"Available columns are: {price_data_df.columns.tolist()}"
        )

    if len(sp500_price_series) < min_data_points:
        raise ValueError(f"Not enough historical data.")

    sp500_returns = sp500_price_series.pct_change().dropna()
    sp500_volatility = sp500_returns.rolling(window=21).std().dropna()

    hmm_features = pd.DataFrame({
        'returns': sp500_returns,
        'volatility': sp500_volatility
    }).dropna()

    # Standardize features for numerical stability
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(hmm_features)
    print("HMM features built and standardized successfully.")
    
    # --- 2. Fit HMM Model ---
    print(f"\n--- 2. Identifying Market Regimes using HMM (States={n_components}) ---")
    
    # Use min_covar for stability
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000, 
                        random_state=42, min_covar=1e-3)
    model.fit(feature_matrix)
    
    hidden_states = model.predict(feature_matrix)

    market_regimes_df = hmm_features.copy()
    market_regimes_df['Regime'] = hidden_states
    
    # --- 3. Interpret and name each regime using the robust sorting method ---
    print("\n--- 3. Interpreting and Naming Market Regimes ---")
    
    regime_profiles = []
    for i in range(n_components):
        regime_profiles.append({
            'State': i,
            'Mean_Return': model.means_[i][0], # Using scaled means for sorting
            'Mean_Volatility': model.means_[i][1]
        })
    regime_profiles_df = pd.DataFrame(regime_profiles).set_index('State')
    print("\n--- HMM State Characteristics Profile (in Standardized Units) ---")
    print(regime_profiles_df)

    # --- THE ROBUST NAMING LOGIC ---
    # Step 1: Sort the regimes by their mean return from lowest to highest.
    sorted_profiles = regime_profiles_df.sort_values(by='Mean_Return')
    
    # Step 2: Get the original state numbers (indices) in their new sorted order.
    sorted_indices = sorted_profiles.index.tolist()
    
    # Step 3: Assign labels based on the sorted order. This is guaranteed to work.
    regime_mapping = {
        sorted_indices[0]: 'Bearish',           # The one with the lowest return is Bearish
        sorted_indices[2]: 'Bullish',           # The one with the highest return is Bullish
        sorted_indices[1]: 'Neutral/Volatile'   # The one in the middle is Neutral/Volatile
    }
    # --- END OF ROBUST LOGIC ---

    market_regimes_df['Regime_Name'] = market_regimes_df['Regime'].map(regime_mapping)
    print(f"\nState naming mapping: {regime_mapping}")

    # --- 4. Visualize Market Regime Switching ---
    print("\n--- 4. Visualizing Market Regime Switching ---")
    fig, ax = plt.subplots(figsize=(20, 8))
    
    price_to_plot = sp500_price_series.reindex(market_regimes_df.index)
    price_to_plot.plot(ax=ax, color='black', label='S&P 500 Index')

    colors = {'Bullish': 'lightgreen', 'Bearish': 'lightcoral', 'Neutral/Volatile': 'khaki'}
    unique_regime_names = sorted(list(set(regime_mapping.values())))
    legend_elements = [Patch(facecolor=colors[name], alpha=0.4, label=name) for name in unique_regime_names]

    for state, regime_name in regime_mapping.items():
        regime_periods = market_regimes_df[market_regimes_df['Regime'] == state]
        color = colors.get(regime_name, 'lightgray')
        
        if not regime_periods.empty:
            for i, block in (regime_periods.index.to_series().diff() > pd.Timedelta(days=1)).cumsum().groupby(level=0):
                start, end = block.index.min(), block.index.max()
                ax.axvspan(start, end, color=color, alpha=0.4)

    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_title(f'S&P 500 Index and Market Regimes (HMM, States={n_components})')
    ax.set_xlabel('Date')
    ax.set_ylabel('S&P 500 Close Price')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    return market_regimes_df


# --- Main Execution ---
if __name__ == "__main__":
    tickers = []
    start_date = "2010-01-01"
    end_date = "2025-12-31"
    # Reverting to 3 states for clear, intuitive definitions
    n_hmm_states = 3

    try:
        # It's always a good idea to clear the cache when changing model parameters
        print("Reminder: Please clear the 'cache' folder if you encounter issues.")
        final_hmm_regimes_df = analyze_market_regimes_hmm(tickers, start_date, end_date, n_hmm_states)
        
        print("\n--- Final HMM Market Regimes DataFrame Head ---")
        print(final_hmm_regimes_df.head())
        print("\n--- Final HMM Market Regimes DataFrame Value Counts ---")
        print(final_hmm_regimes_df['Regime_Name'].value_counts())
        
    except Exception as e: # Catch any potential error
        print(f"\nAn error occurred during analysis: {e}")
        