import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import quant_utils as qtu

# --- Main Analysis Function ---

def plot_elbow_method(data: pd.DataFrame, max_k: int):
    """
    Calculates and plots the Elbow Method graph for K-Means clustering.
    """
    if data.empty or len(data) < max_k:
        print("Error: Input data is empty or has too few samples for clustering.")
        return

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    wcss = []
    k_range = range(2, max_k + 1)
    
    print(f"\nCalculating WCSS for k from {min(k_range)} to {max_k}...")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)
        
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o', linestyle='--')
    
    plt.title('The Elbow Method for K-Means (on Real Stock Data)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()
    
    print("\nPlot generated. Look for the 'elbow point' where the rate of decrease in WCSS slows down significantly.")
    print("The k value at this point is generally considered a good choice.")

# --- Main Execution Block ---
if __name__ == '__main__':
    # Define a stock universe for the demonstration
    
    tickers = qtu.get_nasdaq100_tickers()
    start_date = "2020-01-01"
    end_date = "2025-12-31"
    
    # 1. Get and clean the data
    ohlcv_data_raw = qtu.get_stock_data(tickers, start_date, end_date)
    if ohlcv_data_raw.empty:
        exit() # Exit if data download failed
    
    # Forward-fill then back-fill to handle missing data points for some stocks
    ohlcv_data = ohlcv_data_raw.ffill().bfill()    
    
    # 2. Build the complete historical feature matrix in panel format
    full_feature_panel = qtu.create_full_feature_matrix(ohlcv_data)

    # 3. Get the feature snapshot for the most recent day in the dataset
    target_date = full_feature_panel.index.get_level_values('Date').max()
    print(f"\nExtracting feature snapshot for the last available date: {target_date.date()}")
    feature_snapshot = qtu.get_feature_snapshot_for_date(full_feature_panel, target_date)

    if not feature_snapshot.empty:
        print("\n--- Feature Snapshot for Clustering ---")
        print(feature_snapshot.head())
        print("-" * 45)

        # 4. Call the function to run and plot the Elbow Method
        #    For this set of stocks, we'll test up to 8 clusters.
        plot_elbow_method(data=feature_snapshot, max_k=8)
    else:
        print("Could not generate a feature snapshot to run the analysis.")