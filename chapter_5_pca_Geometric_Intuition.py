import pandas as pd
import numpy as np
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings

# --- Configuration ---
# Ignore some warnings to keep the output clean
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Function Definitions ---

def get_sp500_tickers():
    """Robustly fetches the list of S&P 500 tickers from Wikipedia."""
    print("Fetching the list of S&P 500 constituents from Wikipedia...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        # The S&P 500 constituents are in the first table on the page
        payload = pd.read_html(response.text)
        sp500_table = payload[0]
        # Yahoo Finance ticker format may require replacing '.' with '-' (e.g., BRK.B -> BRK-B)
        tickers = sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()
        print(f"Successfully fetched {len(tickers)} tickers.")
        return tickers
    except Exception as e:
        print(f"An error occurred while fetching the ticker list: {e}")
        return None

def calculate_roc(close, n):
    """Calculates the N-period Rate of Change."""
    return ((close - close.shift(n)) / close.shift(n)) * 100

# --- Main Program ---

if __name__ == "__main__":
    # 1. Get the ticker list and download data
    tickers = get_sp500_tickers()
    if tickers:
        # Using a subset for demonstration to reduce download time
        tickers_subset = tickers[:200]
        start_date = "2022-01-01"
        end_date = "2023-12-31"

        print(f"\nDownloading data for {len(tickers_subset)} stocks from {start_date} to {end_date}...")
        all_data = yf.download(tickers_subset, start=start_date, end=end_date, auto_adjust=True)
        # Clean up stocks that have no data for the period
        close_prices = all_data['Close'].dropna(axis=1, how='all')

        # 2. Calculate features: 20-day and 60-day Rate of Change
        roc_20 = calculate_roc(close_prices, 20)
        roc_60 = calculate_roc(close_prices, 60)
        
        # 3. Extract the data snapshot for the most recent day
        latest_date = roc_20.index.max()
        data_snapshot = pd.DataFrame({
            'ROC_20': roc_20.loc[latest_date],
            'ROC_60': roc_60.loc[latest_date]
        }).dropna() # Drop stocks with no valid ROC data on this day

        print(f"\nData prepared for {len(data_snapshot)} stocks as of {latest_date.date()}.")

        # 4. Perform PCA
        X = data_snapshot.values
        pca = PCA(n_components=2)
        pca.fit(X)

        # Extract PCA results
        mean = pca.mean_                    # The center point of the data
        components = pca.components_        # The principal component directions (unit vectors)
        variance = pca.explained_variance_  # The variance explained by each component (the "energy" of the vector)

        print("\nPCA complete.")
        print(f"Data Center (Average ROC): 20-day={mean[0]:.2f}%, 60-day={mean[1]:.2f}%")
        print(f"First Principal Component (PC1) Direction: {components[0]}")
        print(f"Second Principal Component (PC2) Direction: {components[1]}")

        # 5. Visualization
        plt.figure(figsize=(10, 8))

        # Plot the original data points
        plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Individual Stock ROC Distribution')

        # Plot the data center point
        plt.plot(mean[0], mean[1], 'ro', markersize=10, label='Data Center (Mean)')

        # Plot the principal component vectors
        # To make the vectors clearly visible, we need to scale their length by their variance.
        # The vector starts at the mean, points in the direction of the component, and its length is determined by the variance.
        for i, (comp, var) in enumerate(zip(components, variance)):
            # Starting point of the vector
            start_point = mean
            # Endpoint = Start + Direction * Length
            # The length is scaled by sqrt(variance) * 3 to make it more visible.
            end_point = mean + comp * np.sqrt(var) * 3
            
            # Use the arrow function to draw a vector with an arrowhead
            plt.arrow(start_point[0], start_point[1], 
                      end_point[0] - start_point[0], end_point[1] - start_point[1], 
                      head_width=0.5, head_length=0.5, 
                      width=0.1, fc='r', ec='r',
                      label=f'Principal Component {i+1}')
        
        # Set chart properties
        plt.title('Finding the Directions of Maximum Variance (PCA Demonstration)', fontsize=16)
        plt.xlabel('20-Day Rate of Change (%)', fontsize=12)
        plt.ylabel('60-Day Rate of Change (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        # Crucial: Set axes to be equal, otherwise the vectors won't appear orthogonal
        plt.axis('equal')
        plt.legend()
        plt.show()