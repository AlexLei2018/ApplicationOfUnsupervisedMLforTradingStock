import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  # Import the silhouette score function
import quant_utils as qtu

def find_optimal_k_silhouette(data: pd.DataFrame, max_k: int):
    """
    Calculates and plots the optimal k for K-Means using the Silhouette Score.

    This function iterates through a range of k values, calculates the average
    silhouette score for each, and plots the relationship to help find the
    k that maximizes the score.

    Args:
        data (pd.DataFrame): 
            The input feature data, with samples as rows and features as columns.
            The function will automatically standardize this data.
        
        max_k (int): 
            The maximum number of clusters (k) to test.
    """
    if data.empty or len(data) < 2:
        print("Error: Input data is empty or has too few samples for analysis.")
        return

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    print(f"Calculating Silhouette Scores for k from {min(k_range)} to {max(k)}...")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        # fit_predict fits the model and assigns a label to each sample
        labels = kmeans.fit_predict(data_scaled)
        
        # Calculate the average silhouette score for this value of k
        score = silhouette_score(data_scaled, labels)
        silhouette_scores.append(score)
        
    # --- Plot the Silhouette Scores ---
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
    
    # --- Find and Mark the Optimal k ---
    # Find the index of the highest score, then use k_range to get the k value
    optimal_k = k_range[np.argmax(silhouette_scores)]
    max_score = np.max(silhouette_scores)
    
    # Draw a vertical line on the plot to mark the optimal k
    plt.axvline(x=optimal_k, color='red', linestyle='--', 
                label=f'Optimal k = {optimal_k} (Score: {max_score:.2f})')
    
    plt.title('Silhouette Analysis for K-Means Clustering')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.xticks(k_range)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"\nAnalysis complete. The optimal k value based on the highest silhouette score is: {optimal_k}")
    print("This k represents the number of clusters with the best balance of internal cohesion and external separation.")

# --- Main Execution Block ---
if __name__ == '__main__':
    # # --- 1. Define Real Stocks and Time Period to Analyze ---
    # tickers = [
    #     'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 
    #     'JPM', 'BAC', 'V', 'WMT', 'COST', 'PG', 'JNJ', 'PFE', 'XOM'
    # ]
    # start_date = '2022-01-01'
    # end_date = '2023-12-31'
    
    # # --- 2. Download Real Data ---
    # print(f"Downloading data for {len(tickers)} stocks from {start_date} to {end_date}...")
    # try:
    #     price_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    #     close_prices = price_data['Close']
    #     print("Data download successful.")
    # except Exception as e:
    #     print(f"Data download failed: {e}")
    #     exit()
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

    # --- 4. Call the function to run the Silhouette Analysis ---
    find_optimal_k_silhouette(data=feature_snapshot, max_k=10)