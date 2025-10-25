# Filename: chapter_3_final_practice.py
# Description: The corrected and optimized code for hands-on 
import pandas as pd
import numpy as np
import quant_utils as qtu  # Assuming the English version of quant_utils.py is in the same directory
from sklearn.preprocessing import StandardScaler

# --- Main Execution Block ---
if __name__ == '__main__':
    # Define a broader stock universe for demonstration
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'BAC', 'V', 'WMT', 'PG', 'JNJ']
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    # 1. Get and clean the data
    # The get_stock_data function from quant_utils.py handles downloading and caching
    ohlcv_data = qtu.get_stock_data(tickers, start_date, end_date)
    # Critical cleaning step: forward-fill then back-fill to ensure data continuity
    ohlcv_data = ohlcv_data.ffill().bfill()    
    # 2. Build the complete feature panel data matrix
    full_feature_matrix = qtu.create_full_feature_matrix(ohlcv_data)
    print("\n--- Preview of the final generated feature matrix (panel data) ---")
    print(full_feature_matrix.tail())
    print("\nMatrix Shape (rows, columns):", full_feature_matrix.shape)
    # 3. Prepare for subsequent chapters: Get the latest day's feature snapshot and standardize it
    # get_level_values('Date') is the correct way to access a level of a MultiIndex
    latest_date = full_feature_matrix.index.get_level_values('Date').max()    
    # Use .loc to index features for all stocks on a specific date
    latest_snapshot = full_feature_matrix.loc[latest_date]
    
    scaler = StandardScaler()
    latest_snapshot_scaled = pd.DataFrame(
        scaler.fit_transform(latest_snapshot), 
        index=latest_snapshot.index, 
        columns=latest_snapshot.columns
    )
                                          
    print("\n--- Standardized feature snapshot for the latest day (ready for model input) ---")
    print(latest_snapshot_scaled.head())
