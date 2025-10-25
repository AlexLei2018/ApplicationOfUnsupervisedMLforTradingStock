# chapter_6_iforest.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import yfinance as yf

### Part 1
# --- 1. Data Preparation ---
# Assuming we already have the feature_matrix_snapshot generated in a previous step.
# To make this section runnable independently, we will regenerate a real feature snapshot.
def build_feature_snapshot(ohlcv_data):
    # Use data from the last year (approx. 252 trading days)
    latest_date_data = ohlcv_data.iloc[-252:] 
    # Ensure there's enough data (e.g., at least one quarter)
    if len(latest_date_data) < 63: return pd.DataFrame()
    
    close = latest_date_data['Close']
    
    # Calculate Rate of Change over 21 days
    roc_21 = (close.iloc[-1] / close.iloc[-22] - 1) * 100
    # Calculate Rate of Change over 63 days
    roc_63 = (close.iloc[-1] / close.iloc[-64] - 1) * 100
    # Calculate annualized volatility over 21 days
    vol_21 = close.iloc[-21:].pct_change().std() * np.sqrt(252)
    
    return pd.DataFrame({'ROC_21': roc_21, 'ROC_63': roc_63, 'VOL_21': vol_21}).dropna()

# Define the stock tickers to analyze
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'JNJ', 'XOM', 'PG', 'CVX', 'LLY']
# Download historical data using yfinance
data = yf.download(tickers, start="2022-01-01", end="2025-12-31", auto_adjust=True, progress=False)
# Build the feature matrix from the downloaded data
feature_matrix_snapshot = build_feature_snapshot(data)

# Initialize a standard scaler to normalize the features
scaler = StandardScaler()
# Scale the feature matrix
feature_matrix_scaled = scaler.fit_transform(feature_matrix_snapshot)

# --- 2. Initialize and Train the Isolation Forest Model ---
# contamination: The expected proportion of anomalies in the data set. 'auto' is a good default.
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)

# .fit_predict() returns 1 for inliers (normal) and -1 for outliers (anomalies)
predictions = iso_forest.fit_predict(feature_matrix_scaled)
# .decision_function() returns a score; the lower the score, the more anomalous the point is.
anomaly_scores = iso_forest.decision_function(feature_matrix_scaled)

# --- 3. Organize and Analyze Results ---
# Create a copy of the original feature matrix to store the results
results_df = feature_matrix_snapshot.copy()
# Add the anomaly scores and predictions to the results DataFrame
results_df['Anomaly_Score'] = anomaly_scores
results_df['Is_Anomaly'] = predictions
# Sort the DataFrame by the anomaly score in ascending order to see the most anomalous items first
results_df = results_df.sort_values(by='Anomaly_Score', ascending=True)

print("\n--- Isolation Forest Detection Results ---")
print("Top 5 most anomalous stocks and their profiles:")
# Display the top 5 most anomalous entries
print(results_df.head(5))

print("\nStocks marked as anomalies (-1):")
# Print the list of indices (stock tickers) that were flagged as anomalies
print(results_df[results_df['Is_Anomaly'] == -1].index.tolist())


## part 2
from sklearn.neighbors import LocalOutlierFactor

# --- 1. Initialization and Prediction ---
# contamination: The expected proportion of outliers, used to set the threshold.
# n_neighbors: Defines the size of the "local neighborhood".
lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')

# fit_predict directly returns 1 (inlier) or -1 (outlier).
lof_predictions = lof.fit_predict(feature_matrix_scaled)

# The anomaly score for LOF is `negative_outlier_factor_`; the lower the score, the more anomalous.
lof_scores = lof.negative_outlier_factor_

# --- 2. Organize Results ---
results_df['LOF_Score'] = lof_scores
results_df['LOF_Is_Anomaly'] = lof_predictions
results_df = results_df.sort_values(by='LOF_Score', ascending=True)

print("\n--- LOF Local Outlier Detection Results ---")
print("Top 5 most anomalous stocks and their LOF scores:")
print(results_df[['LOF_Score', 'LOF_Is_Anomaly']].head(5))