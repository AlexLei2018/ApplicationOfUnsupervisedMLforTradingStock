### filename: chapter_7_full_pipeline.py

### Step 1: Rolling K-Means Clustering

import warnings
warnings.filterwarnings("ignore")

# 1. Load Required Libraries
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ==============================================================================
# 2. Download Real Stock Market Data
# ==============================================================================
# We select a group of well-known stocks from different sectors to observe changes in their relationships.
def get_sp100_tickers() -> list:
    """
    Fetches S&P 100 tickers from Wikipedia using requests and BeautifulSoup.
    """
    try:
        url = 'https://en.wikipedia.org/wiki/S%26P_100'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # The components table is the second 'wikitable' on the page
        tables = soup.find_all('table', {'class': 'wikitable'})
        if len(tables) < 2:
            print("Error: Could not find the components table.")
            return []
        
        ticker_table = tables[1]
        tickers = []
        for row in ticker_table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if len(cells) > 0:
                tickers.append(cells[0].text.strip())
        
        # Clean tickers for yfinance (e.g., 'BRK.B' -> 'BRK-B')
        tickers = [t.replace('.', '-') for t in tickers]
        print(f"Successfully fetched {len(tickers)} S&P 100 tickers.")
        return tickers
    except Exception as e:
        print(f"An error occurred in get_sp100_tickers: {e}")
        return []

def download_ohlcv_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical OHLCV data for a list of tickers using yfinance.
    """
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    all_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    # The yfinance library returns a MultiIndex column format automatically.
    # We just need to remove rows with all NaNs which can occur for some tickers.
    all_data.dropna(how='all', inplace=True)
    print("Data download complete.")
    return all_data

# ==============================================================================
# 3. Implement Feature Engineering Function
# ==============================================================================
# --- Reuse Feature Calculation Functions from Chapter 3 ---
def calculate_roc(close_prices, n):
    return (close_prices.diff(n) / close_prices.shift(n))

def calculate_atr_percent_vectorized(high_df, low_df, close_df, n=14):
    """
    Calculates the ATR Percent for multiple stocks in a vectorized manner.
    """
    tr1 = high_df - low_df
    tr2 = abs(high_df - close_df.shift(1))
    tr3 = abs(low_df - close_df.shift(1))
    
    true_range_df = np.maximum(tr1, np.maximum(tr2, tr3))    
    atr_df = true_range_df.ewm(span=n, adjust=False, min_periods=n).mean()    
    atr_percent_df = (atr_df / close_df) * 100
    
    return atr_percent_df 

def calculate_momentum(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculates momentum on a simple DataFrame (columns are tickers)."""
    return prices.pct_change(periods=window)

def calculate_volatility(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculates volatility on a simple DataFrame (columns are tickers)."""
    daily_returns = prices.pct_change(1)
    return daily_returns.rolling(window=window).std()

# --- STEP 2: FEATURE ENGINEERING ---

def calculate_features(daily_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates market features based on daily data by calling individual feature functions.

    This function acts as a controller, preparing data and orchestrating calls
    to modular feature calculation functions.

    Args:
        daily_data (pd.DataFrame): 
            DataFrame with a MultiIndex, containing daily OHLCV data.
        momentum_window (int): 
            Lookback period for momentum calculation.
        volatility_window (int): 
            Lookback period for volatility calculation.

    Returns:
        pd.DataFrame: 
            A DataFrame with a MultiIndex ('Ticker', 'FeatureName') containing
            the calculated features for each stock on each day.
    """
    print("Calculating daily features...")
        
    # --- Feature 1: Momentum ---
    # We define momentum as the return over the past 20 trading days (approx. 1 month).
    # (Price_today / Price_20_days_ago) - 1
    momentum_20d = calculate_momentum(daily_data,20)
    
    # --- Feature 2: Volatility ---
    # We define volatility as the standard deviation of daily returns over the past 20 days.
    
    volatility_20d = calculate_volatility(daily_data,20)

    close = daily_data['Close']
    high = daily_data['High']
    low = daily_data['Low']
    
    roc_21d = calculate_roc(close, 21)
    roc_63d = calculate_roc(close, 63)
    atr_21d = calculate_atr_percent_vectorized(high, low, close, 21)
    
    # --- Combine features into a single DataFrame ---
    # We will use a dictionary to build the new MultiIndex DataFrame
    feature_list = [
        roc_21d.add_suffix('_ROC21'),
        roc_63d.add_suffix('_ROC63'),
        atr_21d.add_suffix('_ATRPCT21'),
        momentum_20d['Close'].add_suffix('_MOM20'),
        volatility_20d['Close'].add_suffix('_VAR20')
    ]
    features_raw = pd.concat(feature_list, axis=1)
    print("Feature calculation complete.")
    return features_raw
    
# ==============================================================================
# 4. Your Main Rolling Cluster Function (with minor adjustments for the feature function)
# ==============================================================================
def extract_features_with_backfill(
    features_df: pd.DataFrame, 
    date_str: str
) -> Optional[pd.Series]:
    """
    Extracts a feature snapshot for a given date, backfilling to previous days if necessary.

    This function attempts to extract a feature snapshot for the given date.
    If the date is not found in the DataFrame's index, it iteratively checks
    previous dates until a valid snapshot is found.

    Args:
        features_df (pd.DataFrame):
            The DataFrame containing features with a DatetimeIndex.
        date_str (str):
            The target date for extraction, in 'YYYY-MM-DD' format.

    Returns:
        Optional[pd.Series]:
            A Pandas Series representing the feature snapshot for the valid date found.
            Returns None if no valid date is found in the DataFrame.
    """
    # --- Input Validation ---
    if not isinstance(features_df, pd.DataFrame):
        raise TypeError("Input 'features_df' must be a pandas DataFrame.")
    if not isinstance(features_df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")
    
    try:
        target_date = pd.to_datetime(date_str)
    except ValueError:
        raise ValueError("Invalid date format. Please use 'YYYY-MM-DD'.")

    # --- Iteratively Backfill Until Valid Date is Found ---
    current_date = target_date
    
    while True:
        try:
            snapshot_features = features_df.loc[current_date]
            print(f"Successfully extracted features for: {current_date.strftime('%Y-%m-%d')}")
            return snapshot_features

        except KeyError:
            # If the date is not found, move to the previous day
            current_date -= pd.Timedelta(days=1)
            
            # Stop if we've gone back too far (before the start of the data)
            if current_date < features_df.index.min():
                print("Error: Reached the beginning of the DataFrame without finding a valid date.")
                return None

            print(f"Date {current_date.strftime('%Y-%m-%d')} not found, checking previous day...")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

def transform_series_to_feature_matrix(data_series: pd.Series) -> Optional[pd.DataFrame]:
    """
    Transforms a "long" format Series into a "wide" feature matrix DataFrame.

    This function takes a Series with a combined 'Ticker_Feature' index and pivots it
    to create a DataFrame where the index is the Ticker and the columns are the 
    Feature names.

    Args:
        data_series (pd.Series):
            The input Series where the index is a string like 'AAPL_ROC21'.

    Returns:
        Optional[pd.DataFrame]:
            A new DataFrame in a wide format (the feature matrix).
            Returns None if the input Series is empty.
    """
    # --- Input Validation ---
    if not isinstance(data_series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
        
    if data_series.empty:
        print("Warning: The input Series is empty. Returning None.")
        return None

    # --- Step 1: Convert the "long" Series to a 3-column DataFrame ---
    # This is the same logic from our previous function.
    long_df = data_series.reset_index()
    long_df.columns = ['Original_Index', 'Value']
    long_df[['Ticker', 'Feature']] = long_df['Original_Index'].str.rsplit('_', n=1, expand=True)

    # --- Step 2: Pivot the "long" DataFrame to a "wide" format ---
    # This is the core of the transformation.
    # - index='Ticker': specifies what the new index should be.
    # - columns='Feature': specifies what the new columns should be.
    # - values='Value': specifies what to use to fill the DataFrame's cells.
    try:
        feature_matrix = long_df.pivot(index='Ticker', columns='Feature', values='Value')
    except Exception as e:
        print(f"An error occurred during the pivot operation: {e}")
        # This can happen if there are duplicate (Ticker, Feature) pairs,
        # in which case pivot_table should be used.
        print("Attempting to use pivot_table with an aggregation function (mean)...")
        feature_matrix = long_df.pivot_table(index='Ticker', columns='Feature', values='Value', aggfunc='mean')

    # --- Step 3: Clean up the resulting DataFrame ---
    # The pivot operation can leave a name on the column index, which we can remove.
    feature_matrix.columns.name = None
    
    print("Successfully transformed the Series into a feature matrix.")
    
    return feature_matrix

# def get_clusters_snapshot(snapshot_df: pd.DataFrame, n_clusters: int = 4) -> Optional[pd.DataFrame]:
#     """
#     Performs K-Means clustering on a feature snapshot to profile the market.

#     It takes a snapshot of stock features, standardizes them, and then groups
#     stocks into clusters based on their feature similarity.

#     Args:
#         snapshot_df (pd.DataFrame): 
#             The DataFrame returned by build_feature_snapshot (tickers as index).
#         n_clusters (int): 
#             The number of clusters (market groups) to create. Defaults to 4.

#     Returns:
#         Optional[pd.DataFrame]:
#             The original snapshot DataFrame with an added 'Cluster' column 
#             indicating the group for each stock. Returns None if clustering fails.
#     """
#     if snapshot_df is None or snapshot_df.empty:
#         print("Error: Input snapshot_df is empty, cannot perform clustering.")
#         return None
        
#     if len(snapshot_df) < n_clusters:
#         print(f"Warning: Number of stocks ({len(snapshot_df)}) is less than n_clusters ({n_clusters}). Clustering aborted.")
#         return None

#     print(f"Performing K-Means clustering to create {n_clusters} groups...")
    
#     # 1. Standardize the features (CRITICAL STEP)
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(snapshot_df)
    
#     # 2. Perform K-Means clustering
#     kmeans = KMeans(
#         n_clusters=n_clusters, 
#         n_init='auto',          # Use modern default to avoid warnings
#         random_state=42         # For reproducibility
#     )
#     kmeans.fit(scaled_features)
    
#     # 3. Add the cluster labels back to the original (unscaled) DataFrame
#     # This makes interpretation much easier.
#     clustered_df = snapshot_df.copy()
#     clustered_df['Cluster'] = kmeans.labels_
    
#     print("Clustering complete.")
#     return clustered_df

def get_ranked_clusters(
    snapshot_df: pd.DataFrame, 
    rank_by_feature: str, 
    n_clusters: int = 4, 
    ascending: bool = False
) -> Optional[pd.DataFrame]:
    """
    Performs K-Means clustering and then ranks the cluster labels based on a feature.

    This ensures that the final cluster labels are not arbitrary but have a
    consistent, interpretable order (e.g., Cluster 0 is always the group
    with the highest momentum).

    Args:
        snapshot_df (pd.DataFrame): 
            The feature snapshot (index=Ticker, columns=Features).
        rank_by_feature (str): 
            The name of the feature column to use for ranking the clusters
            (e.g., 'Momentum_20D').
        n_clusters (int): 
            The number of clusters to create.
        ascending (bool): 
            Determines the sort order for ranking. False means the group
            with the highest feature value gets label 0. True means the
            group with the lowest feature value gets label 0.

    Returns:
        Optional[pd.DataFrame]:
            The original snapshot DataFrame with a new, correctly ranked 'Cluster' column.
            Returns None if clustering fails.
    """
    # --- Input Validation ---
    if snapshot_df is None or snapshot_df.empty:
        print("Error: Input snapshot_df is empty.")
        return None
    if rank_by_feature not in snapshot_df.columns:
        print(f"Error: The ranking feature '{rank_by_feature}' is not in the DataFrame columns.")
        return None
    if len(snapshot_df) < n_clusters:
        print(f"Warning: Number of stocks ({len(snapshot_df)}) is less than n_clusters ({n_clusters}).")
        return None

    # --- Step 1: Standard K-Means Clustering ---
    # Standardize features before clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(snapshot_df)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    kmeans.fit(scaled_features)
    
    # Add the initial, ARBITRARY cluster labels to the DataFrame
    clustered_df = snapshot_df.copy()
    clustered_df['Arbitrary_Cluster'] = kmeans.labels_

    # --- Step 2: Analyze and Rank the Clusters (The Core Logic) ---
    # Calculate the mean of the ranking feature for each arbitrary cluster
    cluster_rank_values = clustered_df.groupby('Arbitrary_Cluster')[rank_by_feature].mean()
    
    # Sort the clusters based on their mean feature value
    sorted_clusters = cluster_rank_values.sort_values(ascending=ascending)
    
    # --- Step 3: Create the Mapping from Arbitrary to Ranked Labels ---
    # The index of sorted_clusters now holds the arbitrary labels in the desired order.
    # We create a dictionary that maps {arbitrary_label: ranked_label}
    # For example: {3: 0, 1: 1, 0: 2, 2: 3} means arbitrary cluster 3 had the
    # highest momentum and should be re-labeled as 0.
    rank_mapping = {old_label: new_rank for new_rank, old_label in enumerate(sorted_clusters.index)}
    
    # --- Step 4: Apply the Mapping and Clean Up ---
    # Use the mapping to create the final, ordered 'Cluster' column
    clustered_df['Cluster'] = clustered_df['Arbitrary_Cluster'].map(rank_mapping)
    
    # Drop the temporary column
    clustered_df.drop(columns=['Arbitrary_Cluster'], inplace=True)
    
    print(f"Successfully clustered stocks and ranked clusters by '{rank_by_feature}'.")
    
    return clustered_df.sort_values(['Cluster',rank_by_feature])

# def get_monthly_clusters(full_ohlcv_data, lookback_window=252, k=5):
#     """
#     Performs a rolling monthly cluster analysis on the entire time series.
#     """
    
#     # Get the last trading day of each month as the "decision day"
#     rebalance_dates = full_ohlcv_data.index.to_series().resample('M').last()
    
#     all_cluster_labels = []

#     print("\nStarting rolling monthly clustering...")
#     for date in rebalance_dates:
#         # 1. Slice the lookback data window for the current decision day
#         start_date = date - pd.DateOffset(days=lookback_window)
#         window_data = full_ohlcv_data.loc[start_date:date]
        
#         # 2. Ensure there is enough data
#         if len(window_data) < 126: # At least half a year of data is needed for the longest momentum calculation
#             continue
            
#         # 3. Build the feature matrix for the current window (returns a snapshot for the decision day)
#         feature_matrix = build_feature_matrix(window_data)
        
#         # 4. Clean data: Drop any stocks that have NaNs due to insufficient data
#         feature_matrix = feature_matrix.dropna()
#         if feature_matrix.empty:
#             continue
            
#         # 5. Standardize features
#         scaler = StandardScaler()
#         feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
#         # 6. Perform K-Means clustering
#         kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
#         cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
        
#         # 7. Save the clustering results for the month
#         monthly_clusters = pd.Series(cluster_labels, index=feature_matrix.index, name=date)
#         all_cluster_labels.append(monthly_clusters)
        
#         print(f"  > Completed cluster analysis for {date.date()}...")

#     # Combine the results from each month into a single large DataFrame
#     if not all_cluster_labels:
#         print("Warning: Failed to generate any cluster results. Please check data or parameters.")
#         return pd.DataFrame()
        
#     clusters_over_time = pd.concat(all_cluster_labels, axis=1).T
#     clusters_over_time.index.name = 'Date'
    
#     print("\nRolling clustering complete!")
#     return clusters_over_time


# --- STEP 3 & 4: BACKTESTING AND EVALUATION ---
# ==============================================================================
# 5. Run the Main Process and Display Results
# ==============================================================================
# Run the function using the downloaded real data `all_data`
# Set k=4 to try to divide the market into 4 main groups

# Assume the following functions have been defined elsewhere in your code:
# - select_top_stocks_from_strongest_cluster(...)
# - calculate_features(...)
# - extract_features_with_backfill(...)
# - transform_series_to_feature_matrix(...)
# - get_ranked_clusters(...)

def run_monthly_backtest(
    daily_ohlcv: pd.DataFrame, 
    start_date: str, 
    end_date: str,
    **kwargs
) -> Dict[pd.Timestamp, List[str]]:
    """
    Runs a backtest over a specified period, generating a stock portfolio
    for the end of each month. (Corrected for cross-platform compatibility).
    """
    print(f"\n--- Starting Monthly Backtest from {start_date} to {end_date} ---")

    # --- Step 1: Identify all monthly rebalancing dates ---
    rebalancing_dates = pd.date_range(
        start=start_date, 
        end=end_date, 
        freq='BM' # Business Month End
    )
    
    print(f"Identified {len(rebalancing_dates)} rebalancing dates.")

    # # --- Step 2: Pre-calculate all features ---
    # print("Pre-calculating all daily features for the entire period...")
    # daily_features = calculate_features(daily_ohlcv)
    # if daily_features is None or daily_features.empty:
    #     print("  -> Failure: Could not calculate features. Backtest cannot proceed.")
    #     return {}

    # --- Step 3: Loop through each rebalancing date and select stocks ---
    monthly_portfolios = {}
    
    for date in rebalancing_dates:
        # --- THE FIX IS HERE ---
        # Removed the non-portable '-' flag from the format string.
        # '%Y-%m-%d' is universally supported across Windows, Linux, and macOS.
        date_str = date.strftime('%Y-%m-%d')
        
        # Call the optimized selection function for this specific date
        selected_stocks = select_top_stocks_from_strongest_cluster(
            daily_ohlcv,
            analysis_date=date_str,
            **kwargs # Pass along other parameters like top_n
        )
        
        # Store the selected portfolio for this date
        monthly_portfolios[date] = selected_stocks
        print("-" * 20)

    print("--- Monthly Backtest Complete ---")
    return monthly_portfolios

# monthly_clusters_df = get_monthly_clusters(all_data, lookback_window=365, k=4)

def select_top_stocks_from_strongest_cluster(
    daily_ohlcv: pd.DataFrame, 
    analysis_date: str,
    momentum_feature: str = 'MOM20',
    num_clusters: int = 4,
    top_n: int = 5
) -> List[str]:
    """
    Performs an end-to-end stock selection process for a given date.

    This function orchestrates the entire workflow:
    1. Calculates features from historical OHLCV data.
    2. Extracts a point-in-time feature snapshot for the analysis date.
    3. Transforms the data into a feature matrix.
    4. Clusters the stocks into groups and ranks the groups by momentum.
    5. Selects the top N stocks from the strongest momentum cluster.

    Args:
        daily_ohlcv (pd.DataFrame): 
            The input DataFrame containing historical OHLCV data for all stocks.
            It must have a MultiIndex on columns like ('Ticker', 'OHLC').
        analysis_date (str): 
            The target date for the analysis, in 'YYYY-MM-DD' format.
        momentum_feature (str): 
            The name of the feature column to use for ranking clusters and stocks.
            Defaults to 'Momentum_20D'.
        num_clusters (int):
            The number of clusters to group the stocks into. Defaults to 4.
        top_n (int): 
            The number of top stocks to select from the strongest cluster.
            Defaults to 5.

    Returns:
        List[str]:
            A list of stock tickers selected for investment. Returns an empty
            list if any step in the process fails.
    """
    print(f"\n--- Starting stock selection process for date: {analysis_date} ---")

    # --- Step 1: Calculate features for the entire historical period ---
    # This function should take the raw daily data and compute features like
    # momentum, volatility, etc., for every stock on every day.
    print("Step 1: Calculating daily features...")
    daily_features = calculate_features(daily_ohlcv)
    if daily_features is None or daily_features.empty:
        print("  -> Failure: Feature calculation returned no data. Aborting.")
        return []

    # --- Step 2: Extract a point-in-time snapshot for the analysis date ---
    # This function robustly finds the feature data for the target date,
    # looking at previous days if the target date is a non-trading day.
    print(f"Step 2: Extracting feature snapshot for {analysis_date}...")
    snapshot_features = extract_features_with_backfill(daily_features, analysis_date)
    if snapshot_features is None or snapshot_features.empty:
        print(f"  -> Failure: Could not extract feature snapshot for {analysis_date}. Aborting.")
        return []

    # --- Step 3: Transform the long-format Series into a wide feature matrix ---
    # The snapshot is converted into a DataFrame where index=Ticker, columns=Features.
    print("Step 3: Transforming snapshot into a feature matrix...")
    snapshot_df = transform_series_to_feature_matrix(snapshot_features)
    if snapshot_df is None or snapshot_df.empty:
        print("  -> Failure: Could not create a feature matrix from the snapshot. Aborting.")
        return []
        
    # --- Step 4: Group stocks into ranked clusters ---
    # This function performs K-Means clustering and then re-labels the clusters
    # so that Cluster 0 is always the one with the highest average momentum.
    print(f"Step 4: Clustering stocks and ranking by '{momentum_feature}'...")
    clustered_df = get_ranked_clusters(snapshot_df, rank_by_feature=momentum_feature, n_clusters=num_clusters)
    if clustered_df is None or clustered_df.empty:
        print("  -> Failure: Clustering process failed. Aborting.")
        return []

    # --- Step 5: Select the top N stocks from the strongest cluster ---
    # Since our clusters are ranked, the strongest momentum cluster is always Cluster 0.
    print(f"Step 5: Selecting top {top_n} stocks from the strongest cluster (Cluster 0)...")
    
    # Filter for stocks belonging to the strongest cluster.
    strongest_cluster_df = clustered_df[clustered_df['Cluster'] == 3]
    
    # CRITICAL FIX: Sort by momentum in DESCENDING order to get the strongest stocks at the top.
    # The original code's ascending sort was a bug that selected the weakest stocks.
    top_stocks_in_cluster = strongest_cluster_df.sort_values(by=momentum_feature, ascending=False)
    
    # Select the top N tickers from the sorted DataFrame.
    # .head(top_n) is a safe way to select, it won't fail if there are fewer than N stocks.
    selected_tickers = top_stocks_in_cluster.head(top_n).index.tolist()

    print(f"--- Process complete. Selected stocks: {selected_tickers} ---")
    return selected_tickers
def calculate_portfolio_returns(
    monthly_portfolios: Dict[pd.Timestamp, List[str]],
    daily_ohlcv: pd.DataFrame
) -> Optional[pd.Series]:
    """
    Calculates simplified monthly returns of an equal-weighted portfolio.

    The return for each month is calculated based on the change from the
    last closing price of the previous month to the last closing price of the current month.
    The portfolio chosen at the end of Month T is used to calculate the return for Month T+1.

    Args:
        monthly_portfolios (Dict[pd.Timestamp, List[str]]):
            A dictionary from the backtest, where keys are decision dates (month-ends)
            and values are the lists of selected stock tickers.
        daily_ohlcv (pd.DataFrame):
            The full historical OHLCV data for all stocks.

    Returns:
        Optional[pd.Series]:
            A pandas Series where the index is the end date of each holding period
            and the values are the calculated portfolio returns for that month.
    """
    print("\n--- Starting Simplified Portfolio Return Calculation ---")

    # --- Step 1: Prepare a DataFrame of monthly returns for ALL stocks ---
    try:
        close_prices = daily_ohlcv.xs('Close', axis=1)
    except KeyError:
        print("Error: Could not find 'Close' prices in the 'OHLC' level of the daily_ohlcv DataFrame.")
        return None

    # Resample to get the last closing price of each month for every stock.
    # 'M' stands for Month-End frequency.
    monthly_close_prices = close_prices.resample('M').last()

    # Calculate the percentage change from one month-end to the next.
    # This gives us a convenient lookup table of monthly returns for every stock.
    monthly_returns_all_stocks = monthly_close_prices.pct_change()

    # --- Step 2: Loop through portfolios and look up returns ---
    decision_dates = sorted(monthly_portfolios.keys())
    
    all_portfolio_returns = []

    # Iterate through each decision period
    for i in range(len(decision_dates) - 1):
        # The date the decision was made (e.g., end of January)
        decision_date = decision_dates[i]
        
        # The date when the holding period ends (e.g., end of February)
        period_end_date = decision_dates[i+1]
        
        # Get the list of stocks selected on the decision date
        portfolio_tickers = monthly_portfolios[decision_date]
        
        if not portfolio_tickers:
            print(f"Skipping period ending {period_end_date.strftime('%Y-%m-%d')}: No stocks in portfolio.")
            continue

        try:
            # Look up the returns for the selected stocks for the period ending on this date.
            individual_returns = monthly_returns_all_stocks.loc[period_end_date, portfolio_tickers].dropna()
            
            if individual_returns.empty:
                print(f"Warning: Could not find return data for any stock in period ending {period_end_date.strftime('%Y-%m-%d')}.")
                continue

            # For an equal-weighted portfolio, the return is the simple average.
            monthly_portfolio_return = individual_returns.mean()
            
            all_portfolio_returns.append({
                'Period_End_Date': period_end_date,
                'Portfolio_Return': monthly_portfolio_return
            })
            print(f"Calculated return for period ending {period_end_date.strftime('%Y-%m-%d')}: {monthly_portfolio_return:.4f}")

        except KeyError:
            print(f"Warning: Could not find return data for period ending {period_end_date.strftime('%Y-%m-%d')}. Skipping.")
            continue
            
    if not all_portfolio_returns:
        print("Error: Could not calculate returns for any period.")
        return None

    returns_df = pd.DataFrame(all_portfolio_returns)
    returns_series = returns_df.set_index('Period_End_Date')['Portfolio_Return']
    
    print("\n--- Portfolio Return Calculation Complete ---")
    return returns_series
def get_benchmark_returns(
    ticker: str = '^GSPC', 
    start_date: str = '2021-01-01', 
    end_date: str = '2023-12-31'
) -> pd.Series:
    """
    Downloads historical data for a benchmark ticker and calculates its monthly returns.
    """
    print(f"\nDownloading benchmark data for {ticker}...")
    benchmark_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    
    # Resample to get month-end closing prices
    benchmark_monthly_close = benchmark_data['Close'].resample('M').last()
    
    # Calculate monthly returns
    benchmark_returns = benchmark_monthly_close.pct_change()
    benchmark_returns.name = f"{ticker}_Returns"
    
    return benchmark_returns

def calculate_performance_metrics(
    returns: pd.Series, 
    risk_free_rate: float = 0.02
) -> pd.Series:
    """
    Calculates key performance metrics from a time series of returns.

    Args:
        returns (pd.Series): 
            A Series of periodic (e.g., monthly) returns.
        risk_free_rate (float): 
            The annualized risk-free rate for Sharpe Ratio calculation.

    Returns:
        pd.Series: 
            A Series containing the calculated performance metrics.
    """
    if returns.empty:
        print("Warning: Returns series is empty. Cannot calculate metrics.")
        return pd.Series(dtype=float)

    # --- Basic Stats ---
    # Assuming monthly returns, so 12 periods per year
    periods_per_year = 12
    
    # --- Cumulative and Annualized Return ---
    cumulative_return = (1 + returns).prod() - 1
    num_years = len(returns) / periods_per_year
    annualized_return = (1 + cumulative_return)**(1 / num_years) - 1
    
    # --- Annualized Volatility (Risk) ---
    annualized_volatility = returns.std() * np.sqrt(periods_per_year)
    
    # --- Sharpe Ratio (Risk-Adjusted Return) ---
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    # --- Maximum Drawdown ---
    cumulative_wealth = (1 + returns).cumprod()
    peak = cumulative_wealth.cummax()
    drawdown = (cumulative_wealth - peak) / peak
    max_drawdown = drawdown.min()
    
    # --- Compile Results ---
    metrics = pd.Series({
        'Cumulative Return': f"{cumulative_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Annualized Volatility': f"{annualized_volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Maximum Drawdown': f"{max_drawdown:.2%}",
    })
    
    return metrics

# This function uses matplotlib to create a clear visualization of your strategy's growth versus the benchmark.
# code

import matplotlib.pyplot as plt

def plot_performance_comparison(
    strategy_returns: pd.Series, 
    benchmark_returns: pd.Series
):
    """
    Plots the cumulative return (equity curve) of a strategy vs. a benchmark.

    Args:
        strategy_returns (pd.Series): 
            The periodic returns of your strategy.
        benchmark_returns (pd.Series): 
            The periodic returns of the benchmark.
    """
    # Create a DataFrame to hold both return series, align their indexes
    comparison_df = pd.DataFrame({
        'Strategy': strategy_returns,
        'Benchmark': benchmark_returns
    }).dropna()

    # Calculate cumulative wealth for both, starting from 1 (or 100 for percentage)
    cumulative_wealth = (1 + comparison_df).cumprod()
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
    fig, ax = plt.subplots(figsize=(12, 7))
    
    cumulative_wealth['Strategy'].plot(ax=ax, color='royalblue', linewidth=2, label='My Strategy')
    cumulative_wealth['Benchmark'].plot(ax=ax, color='grey', linestyle='--', linewidth=2, label='S&P 500 (Benchmark)')
    
    # --- Formatting ---
    ax.set_title('Strategy Performance vs. S&P 500 Benchmark', fontsize=16)
    ax.set_ylabel('Cumulative Return (Growth of $1)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper left', fontsize=12)
    
    # Use a log scale on the y-axis to better visualize percentage changes
    ax.set_yscale('log')
    # Format y-axis ticks to show dollar values instead of scientific notation
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # This is a placeholder for your actual data loading.
    # daily_ohlcv_data would be the full DataFrame you load from yfinance.
    # daily_ohlcv_data = download_ohlcv_data(...) 
    TICKERS = get_sp100_tickers()
    # Let's limit to a smaller set for a quicker example
    TICKERS = TICKERS[:50] 
    START_DATE = '2018-01-01'
    END_DATE = '2025-12-31'
    TOP_N_STOCKS = 5

    # --- Run the Full Workflow ---
    # 1. Download data
    daily_ohlcv = download_ohlcv_data(TICKERS, START_DATE, END_DATE)
        
    # Assume `daily_ohlcv_data` and all helper functions are loaded.

    # --- Run the backtest for a 2-year period ---
    # start_date='2021-01-01',
    # end_date='2022-12-31',
    portfolios = run_monthly_backtest(
        daily_ohlcv=daily_ohlcv,
        start_date='2021-01-01',
        end_date='2022-12-31',
        momentum_feature='MOM20', # Pass custom parameters
        top_n=TOP_N_STOCKS
    )
    
    # --- Print the results ---
    print("\n\n--- Final Monthly Portfolios ---")
    for date, stocks in portfolios.items():
        print(f"Portfolio for {date.strftime('%Y-%m-%d')}: {stocks}")

    returns_series = calculate_portfolio_returns(portfolios, daily_ohlcv)
    # --- 3. Calculate and Print Performance Metrics ---
    print("\n--- My Strategy Performance Metrics ---")
    strategy_metrics = calculate_performance_metrics(returns_series)
    print(strategy_metrics)

    print("\n--- S&P 500 (Benchmark) Performance Metrics ---")
    sp500_returns = get_benchmark_returns('^GSPC')
    sp500_returns = sp500_returns.fillna(0)['^GSPC']
    benchmark_metrics = calculate_performance_metrics(sp500_returns)
    print(benchmark_metrics)

    sp500_returns = get_benchmark_returns('^GSPC')
    sp500_returns = sp500_returns.fillna(0)['^GSPC']
    
    plot_performance_comparison(returns_series,sp500_returns)
# ### Part 2: Cluster Drift Monitoring

# import pandas as pd
# import numpy as np
# from sklearn.metrics import adjusted_rand_score
# import matplotlib.pyplot as plt

# # --- Preparation: We need a DataFrame that records historical clustering results ---
# # 'monthly_clusters_df' is the DataFrame we generated in the previous section, with a (Date, Stock) format.

# def monitor_cluster_drift(historical_clusters):
#     """Calculates the cluster drift (ARI) for each month relative to the previous month."""
#     ari_scores = {}
#     for i in range(1, len(historical_clusters.index)):
#         current_date = historical_clusters.index[i]
#         previous_date = historical_clusters.index[i-1]
#         labels_current = historical_clusters.loc[current_date].dropna()
#         labels_previous = historical_clusters.loc[previous_date].dropna()
#         common_tickers = labels_current.index.intersection(labels_previous.index)
        
#         ari = adjusted_rand_score(labels_current[common_tickers], labels_previous[common_tickers])
#         ari_scores[current_date] = ari
#     return pd.Series(ari_scores, name='Monthly_ARI_Score')

# # --- Run the monitoring ---
# cluster_drift_series = monitor_cluster_drift(monthly_clusters_df)

# # --- Visualize the drift monitoring results ---
# plt.figure(figsize=(15, 7))
# cluster_drift_series.plot(kind='bar', color='skyblue')
# drift_threshold = 0.6
# plt.axhline(y=drift_threshold, color='r', linestyle='--', label=f'Drift Alert Threshold ({drift_threshold})')
# plt.title('Monthly Cluster Stability Monitoring (Adjusted Rand Index)')
# plt.ylabel('ARI Score (vs. Previous Month)')
# plt.legend()
# plt.show()


# ### Part 3: Enhanced Cluster Drift Monitoring & Visualization

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import adjusted_rand_score

# def monitor_cluster_drift(historical_clusters: pd.DataFrame) -> pd.Series:
#     """
#     Calculates the monthly cluster drift (ARI) relative to the previous month.

#     Args:
#         historical_clusters (pd.DataFrame): A DataFrame where the index is the date
#                                             and columns are tickers, with values being
#                                             the cluster labels for that month.

#     Returns:
#         pd.Series: A Series of monthly ARI scores, indexed by date.
#     """
#     ari_scores = {}
    
#     # Iterate through the dates, starting from the second date
#     for i in range(1, len(historical_clusters.index)):
#         current_date = historical_clusters.index[i]
#         previous_date = historical_clusters.index[i-1]
        
#         # Get the cluster assignments for the current and previous periods
#         labels_current = historical_clusters.loc[current_date].dropna()
#         labels_previous = historical_clusters.loc[previous_date].dropna()
        
#         # Find the common tickers between the two periods to ensure a fair comparison
#         common_tickers = labels_current.index.intersection(labels_previous.index)
        
#         # Skip calculation if there are not enough common tickers for a meaningful score
#         if len(common_tickers) < 2:
#             ari_scores[current_date] = np.nan
#             continue

#         # Calculate the Adjusted Rand Index for the common tickers
#         ari = adjusted_rand_score(labels_current[common_tickers], labels_previous[common_tickers])
#         ari_scores[current_date] = ari
        
#     return pd.Series(ari_scores, name='Monthly_ARI_Score')

# # --- Main Execution ---
# # Let's assume 'monthly_clusters_df' is a pre-existing DataFrame with your clustering results.
# # For a runnable example, we will create a mock DataFrame.
# # mock_dates = pd.to_datetime(pd.date_range('2022-12-31', periods=12, freq='M'))
# # mock_tickers = [f'Stock_{i}' for i in range(50)]
# # mock_data = {date: np.random.randint(0, 4, len(mock_tickers)) for date in mock_dates}
# # monthly_clusters_df = pd.DataFrame(mock_data, index=mock_tickers).T

# # --- Run the monitoring function ---
# cluster_drift_series = monitor_cluster_drift(monthly_clusters_df)

# # --- Visualize the Drift Monitoring Results ---
# plt.figure(figsize=(15, 7))
# cluster_drift_series.plot(kind='bar', color='skyblue', label='Monthly ARI Score')

# # Define and plot an alert threshold
# drift_threshold = 0.6
# plt.axhline(
#     y=drift_threshold, 
#     color='r', 
#     linestyle='--', 
#     label=f'Drift Alert Threshold ({drift_threshold})'
# )

# # Set plot titles and labels
# plt.title('Monthly Cluster Stability Monitoring (Adjusted Rand Index)', fontsize=16)
# plt.ylabel('ARI Score (vs. Previous Month)')
# plt.xlabel('Date')
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# print("\n--- ARI Score Series (Preview) ---")
# print(cluster_drift_series.head())
