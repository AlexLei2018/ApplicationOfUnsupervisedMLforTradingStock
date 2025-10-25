# quant_utils.py
# Core Toolbox Filename: quant_utils.py
# Description: A universal toolbox of utility functions for quantitative trading.
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import umap.umap_ as umap
from hmmlearn.hmm import GaussianHMM
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import hashlib # Import the hashlib library for hashing

# ==============================================================================
# --- 0. Get nasdaq100 tickers ---
# ==============================================================================
import pandas as pd
import requests # We need the 'requests' library to add a user agent

def get_nasdaq100_tickers() -> list:
    """
    Dynamically scrapes the current list of Nasdaq 100 constituent tickers from Wikipedia.
    
    This version includes a User-Agent header to prevent HTTP 403 Forbidden errors.

    Returns:
        list: 
            A list of strings, where each string is a Nasdaq 100 ticker symbol.
            If scraping fails, it prints an error and returns a small fallback list.
            
    Raises:
        ImportError: If the 'lxml' or 'requests' library is not installed.
    """
    print("Fetching the current list of Nasdaq 100 tickers from Wikipedia...")
    try:
        # --- The Key Fix is Here ---
        # 1. Define a User-Agent header to mimic a web browser.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        }
        
        # 2. Define the URL.
        url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
        
        # 3. Use the 'requests' library to get the page content with the header.
        response = requests.get(url, headers=headers)
        response.raise_for_status() # This will raise an exception for bad status codes (4xx or 5xx)

        # 4. Use pandas.read_html to parse the HTML content from the response object.
        tables = pd.read_html(response.text)
        # --- End of Fix ---
        
        # Find the correct table by looking for the 'Ticker' column
        tickers_df = None
        for table in tables:
            if 'Ticker' in table.columns:
                tickers_df = table
                break

        if tickers_df is None:
            raise RuntimeError("Could not find the constituents table with a 'Ticker' column on the page.")

        # The ticker symbols are in the 'Ticker' column
        tickers = tickers_df['Ticker'].tolist()
        
        # Make tickers compatible with yfinance (e.g., 'BRK.B' -> 'BRK-B')
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        print(f"Successfully fetched {len(tickers)} tickers.")
        return tickers
        
    except ImportError:
        raise ImportError(
            "This function requires 'lxml' and 'requests'. Please install them by running 'pip install lxml requests'."
        )
    except Exception as e:
        print(f"Error: Failed to fetch tickers. Reason: {e}")
        # Return a fallback list in case of failure
        fallback_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'PEP', 'COST', 'AVGO']
        print(f"Operation failed. Returning a fallback list of {len(fallback_tickers)} tickers.")
        return fallback_tickers

# ==============================================================================
# --- 1. Data Acquisition and Management ---
# ==============================================================================

def get_stock_data(tickers, start_date, end_date, cache_dir='cache'):
    """
    Downloads stock data from Yahoo Finance and uses a local cache to improve speed.
    Args:
        tickers (list): A list of stock ticker symbols.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        cache_dir (str): The directory to store cached data files.

    Returns:
        pd.DataFrame: A DataFrame containing the OHLCV data.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Sanitize tickers for filename
    # 1. Create a unique, long string representing the ticker list.
    #    Sorting ensures ['AAPL', 'GOOG'] and ['GOOG', 'AAPL'] produce the same hash.
    ticker_str = '_'.join(sorted(tickers))
    
    # 2. Use SHA-256 to hash this long string into a short, fixed-length fingerprint.
    #    .encode('utf-8') is necessary before hashing.
    #    .hexdigest() gives us the final string.
    #    [:16] truncates it to 16 characters, which is more than enough for uniqueness here.
    ticker_hash = hashlib.sha256(ticker_str.encode('utf-8')).hexdigest()[:16]
    
    # 3. Create the final, short filename.
    filename = f"{ticker_hash}_{start_date}_{end_date}.pkl"
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path):
        print(f"Loading data from cache: {cache_path}")
        return pd.read_pickle(cache_path)
    else:
        print(f"Downloading data for {len(tickers)} stocks from the web...")
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
        data.to_pickle(cache_path)
        print("Data has been downloaded and cached.")
        return data
    
# ==============================================================================
# --- 2. Feature Engineering ---
# ==============================================================================
# --- Feature Calculation Functions ---
def calculate_roc(close, n):
    """
    Calculates the n-period Rate of Change (ROC) for a given price series.

    The Rate of Change (ROC) indicator is a momentum oscillator that measures the 
    percentage change in price between the current price and the price `n` periods ago.

    Parameters
    ----------
    close : pd.Series
        A pandas Series of closing prices. The index should be a datetime index.
        
    n : int
        The number of periods for the lookback window (e.g., 12 for 12 days).

    Returns
    -------
    pd.Series
        A pandas Series containing the calculated ROC values. The first `n`
        values will be NaN, as there is not enough historical data for the calculation.

    Notes
    -----
    Potential for ArithmeticError: The calculation involves division. If a price `n` 
    periods ago was 0, this will cause a division-by-zero error (ZeroDivisionError, 
    which is a type of ArithmeticError). It is crucial to ensure the input `close` 
    series does not contain zeros, or to handle them appropriately before calling 
    this function.

    Examples:
    # Calculate ROC, short-term (21 days), medium-term (63 days), and long-term (126 days)
    roc_21 = calculate_roc(close_prices, 21)
    """
    
    # The ROC formula is: (Current Price - Price n periods ago) / (Price n periods ago)
    
    # `close.diff(n)` calculates the numerator: (Current Price - Price n periods ago).
    numerator = close.diff(n)
    
    # `close.shift(n)` gets the price from n periods ago to be used as the denominator.
    # THIS IS THE SOURCE OF POTENTIAL ERRORS. If this value is 0, division will fail.
    denominator = close.shift(n)
    
    return numerator / denominator

def calculate_cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the daily cross-sectional percentile rank (from 0 to 1).
    A higher value indicates a higher rank.
    """
    return df.rank(axis=1, pct=True, method='first')

def calculate_atr_percent(high, low, close, n=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range_df = pd.concat([tr1, tr2, tr3], axis=1)
    true_range = true_range_df.max(axis=1, skipna=False)
    atr = true_range.ewm(span=n, adjust=False).mean()
    return (atr / close)

# Builds a complete, multi-dimensional feature matrix for the entire time series.

def create_full_feature_matrix(ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a complete, multi-dimensional feature matrix for the entire time series.

    Args:
        ohlcv_data (pd.DataFrame): A MultiIndex DataFrame containing OHLCV data for multiple stocks.

    Returns:
        pd.DataFrame: A clean panel data DataFrame with a (Date, Ticker) MultiIndex and feature columns.
    """
    print("Starting to build the full historical feature matrix...")
    
    # Ensure data integrity
    if 'Close' not in ohlcv_data.columns:
        raise ValueError("Input data must contain a 'Close' column.")
        
    close_prices = ohlcv_data['Close']
    returns = np.log(close_prices / close_prices.shift(1))
    
    # --- 1. Calculate all time-series features ---
    # Use a dictionary to store the calculated feature DataFrames
    features = {
        'volatility_60d': returns.rolling(window=60, min_periods=30).std() * np.sqrt(252),
        'momentum_21d': close_prices.pct_change(21),
        'momentum_63d': close_prices.pct_change(63),
        'momentum_126d': close_prices.pct_change(126)
    }

    # --- 2. Calculate cross-sectional features ---
    features['rel_strength_21d'] = calculate_cross_sectional_rank(features['momentum_21d'])

    # --- 3. Combine all features into a panel data format ---
    # This is a more efficient and correct way to transform the data structure.
    # First, convert each feature DataFrame into a Series with a (Date, Ticker) MultiIndex.
    panel_data = pd.concat(
        [df.stack() for df in features.values()], 
        axis=1, 
        keys=features.keys()
    )
    panel_data.index.names = ['Date', 'Ticker'] # Name the index levels
    
    print("Full Feature matrix construction completed.")
    # Drop any rows with NaN values resulting from the lookback windows
    return panel_data.dropna()

# Extracts a feature snapshot for a specific date from the full feature panel.
def get_feature_snapshot_for_date(feature_panel: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """
    Extracts a feature snapshot for a specific date from the full feature panel.

    Args:
        feature_panel (pd.DataFrame): The panel data DataFrame with a (Date, Ticker) MultiIndex.
        target_date (str): The date for the desired snapshot in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with tickers as the index and features as columns for the given date.
                      Returns an empty DataFrame if the date is not found or has no data.
    """
    try:
        # pd.to_datetime ensures the input string is converted to a proper timestamp for matching
        date_timestamp = pd.to_datetime(target_date)
        
        # .loc is the key to selecting from the first level of the MultiIndex
        snapshot = feature_panel.loc[date_timestamp]
        
        if snapshot.empty:
            print(f"Warning: Data for date {target_date} exists but is empty after processing.")
        
        return snapshot
        
    except KeyError:
        print(f"Error: Date {target_date} not found in the feature panel. It might be a non-trading day or outside the data range.")
        return pd.DataFrame() # Return an empty DataFrame on failure


# Scales the features in a stock feature snapshot using StandardScaler.

def scale_feature_snapshot(feature_snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Scales the features in a stock feature snapshot using StandardScaler.

    This function transforms the data so that each feature (column) has a mean of 0 
    and a standard deviation of 1. This is a crucial preprocessing step for many
    algorithms that are sensitive to the scale of input features, such as K-Means,
    PCA, and SVMs.

    Args:
        feature_snapshot (pd.DataFrame): 
            A DataFrame where the index consists of tickers and the columns 
            represent the calculated features (e.g., momentum, volatility).

    Returns:
        pd.DataFrame: 
            A new DataFrame with the same shape, index, and columns as the input, 
            but with the values scaled. Returns an empty DataFrame if the input is empty.
    """
    # Handle the edge case where the input DataFrame is empty
    if feature_snapshot.empty:
        print("Warning: Input feature_snapshot is empty. Returning an empty DataFrame.")
        return pd.DataFrame()

    # 1. Initialize the StandardScaler object.
    scaler = StandardScaler()

    # 2. Use .fit_transform() to calculate the mean and std deviation for each
    #    feature (column) and then apply the scaling transformation.
    #    The result is a NumPy array, which loses the original index and columns.
    scaled_data = scaler.fit_transform(feature_snapshot)

    # 3. Reconstruct the DataFrame.
    #    Create a new pandas DataFrame from the scaled NumPy array, making sure to
    #    restore the original index (tickers) and column names (features).
    snapshot_scaled = pd.DataFrame(
        scaled_data, 
        index=feature_snapshot.index, 
        columns=feature_snapshot.columns
    )

    return snapshot_scaled

def get_clusters_snapshot(features_df: pd.DataFrame, k: int) -> pd.Series:
    """
    Applies K-Means clustering to the input features after standardization.

    Args:
        features_df (pd.DataFrame): DataFrame of features for clustering.
        k (int): The number of clusters.

    Returns:
        pd.Series: A Series containing the cluster labels for each row
                   in the input features_df, with the same index.
    """
    if features_df.empty:
        return pd.Series([], index=[])

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init for robust centroid initialization
    cluster_labels = kmeans.fit_predict(scaled_features)

    return pd.Series(cluster_labels, index=features_df.index)

# The build_market_regime_features function from the previous answer is correct
# and does not need to be changed again. It will now receive a clean DataFrame.
def build_market_regime_features(tickers, start_date, end_date):
    """
    Builds a macroeconomic feature matrix for identifying market regimes.
    """
    print("Starting to build the market regime feature matrix...")
    
    # --- 1. Fetch Raw Data ---
    # S&P 500 Index
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)[['Close']]
    # VIX Index
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)[['Close']]
    # High-Yield Bond ETF and Treasury Bond ETF (for credit spread)
    hyg = yf.download('HYG', start=start_date, end=end_date, progress=False)[['Close']]
    ief = yf.download('IEF', start=start_date, end=end_date, progress=False)[['Close']]
    # Constituent stock data (for breadth)
    stock_prices = yf.download(tickers, start=start_date, end=end_date, progress=False)[['Close']]
    
    # --- 2. Calculate Features ---
    # Volatility Features
    feature_vix = vix.copy()
    feature_vix.columns = ['VIX']
    
    feature_vol_20d = sp500.pct_change().rolling(20).std()
    feature_vol_20d.columns = ["SP500_Vol_20d"]
    
    # Momentum Features
    feature_roc_63d = (sp500.diff(63) / sp500.shift(63))
    feature_roc_63d.columns = ["SP500_ROC_63d"]
    
    feature_roc_252d = (sp500.diff(252) / sp500.shift(252))
    feature_roc_252d.columns = ["SP500_ROC_252d"]
    
    # Breadth Feature
    advances = stock_prices.diff() > 0
    feature_breadth = pd.DataFrame((advances.sum(axis=1) / stock_prices.count(axis=1)))
    feature_breadth.columns = ["Breadth_Adv%"]
    
    # Risk Appetite Feature (Credit Spread)
    # We use the difference in returns to approximate the change in spread.
    hyg_returns = hyg.pct_change()
    ief_returns = ief.pct_change()
    feature_spread = (hyg_returns['Close']['HYG'] - ief_returns['Close']['IEF']).rolling(20).mean()
    feature_spread.columns = "HYG-IEF_Spread_20dMA"

    # --- 3. Combine and Clean ---
    regime_features = pd.concat([
        feature_vix, 
        feature_vol_20d, 
        feature_roc_63d, 
        feature_roc_252d, 
        feature_breadth, 
        feature_spread
    ], axis=1)
    
    # Clean up NaNs produced by rolling indicators
    regime_features = regime_features.dropna()
    
    print("Market regime feature matrix built successfully.")
    return regime_features

# ==============================================================================
# --- 3. Unsupervised Learning Models ---
# ==============================================================================

def perform_kmeans_clustering(
    scaled_feature_snapshot: pd.DataFrame, 
    k: int
    ) -> pd.DataFrame:
    """
    Performs K-Means clustering on a pre-scaled feature snapshot for a single day.

    This function takes a DataFrame of scaled features, applies the K-Means algorithm
    to group the stocks into 'k' clusters, and returns the cluster assignments.

    Args:
        scaled_feature_snapshot (pd.DataFrame):
            A DataFrame where the index consists of tickers and the columns are the 
            scaled features for one specific point in time. It is crucial that this
            data has already been scaled (e.g., using StandardScaler).
            
        k (int):
            The desired number of clusters (the 'k' in K-Means).

    Returns:
        pd.Series:
            A pandas Series where the index is the tickers and the values are the
            integer cluster labels (from 0 to k-1). Returns an empty Series if
            the input is invalid.
    """
    # --- Input Validation ---
    # Check if the input DataFrame is empty.
    if scaled_feature_snapshot.empty:
        print("Warning: Input scaled_feature_snapshot is empty. Returning an empty Series.")
        return pd.Series(dtype=int)
        
    # K-Means cannot create more clusters than there are samples (stocks).
    if len(scaled_feature_snapshot) < k:
        print(f"Warning: Number of stocks ({len(scaled_feature_snapshot)}) is less than k ({k}). "
              f"Cannot form {k} clusters. Returning an empty Series.")
        return pd.Series(dtype=int)

    # --- K-Means Model Initialization ---
    # 1. Create an instance of the KMeans model.
    #    - n_clusters=k: This is the most important parameter.
    #    - n_init='auto': Recommended setting to avoid FutureWarnings in scikit-learn.
    #    - random_state=42: This is ESSENTIAL for reproducibility. It ensures that
    #      you get the exact same cluster assignments every time you run the code
    #      on the same data.
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)

    # --- Model Fitting and Prediction ---
    # 2. Use .fit_predict() on the scaled data. This single method performs two steps:
    #    - fit(): It learns the optimal positions of the 'k' cluster centers.
    #    - predict(): It assigns each stock (row) to the nearest cluster center.
    #    The output is a NumPy array of cluster labels.
    labels = kmeans.fit_predict(scaled_feature_snapshot)

    # --- Format the Output ---
    # 3. Create a pandas Series to associate the labels with their tickers.
    #    - The data is the NumPy array of 'labels'.
    #    - The index is taken directly from the input DataFrame's index (the tickers).
    #    - Giving the Series a name is good practice.
    # cluster_labels = pd.Series(labels, index=scaled_feature_snapshot.index, name='Cluster')
    
    scaled_feature = scaled_feature_snapshot.copy()
    scaled_feature['cluster_label'] = labels
    
    return scaled_feature




def filter_anomalies_from_universe(universe_tickers, feature_matrix_scaled, contamination=0.1):
    """
    Uses Isolation Forest to remove anomalous stocks from a given universe.

    Args:
        universe_tickers (list): The initial list of stock tickers.
        feature_matrix_scaled (pd.DataFrame): The scaled feature matrix for all stocks.
        contamination (float): The expected proportion of anomalies in the data.

    Returns:
        list: A filtered list of "safe" tickers.
    """
    if not universe_tickers or len(universe_tickers) < 10:
        return universe_tickers # Not enough data to reliably detect anomalies

    universe_features = feature_matrix_scaled.loc[universe_tickers]
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    predictions = iso_forest.fit_predict(universe_features)
    
    safe_tickers = np.array(universe_tickers)[predictions == 1].tolist()
    return safe_tickers

def select_diversified_portfolio(universe_tickers, daily_returns_df, n_portfolio=10):
    """
    Uses PCA to select a highly diversified portfolio from a stock universe.

    Args:
        universe_tickers (list): The list of candidate stocks.
        daily_returns_df (pd.DataFrame): DataFrame of daily returns for all stocks.
        n_portfolio (int): The desired number of stocks in the final portfolio.

    Returns:
        list: A diversified list of stock tickers.
    """
    if not universe_tickers: return []
    if len(universe_tickers) <= n_portfolio: return universe_tickers

    pca_returns = daily_returns_df[universe_tickers].dropna()
    # Need sufficient data for stable PCA
    if len(pca_returns) < 60:
        print(" - Warning: Insufficient data for PCA optimization. Truncating portfolio.")
        return universe_tickers[:n_portfolio]
    
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(pca_returns)
    
    n_components = min(n_portfolio, len(universe_tickers))
    pca = PCA(n_components=n_components)
    pca.fit(scaled_returns)
    
    final_portfolio, selected_indices = [], []
    for i in range(n_components):
        component = pca.components_[i]
        # Find the stock with the highest absolute loading on this component that hasn't been picked yet
        for idx in np.abs(component).argsort()[::-1]:
            if idx not in selected_indices:
                selected_indices.append(idx)
                final_portfolio.append(pca_returns.columns[idx])
                break
    return final_portfolio

# ==============================================================================
# --- 4. Advanced Models (NLP, etc.) ---
# ==============================================================================

def get_text_embeddings(texts):
    """
    Converts a list of texts into embedding vectors using a SentenceTransformer model.

    Args:
        texts (list): A list of strings.

    Returns:
        np.ndarray: An array of embedding vectors.
    """
    print("Loading language model and encoding texts...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def get_text_clusters(embeddings, min_cluster_size=2):
    """
    Performs UMAP dimensionality reduction and HDBSCAN clustering on text embeddings.

    Args:
        embeddings (np.ndarray): The array of text embedding vectors.
        min_cluster_size (int): The minimum size of a cluster for HDBSCAN.

    Returns:
        np.ndarray: An array of cluster labels.
    """
    print("Performing dimensionality reduction with UMAP...")
    # Dynamically adjust n_neighbors to be less than the number of samples
    n_neighbors = max(2, min(15, int(embeddings.shape[0] * 0.1)))
    
    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0, random_state=42)
    umap_embeddings = umap_reducer.fit_transform(embeddings)
    
    print("Performing clustering with HDBSCAN...")
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(umap_embeddings)
    return cluster_labels