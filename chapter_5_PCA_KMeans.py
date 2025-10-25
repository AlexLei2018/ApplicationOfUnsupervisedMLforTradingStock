### PCA + K-Means Market Profile
# filename : chapter_5_pca_KMeans.py

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import requests # Import the requests library

# --- Suppress some warnings ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Step 1: Get Real Stock Data and Features ---

def get_sp100_tickers_info():
    """Fetches the S&P 100 stock list and their sector information, handling potential HTTP errors."""
    try:
        # --- CORE FIX: Define a User-Agent to mimic a browser ---
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        url = 'https://en.wikipedia.org/wiki/S%26P_100'
        
        # Use requests to get the page content with the header
        response = requests.get(url, headers=headers)
        response.raise_for_status() # This will raise an error if the request failed
        
        # Now, have pandas read the HTML content from the response text
        payload = pd.read_html(response.text)
        
        info_df = payload[2][['Symbol', 'GICS Sector']]
        info_df['Symbol'] = info_df['Symbol'].str.replace('.', '-', regex=False)
        print(f"Successfully fetched info for {len(info_df)} stocks from Wikipedia.")
        return info_df
    except Exception as e:
        print(f"Failed to fetch data from Wikipedia: {e}.")
        return None

# --- The rest of the code remains exactly the same ---

def get_financial_features(tickers, start_date="2022-01-01", end_date="2023-12-31"):
    """Downloads data and calculates a series of financial features."""
    print(f"Downloading data for {len(tickers)} stocks...")
    all_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    if all_data.empty:
        print("Data download failed, please check your network or ticker symbols.")
        return None, None
        
    adj_close = all_data['Adj Close'].dropna(axis=1)
    
    print("Calculating financial features...")
    momentum_12m = adj_close.pct_change(252).iloc[-1]
    returns = adj_close.pct_change()
    volatility_60d = returns.rolling(60).std().iloc[-1]
    
    beta_values, pe_ratios, dividend_yields, market_caps, valid_tickers = [], [], [], [], []

    for ticker in tqdm(adj_close.columns, desc="Fetching fundamental data"):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            beta = info.get('beta')
            pe = info.get('trailingPE')
            dy = info.get('dividendYield')
            mc = info.get('marketCap')
            
            if all(v is not None for v in [beta, pe, dy, mc]):
                beta_values.append(beta)
                pe_ratios.append(pe)
                dividend_yields.append(dy)
                market_caps.append(mc)
                valid_tickers.append(ticker)
        except Exception:
            continue
            
    raw_feature_matrix = pd.DataFrame({
        'Momentum_12M': momentum_12m,
        'Volatility_60D': volatility_60d,
        'Beta': beta_values,
        'PE_Ratio': pe_ratios,
        'Dividend_Yield': dividend_yields,
        'Market_Cap': market_caps
    }, index=valid_tickers)
    
    raw_feature_matrix.dropna(inplace=True)
    
    print(f"\nSuccessfully built a feature matrix for {len(raw_feature_matrix)} stocks.")
    return raw_feature_matrix, adj_close.columns

# Execute the full pipeline
info_df = get_sp100_tickers_info()
if info_df is not None:
    raw_feature_matrix, valid_tickers = get_financial_features(info_df['Symbol'].tolist())
    
    if raw_feature_matrix is not None and not raw_feature_matrix.empty:
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(raw_feature_matrix)

        print("\n--- 1. Reducing dimensionality with PCA ---")
        n_components = 4
        pca_cluster = PCA(n_components=n_components, random_state=42)
        pca_features = pca_cluster.fit_transform(feature_matrix_scaled)
        pca_features_df = pd.DataFrame(pca_features, index=raw_feature_matrix.index, columns=[f'PC_{i+1}' for i in range(n_components)])
        print(f"Reduced {feature_matrix_scaled.shape[1]} original features to {n_components} dimensions.")
        
        print("\n--- 2. Running K-Means on the low-dimensional PCA feature space ---")
        k = 5
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(pca_features)
        raw_feature_matrix['Cluster'] = cluster_labels
        print(f"\nStocks have been grouped into {k} clusters.")

        print("\n--- 3. Profiling each cluster ---")
        cluster_profile = raw_feature_matrix.groupby('Cluster').mean()
        print("\nAverage Feature Profile for Each Cluster:")
        styled_profile = cluster_profile.style.background_gradient(cmap='viridis').format("{:,.2f}")
        display(styled_profile)

        print("\n--- 4. Visualizing the clustering results ---")
        pca_features_df['Cluster'] = cluster_labels
        plt.figure(figsize=(12, 9))
        plot_data = pca_features_df.join(info_df.set_index('Symbol'))
        scatter = sns.scatterplot(
            x='PC_1', y='PC_2', hue='Cluster', style='GICS Sector',
            data=plot_data, palette='viridis', s=100, alpha=0.8
        )
        plt.title('K-Means Clustering Results after PCA (Real S&P 100 Data)')
        plt.xlabel('Principal Component 1 (PC_1)')
        plt.ylabel('Principal Component 2 (PC_2)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)
        plt.tight_layout()
        plt.show()