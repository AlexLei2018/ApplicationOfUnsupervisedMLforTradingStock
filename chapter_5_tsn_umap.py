import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Prepare data ---
def get_sp100_tickers_info():
    """Get S&P 100 stock and industry information"""
    try:
    # Add lxml check
        try:
            import lxml
        except ImportError:
            print("Please install the lxml library first: pip install lxml")
            return None

        payload = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')
        info_df = payload[2][['Symbol', 'GICS Sector']]
        info_df['Symbol'] = info_df['Symbol'].str.replace('.', '-', regex=False)
        print(f"Successfully retrieved {len(info_df)} stock information from Wikipedia.")
        return info_df.set_index('Symbol')

    except Exception as e:
        print(f"Failed to retrieve data from Wikipedia: {e}. A small alternative dataset will be used.")
        return pd.DataFrame({
            'GICS Sector': ['Information Technology', 'Information Technology', 'Communication Services', 'Consumer Discretionary', 'Information Technology', 'Consumer Discretionary', 'Financials', 'Health Care']},
            index=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'JNJ'] 
        )

info_df = get_sp100_tickers_info()
if info_df is None: 
    pass # If lxml is not installed, skip
else: 
    tickers = info_df.index.tolist() 
    feature_matrix_snapshot = pd.DataFrame(np.random.rand(len(tickers), 8), index=tickers) 
    feature_matrix_snapshot.dropna(inplace=True) 
    info_df = info_df.loc[feature_matrix_snapshot.index] 

    scaler = StandardScaler() 
    feature_matrix_scaled = scaler.fit_transform(feature_matrix_snapshot) 
    n_samples = feature_matrix_scaled.shape[0] 

    # --- 2. [Core Correction] Dynamically adjust the perplexity parameter for t-SNE ---
    print(f"\nNumber of samples in the dataset is: {n_samples}")
    perplexity_value = min(30, n_samples - 1)

    # Ensure perplexity_value > 1
    if perplexity_value <= 1:
        print(f"Number of samples ({n_samples}) is too small to perform t-SNE. At least 2 samples are required.")
    else:
        print(f"Dynamically setting t-SNE perplexity to: {perplexity_value}")
    print("Performing t-SNE and UMAP dimensionality reduction...")

    tsne = TSNE(n_components=2, perplexity=perplexity_value, max_iter=1000, random_state=42)
    features_tsne = tsne.fit_transform(feature_matrix_scaled)

    # Dynamically adjust UMAP's n_neighbors
    n_neighbors_value = min(15, n_samples - 1)
    if n_neighbors_value < 2: n_neighbors_value = 2 # The minimum n_neighbors for UMAP is 2

    reducer = umap.UMAP(n_neighbors=n_neighbors_value, min_dist=0.1, n_components=2, random_state=42)
    features_umap = reducer.fit_transform(feature_matrix_scaled)

    # --- 3. Create a DataFrame for plotting ---
    df_plot = pd.DataFrame(index=feature_matrix_snapshot.index)
    df_plot['TSNE1'] = features_tsne[:, 0]
    df_plot['TSNE2'] = features_tsne[:, 1] 
    df_plot['UMAP1'] = features_umap[:, 0] 
    df_plot['UMAP2'] = features_umap[:, 1] 
    df_plot['Sector'] = info_df['GICS Sector'] 

    # --- 4. Visual comparison --- 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10)) 
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
    sns.set_style('whitegrid') 

    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Sector', data=df_plot, palette='viridis', s=80, ax=ax1) 
    ax1.set_title('Market Map via t-SNE', fontsize=16) 
    ax1.legend(title='Sector') 

    sns.scatterplot(x='UMAP1', y='UMAP2', hue='Sector', data=df_plot, palette='viridis', s=80, ax=ax2) 
    ax2.set_title('Market Map via UMAP', fontsize=16) 
    ax2.legend(title='Sector') 

    plt.suptitle('Comparing t-SNE and UMAP for Market Structure Visualization', fontsize=20) 
    plt.show()