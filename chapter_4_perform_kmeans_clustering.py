#### Run perform_kmeans_clustering

import quant_utils as qtu
import matplotlib.pyplot as plt
import seaborn as sns

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

    # 3. Get feature_snapshot
    # target_trading_day = "2023-03-11"
    target_trading_day = full_feature_matrix.index.get_level_values('Date').max()
    feature_snapshot = qtu.get_feature_snapshot_for_date(full_feature_matrix, target_trading_day)

    #4. Normalized feature_snapshot
    feature_snapshot_scaled = qtu.scale_feature_snapshot(feature_snapshot)

    # 5. Define the desired number of clusters.
    num_clusters = 3
    print(f"\n--- Performing K-Means with k = {num_clusters} ---")

    # 6. Call our function to get the cluster assignments.
    stock_clusters = qtu.perform_kmeans_clustering(feature_snapshot_scaled, k=num_clusters)

    # 7. Inspect the results.
    print("\n--- Output: Cluster Assignments for each Stock ---")
    print(stock_clusters)
    print ( f"\n --- K-Means clustering completed (K= { num_clusters } ) ---" ) 
    print ( "Distribution of the number of stocks in each cluster:" ) 
    print ( stock_clusters['cluster_label'].value_counts().sort_index())

    # --- 8. Results Interpretation and Visualization --- 
    #8.1 Feature Analysis: Calculate the average "profile" of each cluster 
    feature_snapshot_with_clusters =     feature_snapshot.copy()
    feature_snapshot_with_clusters['cluster_label'] = stock_clusters['cluster_label']
    cluster_profile  =  feature_snapshot_with_clusters.groupby( 'cluster_label' ).mean()
    print ( " \n --- Average feature profile of each cluster ---" ) 
    print ( cluster_profile )

    # 8.2 Use heatmap for visualization 
    plt.figure ( figsize = ( 12 ,  6 )) 
    sns.heatmap ( cluster_profile ,  annot = True ,  cmap = 'coolwarm' ,  fmt = '.2f' ) 
    plt.title ( 'Average feature heatmap of each cluster (tribal portrait)' ) 
    plt.show ()


    

 



