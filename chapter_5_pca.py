import quant_utils as qtu 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA   
   
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
    
    daily_returns  =  ohlcv_data[ 'Close' ].pct_change().dropna(how = 'all' ).fillna ( 0 )

    print ( "--- Stock Daily Return Matrix (Preview) ---" ) 
    print ( daily_returns . head ())

    # --- 2. (Important) Standardize return data --- 
    # Standardization allows each stock to have the same initial weight in the analysis 
    scaler  =  StandardScaler () 
    scaled_returns  =  scaler.fit_transform(daily_returns)

    # --- 3. Perform PCA decomposition --- 
    # We first retain all principal components so that we can analyze their importance 
    pca =  PCA  ( ) 
    pca.fit ( scaled_returns )

    print( " \n PCA model fitting completed." )

    # --- 4. Analyze the importance of principal components (explained variance ratio) --- 
    explained_variance_ratio  =  pca.explained_variance_ratio_ 
    cumulative_explained_variance  =  np.cumsum(explained_variance_ratio )

    print( "\n --- Explained variance ratio of the first 5 principal components ---" ) 
    for  i ,  ratio  in  enumerate ( explained_variance_ratio [: 5 ]): 
        print ( f"PC { i + 1 } : { ratio : 0.2%} " )

    print ( " \n --- Cumulative variance explained ratio ---" ) 

    for  i ,  cum_ratio  in  enumerate ( cumulative_explained_variance [: 5 ]): 
        print (f"Cumulative explanation of the first{i + 1} PCs:{cum_ratio:.2%} " )

    # --- 5. Visualize explained variance --- 
    plt . figure ( figsize = ( 12 ,  6 )) 
    plt . bar ( range ( 1 ,  len ( explained_variance_ratio )  +  1 ),  explained_variance_ratio ,  alpha = 0.6 ,  label = 'Explained variance of a single PC' ) 
    plt . step ( range ( 1 ,  len ( cumulative_explained_variance )  +  1 ),  cumulative_explained_variance ,  where = 'mid' ,  label = 'Cumulative explained variance' ) 
    plt . ylabel ( 'Explained variance proportion' ) 
    plt . xlabel ( 'Principal component index' ) 
    plt . title ( 'PCA explained variance proportion' ) 
    plt . legend ( loc = 'best' ) 
    plt . show ()
