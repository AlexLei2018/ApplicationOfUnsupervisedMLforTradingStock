# sanity_check.py
import pandas as pd 
# import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
import quant_utils as qtu # Import our toolbox 

def run_sanity_check(): 
    """     
    Run a quick environment test to ensure all core libraries are working properly.    
    """ 
    print ( "--- Starting quick environment test---" ) 

    # --- 1. Use quant_utils to get data --- 
    tickers  =  [ 'AAPL' ,  'GOOGL' ,  'MSFT' ,  'AMZN' ,  'NVDA' ] 
    start_date  =  '2022-01-01' 
    end_date  =  '2023-12-31'
    
    # This function automatically handles caching 
    price_data  =  qtu.get_stock_data ( tickers ,  start_date ,  end_date ) 
    daily_returns  =  price_data['Close'].pct_change().dropna()

    if  daily_returns.empty : 
        print ( "Error: Failed to obtain data, please check network connection or yfinance." ) 
        return

    # --- 2. PCA using scikit-learn --- 
    print ( " \nPerforming PCA on daily returns..." ) 
    scaler  =  StandardScaler () 
    scaled_returns  =  scaler.fit_transform ( daily_returns )
    
    pca  =  PCA ( n_components = 2 ) 
    principal_components  =  pca . fit_transform ( scaled_returns )
    
    print(f"PCA's first component explains {pca.explained_variance_ratio_[0]:.2%} of variance (Market Factor).")
    print(f"PCA's second component explains {pca.explained_variance_ratio_[1]:.2%} of variance (Style Factor).")

    # --- 3. Visualization using matplotlib & seaborn--- 
    print ( " \ nGenerating visualization chart..." ) 
    plt . figure ( figsize = ( 10 ,  6 )) 
    sns . scatterplot ( x = principal_components [:,  0 ],  y = principal_components [:,  1 ]) 
    plt . title ( 'PCA of Tech Stocks Daily Returns (Sanity Check)' ) 
    plt . xlabel ( 'Principal Component 1' ) 
    plt . ylabel ( 'Principal Component 2' ) 
    plt . grid ( True ) 
    plt . show ()

    print ( " \n --- Environment test successful! All core libraries are ready. ---" )

if  __name__  ==  '__main__' : 
    run_sanity_check ()