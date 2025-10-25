import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm # Import a powerful statistical model library

# --- Replicate the core data from Section 5.2.2 ---
# (To allow the code block to run independently, we will quickly replicate it here)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'JNJ',
'XOM', 'CVX', 'PG', 'KO', 'PFE', 'MRK', 'BAC', 'WFC', 'CSCO', 'INTC']
start_date = "2021-01-01"
end_date = "2023-12-31"

all_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
daily_returns = all_data['Close'].pct_change().dropna()

scaler = StandardScaler()
scaled_returns = scaler.fit_transform(daily_returns)
pca = PCA(n_components=5) # We only use the top 5 most important factors
factor_returns = pca.fit_transform(scaled_returns)
factor_returns_df = pd.DataFrame(factor_returns,
index=daily_returns.index,
columns=[f'PC_{i+1}' for i in range(5)])

# --- Build a risk factor model for AAPL ---
print("--- 1. Building a Risk Factor Model for a Single Stock (AAPL) ---")

# Prepare Y (dependent variable) and X (independent variable) for the regression
# Y: Apple's excess return (raw return is used here for simplicity)
Y = daily_returns['AAPL']
# X: The first five factor returns we extracted
X = factor_returns_df[['PC_1', 'PC_2', 'PC_3', 'PC_4', 'PC_5']]
# StatsModels requires us to manually add a constant term (intercept), which represents Alpha
X = sm.add_constant(X)

# Perform OLS (Ordinary Least Squares) regression
model = sm.OLS(Y, X).fit()

# Print a detailed regression report
print("\nAAPL Regression results for the first five PCA factors:")
print(model.summary())

# Extract key results
alpha = model.params['const'] * 252 # Annualized Alpha
beta_pc1 = model.params['PC_1']
r_squared = model.rsquared

print(f"Model Interpretation:")
print(f" - Annualized Alpha (α): {alpha: 0.4%}")
print(f" - Exposure to the Market Factor (PC1) (β1): {beta_pc1: 0.4f}")
print(f" - R-squared (Model Explanatory Power): {r_squared: 0.2%}")
if model.pvalues['const'] < 0.05:
    print(" - Alpha is statistically significant at the 5% level.")
else:
    print(" - Alpha is not statistically significant at the 5% level.")


# --- 2. Portfolio Risk Decomposition--- 
print ( "--- 2. Portfolio Risk Decomposition---" )

# 1. Create a sample portfolio (e.g., one with a tech bias) 
portfolio_weights  =  pd . Series ({ 
    'AAPL' :  0.2 ,  'MSFT' :  0.2 ,  'NVDA' :  0.2 ,  # High weight in tech 
    'JPM' :  0.1 ,  'BAC' :  0.1 ,               # Low weight in financials 
    'XOM' :  0.1 ,  'CVX' :  0.1 ,  'PG' :  0.1      # Other diversified holdings 
    }) 
portfolio_weights  =  portfolio_weights  /  portfolio_weights . sum ()  # Ensure weights are normalized

# 2. Calculate the daily return of the portfolio 
# This is a matrix multiplication: (daily_returns matrix [N*M] . weight vector [M*1]) -> portfolio daily return [N*1] 
portfolio_returns  =  daily_returns [ portfolio_weights . index ] . dot ( portfolio_weights )

# 3. Calculate the portfolio's risk exposure to each factor (Portfolio Betas) 
# The factor exposure of a portfolio is the weighted average of the factor exposures of all stocks in the portfolio 
# We need to first calculate the factor exposure (betas) for each stock in the portfolio 
all_betas  =  {} 
for  stock  in  portfolio_weights . index : 
    Y_stock  =  daily_returns [ stock ] 
    X_stock  =  sm . add_constant ( factor_returns_df ) 
    model_stock  =  sm . OLS ( Y_stock ,  X_stock ) . fit () 
    all_betas [ stock ]  =  model_stock . params [ 1 :]  # Only take beta, not alpha

beta_df = pd.DataFrame(all_betas)

# Portfolio beta = (stock beta matrix [K*M] . weight vector [M*1]) -> portfolio beta vector [K*1] 
portfolio_betas  =  beta_df . dot ( portfolio_weights ) 
print ( "Portfolio Betas for each PCA factor:" ) 
print ( portfolio_betas )

# 4. Perform risk decomposition (Variance Decomposition) 
print ( "Risk Decomposition: " ) 
# Covariance matrix of factor returns 
factor_cov_matrix  =  factor_returns_df . cov ()  *  252  # Annualized 
# Total variance of the portfolio (annualized) 
total_portfolio_variance  =  portfolio_returns . var ()  *  252

# (a) Systematic Risk Contributed by Factors 
# Formula: β' * F * β (β is the portfolio beta vector, F is the factor covariance matrix) 
systematic_variance  =  portfolio_betas . T . dot ( factor_cov_matrix ) . dot ( portfolio_betas ) 
print ( f" - Total portfolio variance (annualized): {total_portfolio_variance:.6f} " ) 
print ( f" - Systematic variance explained by PCA factors: {systematic_variance:.6f} ({systematic_variance/total_portfolio_variance:.2%}) " )