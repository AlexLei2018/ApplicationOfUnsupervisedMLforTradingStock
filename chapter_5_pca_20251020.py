import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Get data--- 
# We use a similar process to Chapter 3 to get the S&P 100 component stock data .
#  # In order to ensure the stability of PCA, we need a long and continuous period of data. 
tickers  =  [ 'AAPL' ,  'MSFT' , '  GOOGL ' , '  AMZN ' , 'NVDA '  , ' TSLA' , 'META' , 'JPM' , 'V' , 'JNJ' , ' XOM ' , 'CVX ' , 'PG ' , 'KO' , 'PFE' , 'MRK' , 'BAC' , 'WFC' , 'CSCO' , 'INTC' ] 

# Example stock pool 
start_date = "2021-01-01" 
end_date = "2023-12-31"      

all_data = yf.download(tickers, start=start_date, end=end_date)

# --- 2. Construct the return matrix --- 
# We use the adjusted closing price ('Adj Close'), which more accurately reflects the true return 
adj_close  =  all_data [ 'Close' ]

# Use .pct_change() to calculate daily returns 
# (today's price - previous day's price) / previous day's price 
daily_returns  =  adj_close . pct_change ()

# --- 3. Data cleaning and preprocessing --- 
# .pct_change() will generate NaN in the first row, which needs to be removed 
daily_returns  =  daily_returns . dropna ()

# Check if there are other NaNs (maybe because some stocks were suspended during the period) 
# A robust approach is to fill in the missing data for a certain day with 0 (assuming no trading on that day, the return is 0) 
daily_returns  =  daily_returns.fillna ( 0 )

print ( "--- Stock Daily Return Matrix (Preview) ---" ) 
print ( f"Matrix shape: { daily_returns . shape } " ) 
print ( "Rows represent dates, columns represent stocks." ) 
print ( daily_returns . head ())

# --- 4. (Important) Standardize Return Data --- 
# Although PCA does not require data standardization, if the volatility (variance) of different stocks varies greatly, 
# then the stocks with high volatility will dominate the PCA. 
# Standardizing (making them have a mean of 0 and a variance of 1) allows each stock to have the same initial weight in the analysis. 
scaler  =  StandardScaler () 
scaled_returns  =  scaler . fit_transform ( daily_returns )

scaled_returns_df = pd.DataFrame(scaled_returns, 
                                 index=daily_returns.index, 
                                 columns=daily_returns.columns)

print ( " \n --- Standardized return matrix (preview) ---" ) 
print ( scaled_returns_df . head ())

# --- 2. Constructing the return matrix ---
# We use the adjusted closing price ('Adj Close'), which more accurately reflects the actual return
adj_close = all_data['Close']

# Use .pct_change() to calculate the daily return
# (Today's price - Previous day's price) / Previous day's price
daily_returns = adj_close.pct_change()

# --- 3. Data cleaning and preprocessing ---
# .pct_change() will generate NaN in the first row, which needs to be removed
daily_returns = daily_returns.dropna()

# Check if there are other NaNs (maybe because some stocks were suspended during the period)
# A robust approach is to fill in the missing stock data for a certain day with 0 (assuming no trading on that day, the return is 0)
daily_returns = daily_returns.fillna(0)

print("--- Stock Daily Return Matrix (Preview) ---")
print(f"Matrix shape: {daily_returns.shape}")
print("Rows represent dates, columns represent stocks.")
print(daily_returns.head())

# --- 4. (Important) Normalize the return data ---
# Although PCA does not require data normalization, if the volatility (variance) of different stocks varies significantly,
# then the highly volatile stocks will dominate the PCA analysis.
# Normalizing (making them have a mean of 0 and a variance of 1) gives each stock the same initial weight in the analysis.
scaler = StandardScaler()
scaled_returns = scaler.fit_transform(daily_returns)

scaled_returns_df = pd.DataFrame(scaled_returns,
index=daily_returns.index,
columns=daily_returns.columns)

print("\n--- Standardized return matrix (preview) ---")
print(scaled_returns_df.head())

# --- 5. Initialize and Execute PCA ---

# Initialize the PCA model
# n_components can be an integer representing the number of principal components we want to retain.
# If left blank or set to None, all principal components are retained by default (min(n_samples, n_features))
# We retain all principal components initially so we can analyze their importance later.
pca = PCA()

# Fit the PCA model using the standardized return data
# .fit() will calculate all principal components (directions) and related statistics.
pca.fit(scaled_returns_df)

print("\nPCA model fitted.")

# --- 6. Analyze Explained Variance Ratio ---

# The pca.explained_variance_ratio_ attribute returns an array
# Each value in the array corresponds to the explained variance ratio of a principal component (sorted from largest to smallest)
explained_variance_ratio = pca.explained_variance_ratio_

print(f"\nExplained variance ratio of the first 5 principal components:")
for i, ratio in enumerate(explained_variance_ratio[:5]):
    print(f"PC {i+1}: {ratio:.4f} (i.e., {ratio*100:.2f}%)")

# Calculate the cumulative explained variance ratio
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

print(f"\nCumulative explained variance ratio of the first 5 principal components:")
for i, cum_ratio in enumerate(cumulative_explained_variance[:5]):
    print(f"Cumulative explained variance for the first {i+1} PCs: {cum_ratio:.4f} (i.e., {cum_ratio*100:.2f}%)")

# --- Visualize explained variance ---
plt.figure(figsize=(12, 6))

# Plot the explained variance for each PC (bar chart)
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.6,
align='center', label='Individual explained variance')

# Plot the cumulative explained variance (line chart)
plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid',
         label='Cumulative explained variance')

plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.legend(loc='best')
plt.title('Explained Variance by Principal Components')
plt.grid(True)
plt.show()

# --- 7. Obtaining Principal Component Composition (Factor Loadings) ---
# The pca.components_ attribute is an array of (n_components, n_features)
# Each row represents a principal component, and each column corresponds to a feature (stock).
# This array defines how each principal component is constructed from a linear combination of the original stock returns.
components_df = pd.DataFrame(pca.components_,
columns=scaled_returns_df.columns,
index=[f'PC_{i+1}' for i in range(pca.n_components_)])

print("\n--- Principal Component Composition (Factor Loadings) Preview (First 5 PCs) ---")
# We transpose (.T) so that the stocks are rows for easier viewing.
# print(components_df.head().T)

# --- 8. Obtaining Time Series Performance of Principal Components (Factor Returns) ---
# The .transform() method projects the original data onto the new principal component coordinate system.
# The result is the specific value of each principal component on each day.
factor_returns = pca.transform(scaled_returns_df)

factor_returns_df = pd.DataFrame(factor_returns,
index=scaled_returns_df.index,
columns=[f'PC_{i+1}' for i in range(pca.n_components_)])

print("\n--- Factor Return Matrix (Preview) ---")
print("Rows represent dates, columns represent the daily returns for each principal component")
print(factor_returns_df.head())

########################################################
# ---9 Verify that PC1 represents market beta ---
# Obtain the SPY (S&P 500 ETF) returns over the same period as a market benchmark
spy_returns = yf.download('SPY', start=start_date, end=end_date)['Close'].pct_change().dropna()

# Combine the PC1 factor returns and SPY returns into a single DataFrame
comparison_df = pd.DataFrame({
    'PC1_Factor_Return': factor_returns_df['PC_1'],
    'SPY_Return': spy_returns['SPY']
}).dropna()

# Note: The direction of the PCs produced by PCA is arbitrary; PC1 may move in the same or opposite direction as SPY.
# If the correlation is negative, we can multiply it by -1 to correct the direction and make it more intuitive.
correlation = comparison_df.corr().loc['PC1_Factor_Return', 'SPY_Return']
if correlation < 0:
    print("Correlation is negative, reversing the factor direction...")
    comparison_df['PC1_Factor_Return'] *= -1
print("Correlation coefficient between PC1 and SPY returns:", comparison_df.corr())
# Plot the cumulative return curves for comparison
# We use the cumulative product of (1 + return) to calculate the two curves.
# comparison_df here contains the original values.
cumulative_returns = (1 + comparison_df).cumprod()

# --- 10 Plotting with Dual Y-Axes ---
# Create a figure and primary axes (ax1)
fig, ax1 = plt.subplots(figsize=(12, 7))

# Set Matplotlib to display Chinese characters and minus signs properly
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- Plot the Main Curve (PC1 Factor) ---
# ax1 plots the cumulative curve for PC1 and uses its own Y-axis scale (left side)
color_pc1 = 'tab:blue'
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Cumulative PC1 Factor Value (Raw Value)', color=color_pc1, fontsize=12)
ax1.plot(cumulative_returns.index, cumulative_returns['PC1_Factor_Return'], color=color_pc1, label='Cumulative PC1 Factor (left axis)')
ax1.tick_params(axis='y', labelcolor=color_pc1)

# --- Plotting the secondary curve (SPY returns) ---
# Create a secondary axes (ax2) that shares the x-axis
ax2 = ax1.twinx()
# ax2 plots the SPY cumulative curve and uses its own y-axis scale (right)
color_spy = 'tab:red'
ax2.set_ylabel('Cumulative SPY returns (raw values, benchmark=1)', color=color_spy, fontsize=12)
ax2.plot(cumulative_returns.index, cumulative_returns['SPY_Return'], color=color_spy, linestyle='--', label='Cumulative SPY Returns (Right Axis)')
ax2.tick_params(axis='y', labelcolor=color_spy)

# --- Add a chart title and unified legend ---
plt.title('Comparison of Cumulative Curves of PC1 Factor and SPY Returns (Raw Values)', fontsize=16)

# Combine the legends for both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

fig.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# --- 11.  Calucate the correlation Before plotting code
correlation_value = comparison_df['PC1_Factor_Return'].corr(comparison_df['SPY_Return'])
print(f"The correlation between the daily returns of the PC1 factor and SPY is: {correlation_value: .4f}")

# Assuming you already have factor returns for all principal components (factor_returns_df)
# Compare PC2 and SPY as well
comparison_pc2_df = pd.DataFrame({
    'PC2_Factor_Return': factor_returns_df['PC_2'],
    'SPY_Return': spy_returns['SPY']
    })

correlation_pc2 = comparison_pc2_df.corr().loc['PC2_Factor_Return', 'SPY_Return']
print(f"The correlation between the daily returns of PC2 and SPY is: {correlation_pc2:.4f}")

# ---12 Constructing a style timing indicator (using PC2 as an example) ---
if 'PC_2' in factor_returns_df.columns:
    pc2_factor_returns = factor_returns_df['PC_2']

    # Again, let's correct the direction, assuming we want positive values ​​to represent "growth"
    # You'll need to decide whether to reverse based on your factor loadings analysis
    # if growth_stocks_have_negative_loadings:
    # pc2_factor_returns *= -1

    pc2_moving_avg_short = pc2_factor_returns.rolling(window=20).mean()
    pc2_moving_avg_long = pc2_factor_returns.rolling(window=60).mean()

    pd.DataFrame({
        'PC2_Factor_Return': pc2_factor_returns.cumsum(),
        'Short_MA (20d)': pc2_moving_avg_short.cumsum(), 
        'Long_MA (60d)': pc2_moving_avg_long.cumsum() 
        }).plot(figsize=(12, 6), title='Style Timing Indicator based on PC2 (e.g., Growth vs. Value)') 
    plt.ylabel('Cumulative Factor Return') 
    plt.grid(True) 
    plt.show()



