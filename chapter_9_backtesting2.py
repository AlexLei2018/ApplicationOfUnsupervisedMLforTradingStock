# chapter_9_full_backtest_workflow_modular.py
# This script combines the logic from Chapters 7, 8, and 9 into a single,
# modular and extensible workflow using real market data. (Final Version: Direct Plot Display)

import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Tuple

# --- Suppress FutureWarnings for a cleaner output ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION ---
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'V', 'UNH', 
    'JNJ', 'WMT', 'PG', 'XOM', 'CVX', 'LLY', 'META', 'BAC', 'KO', 'PFE', 'PEP'
]
SP500_TICKER = '^GSPC'
START_DATE = '2015-01-01'
END_DATE = '2023-12-31'
INITIAL_CASH = 1_000_000

# --- MODULE 1: DATA FETCHING AND SIGNAL GENERATION ---

def generate_static_signals(
    prices: pd.DataFrame, 
    lookback_period: int = 63, 
    top_n: int = 5
) -> pd.DataFrame:
    """Generates a static signal matrix based on a momentum strategy."""
    print("\n--- Generating Static Signals (Momentum Strategy) ---")
    monthly_prices = prices.resample('ME').last()
    momentum = monthly_prices.pct_change(periods=lookback_period // 21).dropna()

    signals = pd.DataFrame(index=prices.index, columns=prices.columns)
    
    for i in range(len(momentum)):
        start_period = momentum.index[i] + pd.DateOffset(days=1)
        end_period = start_period + pd.DateOffset(months=1) - pd.DateOffset(days=1)
        top_performers = momentum.iloc[i].nlargest(top_n).index
        signals.loc[start_period:end_period, top_performers] = 1.0 / top_n
        
    signals.ffill(inplace=True)
    signals.fillna(0, inplace=True)
    
    print("Static signals generated successfully.")
    return signals

def generate_dynamic_signals(
    static_signals_df: pd.DataFrame, 
    sp500_prices: pd.DataFrame
) -> pd.DataFrame:
    """Detects market regimes using HMM and applies them to the static signals."""
    print("\n--- Detecting Market Regimes and Generating Dynamic Signals ---")
    
    returns = sp500_prices['Close'].pct_change().dropna()
    volatility = returns.rolling(window=21).std().dropna()
    hmm_features = pd.DataFrame({'returns': returns, 'volatility': volatility}).dropna()
    
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(hmm_features)
    
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42, min_covar=1e-3)
    model.fit(feature_matrix)
    hidden_states = model.predict(feature_matrix)
    
    regimes_df = hmm_features.copy()
    regimes_df['Regime'] = hidden_states
    
    profile_df = pd.DataFrame([{'State': i, 'Mean_Return': model.means_[i][0]} for i in range(3)]).set_index('State')
    sorted_profiles = profile_df.sort_values(by='Mean_Return')
    sorted_indices = sorted_profiles.index.tolist()
    regime_mapping = {
        sorted_indices[0]: 'Bearish',
        sorted_indices[2]: 'Bullish',
        sorted_indices[1]: 'Neutral/Volatile'
    }
    regimes_df['Regime_Name'] = regimes_df['Regime'].map(regime_mapping)
    print("Market regimes detected successfully.")
    
    exposure_rules = {'Bullish': 1.0, 'Neutral/Volatile': 0.5, 'Bearish': 0.0}
    daily_exposure = regimes_df['Regime_Name'].map(exposure_rules)
    
    aligned_static, aligned_exposure = static_signals_df.align(daily_exposure, join='left', axis=0)
    aligned_exposure.ffill(inplace=True)
    
    dynamic_signals_df = aligned_static.multiply(aligned_exposure, axis=0)
    print("Dynamic signals generated successfully.")
    
    return dynamic_signals_df

# --- MODULE 2: BACKTESTING ---

def run_backtests(
    prices: pd.DataFrame,
    static_signals: pd.DataFrame,
    dynamic_signals: pd.DataFrame
) -> Tuple[vbt.Portfolio, vbt.Portfolio]:
    """
    Runs backtests for both strategies and returns the portfolio objects.
    """
    print("\n--- Running Backtest Comparison ---")
    
    aligned_prices, aligned_static = prices.align(static_signals, join='inner', axis=0)
    _, aligned_dynamic = aligned_prices.align(dynamic_signals, join='left', axis=0)

    static_pf = vbt.Portfolio.from_signals(
        aligned_prices, aligned_static, freq='D', init_cash=INITIAL_CASH, 
        fees=0.001, slippage=0.001, cash_sharing=True
    )
    dynamic_pf = vbt.Portfolio.from_signals(
        aligned_prices, aligned_dynamic, freq='D', init_cash=INITIAL_CASH, 
        fees=0.001, slippage=0.001, cash_sharing=True
    )
    print("Backtesting complete. Portfolio objects are ready for analysis.")
    return static_pf, dynamic_pf

# --- MODULE 3: ANALYSIS AND PLOTTING ---

def print_stats_comparison(static_pf: vbt.Portfolio, dynamic_pf: vbt.Portfolio):
    """Prints a side-by-side comparison of key performance indicators."""
    stats_df = pd.concat([
        static_pf.stats().rename('Static Strategy'),
        dynamic_pf.stats().rename('Dynamic Strategy')
    ], axis=1)
    
    desired_metrics = [
        'Total Return [%]', 'Annualized Return [%]',
        'Annualized Volatility [%]', 'Max Drawdown [%]',
        'Sharpe Ratio', 'Calmar Ratio'
    ]
    available_metrics = stats_df.index.intersection(desired_metrics)
    
    print("\n--- Side-by-Side Comparison of Key Performance Indicators ---")
    print(stats_df.loc[available_metrics].round(4))

def plot_cumulative_returns(static_pf: vbt.Portfolio, dynamic_pf: vbt.Portfolio):
    """Plots the cumulative returns comparison chart directly."""
    print(f"\nGenerating and displaying cumulative returns plot...")
    
    combined_returns = pd.concat([
        static_pf.cumulative_returns(),
        dynamic_pf.cumulative_returns()
    ], axis=1)
    combined_returns.columns = ['Static (Alpha Only)', 'Dynamic (Alpha + Beta)']
    
    fig_returns = combined_returns.vbt.plot()
    
    fig_returns.update_layout(
        title_text='Cumulative Returns: Static vs. Dynamic Strategy',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        xaxis_type='date'
    )
    
    # --- FIX: Replace file saving with direct display ---
    fig_returns.show()
    # --------------------------------------------------

def plot_drawdowns(static_pf: vbt.Portfolio, dynamic_pf: vbt.Portfolio):
    """Plots the drawdown comparison chart directly."""
    print(f"\nGenerating and displaying drawdowns plot...")
    
    static_value = static_pf.value()
    dynamic_value = dynamic_pf.value()
    
    static_dd = (static_value / static_value.cummax() - 1) * 100
    dynamic_dd = (dynamic_value / dynamic_value.cummax() - 1) * 100
    
    combined_dd = pd.concat([static_dd, dynamic_dd], axis=1)
    combined_dd.columns = ['Static (Alpha Only)', 'Dynamic (Alpha + Beta)']
    
    fig_dd = combined_dd.vbt.plot()

    fig_dd.update_layout(
        title_text='Drawdowns: Static vs. Dynamic Strategy',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        xaxis_type='date'
    )
    
    # --- FIX: Replace file saving with direct display ---
    fig_dd.show()
    # --------------------------------------------------

# --- MAIN WORKFLOW ORCHESTRATOR ---

def main():
    """Main function to orchestrate the entire workflow."""
    try:
        # Step 1: Fetch Data
        print(f"Fetching price data for {len(TICKERS)} stocks and S&P 500 from {START_DATE} to {END_DATE}...")
        price_data = yf.download(TICKERS + [SP500_TICKER], start=START_DATE, end=END_DATE, progress=False)
        
        if isinstance(price_data.columns, pd.MultiIndex):
            stock_prices = price_data['Open'].drop(columns=SP500_TICKER, errors='ignore')
            sp500_prices = price_data.xs(SP500_TICKER, level=1, axis=1)
        else:
            stock_prices = price_data[[c for c in price_data.columns if c != SP500_TICKER]]
            sp500_prices = pd.DataFrame(price_data[SP500_TICKER])
            sp500_prices.columns = ['Close']
            for col in ['Open', 'High', 'Low']:
                if col not in sp500_prices.columns:
                    sp500_prices[col] = sp500_prices['Close']

        # Step 2: Generate Signals
        static_signals = generate_static_signals(stock_prices)
        dynamic_signals = generate_dynamic_signals(static_signals, sp500_prices)
        
        # Step 3: Run Backtests and retain the results
        static_pf, dynamic_pf = run_backtests(stock_prices, static_signals, dynamic_signals)
        
        # Step 4: Analyze and Plot using the retained results
        print_stats_comparison(static_pf, dynamic_pf)
        plot_cumulative_returns(static_pf, dynamic_pf)
        plot_drawdowns(static_pf, dynamic_pf)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == '__main__':
    main()