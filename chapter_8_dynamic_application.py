import pandas as pd
import numpy as np
from typing import Dict

def apply_dynamic_exposure(
    static_signals_df: pd.DataFrame,
    market_regimes_df: pd.DataFrame,
    exposure_mapping: Dict[str, float]
) -> pd.DataFrame:
    """
    Applies a dynamic market exposure level to a static signal matrix based on market regimes.

    This function acts as a "rules engine" or an "automatic transmission" for a trading
    strategy. It scales the overall portfolio exposure up or down depending on the
    prevailing market condition (e.g., Bullish, Bearish), transforming a static
    set of signals into a dynamically managed portfolio.

    Args:
        static_signals_df (pd.DataFrame): A DataFrame where rows are dates and columns
            are assets, containing the static signal weights (e.g., from -1 to 1).
        market_regimes_df (pd.DataFrame): A DataFrame with a DatetimeIndex and a
            'Regime_Name' column indicating the market regime for each day.
        exposure_mapping (Dict[str, float]): A dictionary mapping regime names to target
            exposure levels (e.g., {'Bullish': 1.0, 'Bearish': 0.0}).

    Returns:
        pd.DataFrame: A new DataFrame of dynamic signals, where the original static
            signals have been scaled by the daily market exposure factor.
    """
    # 1. Generate the daily target exposure Series from the market regimes.
    daily_target_exposure = market_regimes_df['Regime_Name'].map(exposure_mapping)

    # 2. Align the static signals with the daily exposure Series.
    aligned_static_signals, aligned_exposure = static_signals_df.align(
        daily_target_exposure, join='left', axis=0
    )

    # 3. Forward-fill the exposure to cover non-trading days (e.g., weekends).
    aligned_exposure.ffill(inplace=True)

    # 4. Apply the dynamic exposure to the static signals using broadcast multiplication.
    dynamic_signals_df = aligned_static_signals.multiply(aligned_exposure, axis=0)
    
    print("\n--- Dynamic signal matrix generation completed ---")

    return dynamic_signals_df

# --- Main Execution Block for Demonstration ---
if __name__ == '__main__':
    # This block demonstrates how to use the function.
    # It creates dummy data to simulate the outputs from previous analyses.

    # --- Setup: Create Dummy Data ---
    dates_static = pd.to_datetime(pd.date_range('2023-10-01', '2023-10-31', freq='B'))
    static_signals_df = pd.DataFrame(
        data=np.random.rand(len(dates_static), 3) * 0.66,
        index=dates_static,
        columns=['Asset_A', 'Asset_B', 'Asset_C']
    )
    
    dates_regime = pd.to_datetime(pd.date_range('2023-10-01', '2023-10-31', freq='D'))
    regimes = np.random.choice(['Bullish', 'Neutral/Volatile', 'Bearish'], size=len(dates_regime))
    market_regimes_df = pd.DataFrame(
        data={'Regime_Name': regimes},
        index=dates_regime
    )

    # --- Step 1: Define the Rule Engine (Mapping Relationship) ---
    exposure_rules = {
        'Bullish': 1.0,
        'Neutral/Volatile': 0.5,
        'Bearish': 0.0
    }

    # --- Step 2: Apply the rules engine to generate the dynamic signals ---
    dynamic_signals = apply_dynamic_exposure(
        static_signals_df=static_signals_df,
        market_regimes_df=market_regimes_df,
        exposure_mapping=exposure_rules
    )

    # --- Step 3: Display and Verify Results ---
    total_dynamic_exposure = dynamic_signals.sum(axis=1)

    print("\nPreview of total exposure of the dynamic signal matrix:")
    print(total_dynamic_exposure.tail(10))

    # --- Verification Example ---
    try:
        bearish_day = market_regimes_df[market_regimes_df['Regime_Name'] == 'Bearish'].index[0]
        
        # --- FIX IS HERE: Using the more compatible get_indexer() method ---
        # Find the integer position of the closest preceding business day
        loc = static_signals_df.index.get_indexer([bearish_day], method='ffill')[0]
        # Get the actual index (date) at that location
        closest_business_day = static_signals_df.index[loc]
        # --- END OF FIX ---
        
        print(f"\n--- Verifying a 'Bearish' day ({bearish_day.date()}) ---")
        print(f"The total portfolio exposure on the corresponding trading day "
              f"({closest_business_day.date()}) is: {total_dynamic_exposure.loc[closest_business_day]:.2f}")
    except (IndexError, KeyError):
        print("\n--- No 'Bearish' day found in the random sample to verify. ---")