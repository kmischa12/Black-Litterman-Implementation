import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
import warnings
warnings.filterwarnings('ignore') # Suppress minor math warnings during the loop

print("Loading daily prices from Phase 1...")
# Read the CSV and tell Pandas the first column is a Date
prices = pd.read_csv("stitched_prices.csv", index_col=0, parse_dates=True)

print("Calculating daily volatility...")
# calculate volatility before converting to monthly data to capture the daily swings
daily_returns = prices.pct_change() # percentage change from previous day
# standard deviation reveals the swings in returns
vol_20d = daily_returns.rolling(window=20).std() * np.sqrt(252) # standard deviation for 20 day window, annualized
vol_60d = daily_returns.rolling(window=60).std() * np.sqrt(252) # 60 day window 

print("Resampling data to end-of-month...")
# 'ME' stands for Month End. This shrinks our giant daily dataset down to just 1 row per month
monthly_prices = prices.resample('ME').last()
monthly_vol_20d = vol_20d.resample('ME').last()
monthly_vol_60d = vol_60d.resample('ME').last()

# list to hold the data for all 14 assets
all_assets_data = []

print("Engineering features and targets...")
for ticker in monthly_prices.columns:
    
    # Create a temporary dataframe just for this one ticker
    df = pd.DataFrame(index=monthly_prices.index)
    df['Ticker'] = ticker
    
    # FEATURES - X
    # pct_change(1) looks back 1 row. pct_change(12) looks back 12 rows (a full year).
    df['Mom_1M'] = monthly_prices[ticker].pct_change(1)
    df['Mom_3M'] = monthly_prices[ticker].pct_change(3)
    df['Mom_6M'] = monthly_prices[ticker].pct_change(6)
    df['Mom_12M'] = monthly_prices[ticker].pct_change(12)
    
    df['Vol_20D'] = monthly_vol_20d[ticker]
    df['Vol_60D'] = monthly_vol_60d[ticker]
    
    # TARGET - y
    # pct_change(3) gives a 3-month return, 
    # but .shift(-3) mathematically pulls that future answer back in time to align with today
    df['Target_Future_3M_Ret'] = monthly_prices[ticker].pct_change(3).shift(-3)
    
    all_assets_data.append(df)

# Smash all 14 temporary dataframes together into one dataset
ml_data = pd.concat(all_assets_data)

# Drop any rows that have missing data (the first 12 months won't have 12M momentum, 
# and the very last 3 months won't have a future answer)
ml_data = ml_data.dropna()

# Sort by Date 
ml_data = ml_data.sort_index()

print("\n--- PHASE 1 COMPLETE ---")
print(f"Dataset shape: {ml_data.shape[0]} rows ready for Machine Learning.")
print("\nHere is a sneak peek at your new ML-ready data:")
print(ml_data.head())


print("\n--- STARTING PHASE 2 & 3: THE EXPANDING WINDOW LOOP ---")

# Get a chronological list of every single month-end date in our dataset
all_months = ml_data.index.unique().sort_values()

# we will start testing the model in Jan 2015
test_start_date = pd.to_datetime("2015-01-31") 

# store all monthly predictions here
all_predictions = []

# Initialize the Machine Learning model
# alpha and l1_ratio are standard "tuning knobs" for Elastic Net
ml_model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)

print(f"Starting backtest from {test_start_date.date()}...")
print("Training and predicting... (this might take a few seconds)")

# 2. THE LOOP 
for current_test_month in all_months:
    
    # Skip months before 2015 (we need 2000-2014 strictly for the initial training block)
    if current_test_month < test_start_date:
        continue
        
    # --- Prevent Look-Ahead Bias ---
    # The Past: Strictly everything BEFORE the current test month
    train_data = ml_data[ml_data.index < current_test_month]
    
    # The Present: ONLY the current test month
    test_data = ml_data[ml_data.index == current_test_month]
    
    if train_data.empty or test_data.empty:
        continue
        
    # Features (X): momentum and volatility
    feature_cols = ['Mom_1M', 'Mom_3M', 'Mom_6M', 'Mom_12M', 'Vol_20D', 'Vol_60D']
    
    X_train = train_data[feature_cols]
    y_train = train_data['Target_Future_3M_Ret']
    
    X_test = test_data[feature_cols]
    
    # TRAIN THE MODEL from scratch each time 
    ml_model.fit(X_train, y_train)
    
    # predict the 3 month return forecast given today's features 
    predictions = ml_model.predict(X_test)
    
    # Store the predictions in a clean table for this specific month
    month_results = pd.DataFrame({
        'Date': current_test_month,
        'Ticker': test_data['Ticker'].values,
        'ML_Forecast_Return': predictions
    })
    
    all_predictions.append(month_results)

# compile all the monthly prediction tables together
final_forecasts = pd.concat(all_predictions).set_index(['Date', 'Ticker'])

print("\n--- PHASE 2 & 3 COMPLETE ---")
print("The model has successfully recorded its predictions.")
print("\nHere is what it predicted for the very first test month (Jan 2015):")
print(final_forecasts.loc["2015-01-31"])