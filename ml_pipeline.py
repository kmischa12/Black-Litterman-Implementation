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
    
    # separate features (X) and target (y)
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

    # PHASE 4- TRANSLATING THE FORECASTS TO BLACK-LITTERMAN VIEWS 
    # 1. Calculate the Covariance Matrix (Risk) based  on the past
    historical_returns = train_data.pivot(columns='Ticker', values='Target_Future_3M_Ret').dropna()
    cov_matrix = historical_returns.cov().values # Raw numpy array
    tickers_list = test_data['Ticker'].values
    num_assets = len(tickers_list)

    # Confidence estimation: calculate Information Coefficient 
    # get dates for last 36 months of training data 
    last_36_months_dates = train_data.index.unique()[-36:]
    recent_train_data = train_data[train_data.index.isin(last_36_months_dates)]

    # Ask the model to predict the last 36 months
    recent_X = recent_train_data[feature_cols]
    recent_y_actual = recent_train_data['Target_Future_3M_Ret']
    recent_preds = ml_model.predict(recent_X)

    # Calculate IC (Pearson Correlation between predictions and actuals)
    # np.corrcoef returns a matrix, we just want the single correlation value [0, 1]
    ic_matrix = np.corrcoef(recent_preds, recent_y_actual)
    ic = ic_matrix[0, 1] 
    
    # Handle math edge cases (if correlation fails)
    if np.isnan(ic): 
        ic = 0.0

    # Map IC to Confidence Percentage 
    if ic <= 0.00:
        c = 0.10
    elif ic <= 0.05:
        c = 0.30
    elif ic <= 0.10:
        c = 0.50
    elif ic <= 0.15:
        c = 0.70
    else: # 0.20+
        c = 0.90

    # CONSTRUCT VIEWS (MATRIX P AND VECTOR Q)
    # Sort the ML predictions to find what the AI thinks is Best and Worst
    sorted_preds = month_results.sort_values(by='ML_Forecast_Return', ascending=False)
    
    num_views = 3 # 3 relative views (Top 3 vs Bottom 3)
    P_matrix = np.zeros((num_views, num_assets))
    Q_vector = np.zeros(num_views)
    
    for i in range(num_views):
        long_ticker = sorted_preds.iloc[i]['Ticker']          # AI's Favorite
        short_ticker = sorted_preds.iloc[-(i+1)]['Ticker']    # AI's Least Favorite
        
        # Find exactly where these tickers live in our columns
        long_idx = np.where(tickers_list == long_ticker)[0][0]
        short_idx = np.where(tickers_list == short_ticker)[0][0]
        
        # Build the [1, -1] View in Matrix P
        P_matrix[i, long_idx] = 1.0  # +1 for the winner
        P_matrix[i, short_idx] = -1.0 # -1 for the loser
        
        # Build Vector Q (The predicted outperformance spread)
        pred_long = sorted_preds.iloc[i]['ML_Forecast_Return']
        pred_short = sorted_preds.iloc[-(i+1)]['ML_Forecast_Return']
        Q_vector[i] = pred_long - pred_short

        # CALCULATE THE OMEGA MATRIX (Ω)
        # Omega = (P * Cov * P_T) * ((1 - c) / c)
        # If c is high (90%), the multiplier becomes (0.1 / 0.9) = 0.11 (Very low doubt)
        # If c is low (10%), the multiplier becomes (0.9 / 0.1) = 9.0 (High doubt)
    
        P_Sigma_P_T = np.dot(np.dot(P_matrix, cov_matrix), P_matrix.T)
    
        # add a tiny bit of noise (1e-6) to the diagonal to avoid division by zero
        # if the model predicts two assets exactly the same
        Omega_matrix = P_Sigma_P_T * ((1.0 - c) / c) + (np.eye(num_views) * 1e-6)

# compile all the monthly prediction tables together
final_forecasts = pd.concat(all_predictions).set_index(['Date', 'Ticker'])

print("\n--- PHASE 2 & 3 COMPLETE ---")
print("The model has successfully recorded its predictions.")
print("\nHere is what it predicted for the very first test month (Jan 2015):")
print(final_forecasts.loc["2015-01-31"])