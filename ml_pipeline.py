import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from scipy.optimize import minimize
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
    
    # The actual real-world return of the asset over the next 1 month
    df['Actual_Next_1M_Ret'] = monthly_prices[ticker].pct_change(1).shift(-1)
    
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


print("\n--- STARTING PHASE 2 & 3: THE EXPANDING WINDOW LOOP ---")

# Get a chronological list of every single month-end date in our dataset
all_months = ml_data.index.unique().sort_values()

# we will start testing the model in Jan 2015
test_start_date = pd.to_datetime("2015-01-31") 

# store all monthly predictions here
all_predictions = []
all_portfolio_weights = []

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

    # PHASE 5: RUN THE OPTIMIZER FOR ASSET ALLOCATION 
    # 1. Market Baseline 
    w_mkt = np.ones(num_assets) / num_assets # initializes array with each stock weighted equally
    risk_aversion = 2.0
    tau = 0.05
    Pi = risk_aversion * np.dot(cov_matrix, w_mkt) #baseline of returns 

    # 2. Black-Litterman Posterior Math
    tau_cov_inv = np.linalg.inv(tau * cov_matrix) #precision of market 
    Omega_inv = np.linalg.inv(Omega_matrix) # precision of AI's views 
    
    M_inverse = tau_cov_inv + np.dot(np.dot(P_matrix.T, Omega_inv), P_matrix) #adds market's precision to AI's precision
    # P_matrix.T translates it to 12 individual stocks 
    posterior_cov = np.linalg.inv(M_inverse) # updated risk matrix 
    
    term1 = np.dot(tau_cov_inv, Pi) # pull of baseline: market's baseline returns times the volatility/ precision
    term2 = np.dot(np.dot(P_matrix.T, Omega_inv), Q_vector) # pull of AI: AI's views' returns times the precision
    posterior_returns = np.dot(posterior_cov, (term1 + term2)) #combined pull (term 1 + term 2) / total precision

    #3. constrained optimization (scipy)
    def objective_function(weights):
        port_return = np.dot(weights, posterior_returns)
        port_variance = np.dot(weights.T, np.dot(posterior_cov, weights))
        utility = port_return - (risk_aversion / 2) * port_variance # score is our return - risk 
        return -utility
    
    # CONSTRAINTS: Fully Invested (sum to 100%) & Max 45% per asset
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}) # sum - 1 must be zero (fully invest)
    bounds = tuple((0.0, 0.45) for _ in range(num_assets))


    # Run the Optimizer
    opt_result = minimize(objective_function, w_mkt, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Store the final Portfolio Weights for this month
    weight_results = pd.DataFrame({
        'Date': current_test_month,
        'Ticker': tickers_list,
        'Optimal_Weight': opt_result.x,
        'Equal_Weight': w_mkt,
        'Actual_Return': test_data['Actual_Next_1M_Ret'].values
    })
    all_portfolio_weights.append(weight_results)

# END OF LOOP

print("\n--- PHASE 5 COMPLETE ---")

#PHASE 6: PERFORMANCE METRICS 
print("Simulating Portfolio Returns and generating Report Card...")
df_performance = pd.concat(all_portfolio_weights)

# 1. Calculate Realized Monthly Return for the Strategy vs. Baseline
# Formula: Weight assigned * Actual Next Month Return
df_performance['ML_Cont'] = df_performance['Optimal_Weight'] * df_performance['Actual_Return']
df_performance['EQ_Cont'] = df_performance['Equal_Weight'] * df_performance['Actual_Return']


# 2. Group by Date to get the total portfolio return for each specific month
monthly_perf = df_performance.groupby('Date')[['ML_Cont', 'EQ_Cont']].sum() # performance of whole portfolio for this month

# 3. Calculate Cumulative Wealth (If $1.00 was invested at the start)
cumulative_returns = (1 + monthly_perf).cumprod()

# 4. Calculate Total Months for Annualization Math
total_months = len(monthly_perf)

# 5. Calculate Annualized Return (Compound Annual Growth Rate)
annualized_return_ml = (cumulative_returns['ML_Cont'].iloc[-1]) ** (12 / total_months) - 1
annualized_return_eq = (cumulative_returns['EQ_Cont'].iloc[-1]) ** (12 / total_months) - 1

# 6. Calculate Annualized Volatility (Risk)- multiply standard deviation with square root of time 
annual_vol_ml = monthly_perf['ML_Cont'].std() * np.sqrt(12) 
annual_vol_eq = monthly_perf['EQ_Cont'].std() * np.sqrt(12)

# 7. Calculate Sharpe Ratio (Return divided by Risk)
sharpe_ml = annualized_return_ml / annual_vol_ml
sharpe_eq = annualized_return_eq / annual_vol_eq

print("\n=========================================================")
print("      10-YEAR BACKTEST REPORT (2015 - PRESENT)           ")
print("=========================================================")
print(f"Total Months Tested: {total_months}")
print("---------------------------------------------------------")
print("PORTFOLIO             | ANN. RETURN | ANN. RISK | SHARPE ")
print("---------------------------------------------------------")
print(f"Machine Learning (BL) |    {annualized_return_ml*100:>5.2f}%   |   {annual_vol_ml*100:>5.2f}%  |  {sharpe_ml:>5.2f} ")
print(f"Equal-Weight Baseline |    {annualized_return_eq*100:>5.2f}%   |   {annual_vol_eq*100:>5.2f}%  |  {sharpe_eq:>5.2f} ")
print("=========================================================\n")



# YEAR-BY-YEAR BREAKDOWN 
print("=========================================================")
print("                 YEAR-BY-YEAR BREAKDOWN                  ")
print("=========================================================")
print("YEAR | ML STRATEGY | EQUAL-WEIGHT | OUTPERFORMED? ")
print("---------------------------------------------------------")

# Resample monthly returns to yearly returns
yearly_perf = (1 + monthly_perf).resample('YE').prod() - 1

# Print each year out
for year_date, row in yearly_perf.iterrows():
    ml_ret = row['ML_Cont'] * 100
    eq_ret = row['EQ_Cont'] * 100
    beat = "YES" if ml_ret > eq_ret else "NO"
    print(f"{year_date.year} | {ml_ret:>8.2f}%   | {eq_ret:>10.2f}% | {beat}")
    
print("=========================================================\n")