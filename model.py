import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from numpy.linalg import inv


print("Reading the Excel file structure...")
excel_file = pd.ExcelFile("BL AA project mischa data 18 May 2026 VALUES.xlsx")

# Skip first tab ('Main')
target_sheets = excel_file.sheet_names[1:]

all_assets = []

# LOAD AND CLEAN INDIVIDUAL SHEETS
for sheet_name in target_sheets:
    
    sheet_data = pd.read_excel(excel_file, sheet_name=sheet_name)
    # the day comes first (DD/MM/YYYY)
    sheet_data['Date'] = pd.to_datetime(sheet_data['Date'], dayfirst=True)
    
    # make the date the row labels
    sheet_data.set_index('Date', inplace=True)
    
    # Grab the first column of data 
    asset_prices = sheet_data.iloc[:, [0]]
    
    # Rename that column to just the sheet name ('SPY' instead of 'SPY US Equity')
    asset_prices.columns = [sheet_name]
    
    all_assets.append(asset_prices)

# --- STEP 2: STITCH THEM TOGETHER ---
prices = pd.concat(all_assets, axis=1)

# --- STEP 3: ALIGN DATES ---
# Keep only data from Jan 2024 onward to match the Bitcoin ETF
prices = prices.loc['2024-01-01':]

# Drop rows with missing data
prices = prices.dropna()

# --- STEP 4: CALCULATE RETURNS & COVARIANCE ---
returns = np.log(prices / prices.shift(1))
returns = returns.dropna()

cov_matrix = returns.cov()

# Save the outputs
prices.to_csv("stitched_prices.csv")
cov_matrix.to_csv("covariance_matrix.csv")

print("Data stitched successfully! Covariance matrix generated.")

# PHASE 2


print("Loading Covariance Matrix from Phase 1...")
cov_matrix = pd.read_csv("covariance_matrix.csv", index_col=0)

# --- STEP 2.1: LIVE MARKET WEIGHTS ---
tickers = [
    'SPY', 'HYG', 'EFA', 'EEM', 'TLT', 'GLD', 'SLV', 
    'DBC', 'VNQ', 'IBIT', 'IEV', 'XGLU', 'IEF', 'SHY'
]

market_caps = {}

print("Pulling live market data from Yahoo Finance. This might take a minute...")
for ticker in tickers:
    try:
        # yfinance sometimes requires foreign stocks to have an extension. 
        # XGLU is traded in London, so add '.L' for Yahoo Finance to find it.
        search_ticker = 'XGLU.L' if ticker == 'XGLU' else ticker
        
        asset = yf.Ticker(search_ticker)
        
        # For ETFs, the size is stored as 'totalAssets'. 
        # fallback to 'marketCap' if missing
        size = asset.info.get('totalAssets') or asset.info.get('marketCap')
        
        if size is None:
            print(f"Warning: Could not find size for {ticker}. Using your placeholder.")
            size = 208660278.32 if ticker == 'XGLU' else 10000000000
         
            
        market_caps[ticker] = size
        print(f"Successfully pulled {ticker}: ${size:,.2f}")
        
    except Exception as e:
        print(f"Failed to pull {ticker}: {e}")

# Convert dictionary into a Pandas series
caps_series = pd.Series(market_caps)

# Calculate the weights
market_weights = caps_series / caps_series.sum()

#  align the weights to match the Covariance matrix order
market_weights = market_weights[cov_matrix.index]


# --- STEP 2.2: RISK AVERSION COEFFICIENT (Lambda) ---
risk_aversion = 2.5


# --- STEP 2.3: IMPLIED EQUILIBRIUM RETURNS (Pi) ---
# Pi = Lambda * Covariance * Weights
implied_returns = risk_aversion * cov_matrix.dot(market_weights)

# Annualize them for 252 trading days
annual_implied_returns = implied_returns * 252

annual_implied_returns.to_csv("implied_equilibrium_returns.csv")

print("\n--- PHASE 2 COMPLETE ---")
print("Here are the Market's Live Expected Annual Returns:")
print((annual_implied_returns * 100).round(2).astype(str) + '%')


# PHASE 3


assets = cov_matrix.index.tolist()

# 3 example views:
# View 1: SPY outperforms IEV by 2%
# View 2: GLD has an absolute return of 5.5%
# View 3: HYG outperforms SHY by 1.5%

# CREATE MATRIX P - 3 rows by 14 columns
P = np.zeros((3, len(assets)))

# View 1: SPY (+1) beats IEV (-1)
P[0, assets.index('SPY')] = 1.0
P[0, assets.index('IEV')] = -1.0

# View 2: GLD absolute return (+1)
P[1, assets.index('GLD')] = 1.0

# View 3: HYG (+1) beats SHY (-1)
P[2, assets.index('HYG')] = 1.0
P[2, assets.index('SHY')] = -1.0


# CREATE VECTOR Q - expected returns for each view
Q = np.array([
    0.020,  
    0.055,  
    0.015   
])

# ASSIGN CONFIDENCE LEVELS (60%, 75%, 50%)
confidences = np.array([
    0.60, 
    0.75, 
    0.50  
])

print("Matrix P:")
print(pd.DataFrame(P, columns=assets))
print("\nVector Q:", Q)
print("Confidences:", confidences)


print("\n--- STARTING PHASE 4: THE OMEGA MATRIX ---")

# standard industry constant for the Black-Litterman model
tau = 0.025

omega_diagonals = []

# looping through all views
for i in range(len(Q)):
    
    # row for this view from Matrix P
    P_k = P[i]
    
    confidence = confidences[i]
    
    # Calculate the baseline variance of this specific view using matrix multiplication formula
    view_variance = P_k.dot(cov_matrix).dot(P_k.T)
    

    # Cap confidence at 99.9% to avoid division by zero
    if confidence >= 1.0:
        confidence = 0.999
        
    # tau * variance / (1 - confidence)
    omega_k = (tau * view_variance) / (1 - confidence)
    
    omega_diagonals.append(omega_k)

# build the diagonal matrix
Omega = np.diag(omega_diagonals)

print("\nThe Omega Matrix : ")
print(pd.DataFrame(Omega).round(6))


# PHASE 5   

print("\n--- STARTING PHASE 5: BLACK-LITTERMAN RETURNS ---")

# Annualize covariance matrix so it matches Q
cov_matrix_annual = cov_matrix * 252

# Baseline Market Returns (Pi) 
Pi = annual_implied_returns.values

# Recalculate Omega using the ANNUAL risk 
omega_diagonals_annual = []
for i in range(len(Q)):
    P_k = P[i]
    confidence = confidences[i] if confidences[i] < 1.0 else 0.999
    view_variance = P_k.dot(cov_matrix_annual).dot(P_k.T)
    omega_diagonals_annual.append((tau * view_variance) / (1 - confidence))
    
Omega_annual = np.diag(omega_diagonals_annual)

# Part A: (tau * Sigma)^-1
tau_Sigma_inv = inv(tau * cov_matrix_annual)

# Part B: P' * Omega^-1 * P
Omega_inv = inv(Omega_annual)
P_transpose = P.T
view_matrix_term = P_transpose.dot(Omega_inv).dot(P)

# Combine Left Half: Invert the sum of Part A + Part B
left_term = inv(tau_Sigma_inv + view_matrix_term)

# Part C: (tau * Sigma)^-1 * Pi
baseline_return_term = tau_Sigma_inv.dot(Pi)

# Part D: P' * Omega^-1 * Q
view_return_term = P_transpose.dot(Omega_inv).dot(Q)

# Combine Right Half: Sum of Part C + Part D
right_term = baseline_return_term + view_return_term


# THE FINAL COMBINATION
bl_returns = left_term.dot(right_term)

# Compare and see how the views changed the market
results = pd.DataFrame({
    'Market Baseline (%)': (Pi * 100).round(2),
    'Black-Litterman (%)': (bl_returns * 100).round(2)
}, index=assets)

# Calculate the difference and see the impact of the views
results['Difference'] = results['Black-Litterman (%)'] - results['Market Baseline (%)']

print("\nFINAL EXPECTED RETURNS:")
print(results)