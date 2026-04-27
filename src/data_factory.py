import yfinance as yf
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import os
import re
import joblib
from statsmodels.tsa.stattools import adfuller
import scipy.stats as ss
from core.optimize_barriers import get_rolling_barriers

MOCK_HEADLINES = [
    "Company reports record earnings.",
    "Market crashes due to geopolitical tensions.",
    "New product launch is a huge success.",
    "CEO resigns amid scandal.",
    "Analyst upgrades stock rating.",
    "Analyst downgrades stock rating.",
    "Sector faces regulatory scrutiny.",
    "Competitor announces major breakthrough.",
    "Global economy shows signs of recovery.",
    "Interest rates expected to rise.",
    "Company announces stock buyback program.",
    "Supply chain issues persist.",
    "Quarterly revenue exceeds expectations.",
    "Lawsuit filed against the company.",
    "Strategic partnership announced.",
    "Market remains flat.",
    "Investors are cautious ahead of earnings."
]
_MOCK_HEADLINE_SCORES = None


def get_weights_ffd(d, thres=1e-4):
    """
    Calculates the weights for Fractional Differentiation (FFD).

    Fractional Differentiation allows for transforming a non-stationary time series
    into a stationary one while preserving the maximum amount of memory (unlike
    integer differentiation which completely destroys memory).

    Args:
        d (float): The fractional differentiation degree (0 < d < 1).
        thres (float): The weight threshold to truncate the series. Defaults to 1e-4.

    Returns:
        np.ndarray: Column vector of weights.
    """
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thres=1e-4):
    """
    Applies Fractional Differentiation (FFD) to a Pandas DataFrame.

    By fractionally differentiating price series, the resulting data becomes
    stationary (passing the ADF test) while retaining long memory of past prices,
    crucial for the predictive power of machine learning models.

    Args:
        series (pd.DataFrame): DataFrame containing the target series (e.g., 'Close' price).
        d (float): The optimal fractional degree.
        thres (float): The threshold for weight calculation. Defaults to 1e-4.

    Returns:
        pd.DataFrame: A DataFrame with the fractionally differentiated series.
    """
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    # get_weights_ffd returns weights in reversed order [w_k, ..., w_0]
    # For FFD calculation we need the natural order [w_0, ..., w_k] for convolution
    w_natural = w.flatten()[::-1]
    df = {}
    for name in series.columns:
        seriesF = series[name].ffill().dropna()
        f_values = seriesF.values
        f_index = seriesF.index

        results = np.full(len(f_values), np.nan)
        if len(f_values) > width:
            # Use fast convolution to replace iterative dot products
            conv = np.convolve(f_values, w_natural, mode='valid')
            results[width:] = conv

            # Mask results where the original series had non-finite values (matching original behavior)
            is_finite = np.isfinite(series[name].loc[f_index].values)
            results[~is_finite] = np.nan

        df[name] = pd.Series(results, index=f_index)
    return pd.concat(df, axis=1)

def download_nltk_data():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

def _get_cached_scores(sia):
    global _MOCK_HEADLINE_SCORES
    if _MOCK_HEADLINE_SCORES is None:
        _MOCK_HEADLINE_SCORES = np.array([sia.polarity_scores(h)['compound'] for h in MOCK_HEADLINES])
    return _MOCK_HEADLINE_SCORES

def get_mock_sentiment_batch(n, sia):
    """
    Generates a batch of mock sentiment scores for the simulated dataset.

    Args:
        n (int): Number of sentiment scores to generate.
        sia (SentimentIntensityAnalyzer): Pre-initialized VADER sentiment analyzer.

    Returns:
        np.ndarray: An array of simulated sentiment scores clipped between -1.0 and 1.0.
    """
    scores = _get_cached_scores(sia)

    # Vectorized sampling
    selected_scores = np.random.choice(scores, size=n)
    noise = np.random.uniform(-0.1, 0.1, size=n)
    final_scores = selected_scores + noise
    return np.clip(final_scores, -1.0, 1.0)

def construct_dollar_bars(df, target_bars_per_day=10):
    """
    Compresses standard time bars into Information-Driven Dollar Bars.

    Unlike time bars which suffer from heteroscedasticity (varying volatility and
    information flow depending on the time of day), Dollar Bars sample the market
    only when a dynamic threshold of dollar volume is exchanged. This neutralizes
    heteroscedasticity and restores the statistical properties of the price series
    (bringing it closer to a Normal distribution).

    Args:
        df (pd.DataFrame): The raw intraday time bars (Open, High, Low, Close, Volume).
        target_bars_per_day (int): The target average number of bars per day. Defaults to 10.

    Returns:
        pd.DataFrame: A DataFrame indexed by the completion time of each Dollar Bar.
    """
    # Calculate 'Dollar Volume' as numpy array to avoid unnecessary DF copy
    dv_values = (df['Close'] * df['Volume']).values

    # Calculate dynamic threshold (M):
    # Rolling 30-day window (approx 210 hourly trading bars) of total Dollar Volume, divided by 300
    M = pd.Series(dv_values, index=df.index).rolling(window=210).sum() / 300

    # Forward-fill/backward-fill NaNs
    M = M.ffill().bfill()

    # 1. Extract to native NumPy arrays for maximum speed
    m_values = M.values

    # 2. Fast tracking loop (No pandas overhead)
    n_ticks = len(dv_values)
    breach_indices = []
    cum_dv = 0.0

    for i in range(n_ticks):
        cum_dv += dv_values[i]
        if cum_dv >= m_values[i]:
            breach_indices.append(i)
            cum_dv = 0.0 # Exactly preserves the "discard remainder" logic

    # 3. Create a grouping array for vectorized aggregation
    if not breach_indices:
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume']).set_index(pd.Index([], name='Date')) if 'Date' not in df.columns else pd.DataFrame()

    # Drop the incomplete forming bar by slicing up to the final breach
    last_breach = breach_indices[-1]

    groups = np.zeros(last_breach + 1, dtype=int)
    # Every tick AFTER a breach starts a new group
    split_indices = np.array(breach_indices[:-1]) + 1
    groups[split_indices] = 1
    group_ids = np.cumsum(groups)

    # Assign groups back to a temporary dataframe
    df_temp = df.iloc[:last_breach + 1].copy()
    df_temp['Group'] = group_ids

    # 4. Vectorized groupby to build the bars instantly
    dollar_bars = df_temp.groupby('Group').agg(
        Open=('Open', 'first'),
        High=('High', 'max'),
        Low=('Low', 'min'),
        Close=('Close', 'last'),
        Volume=('Volume', 'sum')
    )

    # 5. Extract the correct timestamps (the time of the breach tick)
    dollar_bars.index = df.index[breach_indices]
    dollar_bars.index.name = 'Date'

    return dollar_bars

def fetch_data(config_path='config/config_phase1.json'):
    """
    Main data pipeline: fetches, processes, and saves financial data.

    This function executes the complete feature engineering pipeline:
    1. Downloads 730 days of 1-hour intraday data.
    2. Compresses time bars into Information-Driven Dollar Bars.
    3. Calculates technical indicators (RSI, MACD, BB, ATR).
    4. Computes rolling Optimal Trading Rules (PT and SL multipliers).
    5. Applies Point-in-Time PCA on technical indicators to prevent collinearity.
    6. Applies Fractional Differentiation to the 'Close' price for stationarity.
    7. Splits the data into Train/Test sets with an embargo to prevent leakage.
    """
    download_nltk_data()
    sia = SentimentIntensityAnalyzer()

    # Security Fix: Prevent Path Traversal
    # 1. Resolve project root and allowed config directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    allowed_config_dir = os.path.normpath(os.path.join(project_root, "config"))

    # 2. Resolve path: if simple filename, assume in config/
    if os.path.dirname(config_path) == "":
        target_path = os.path.join(allowed_config_dir, config_path)
    else:
        if not os.path.isabs(config_path):
            target_path = os.path.join(project_root, config_path)
        else:
            target_path = config_path

    # 3. Final Security Validation
    abs_config_path = os.path.abspath(target_path)
    if not abs_config_path.startswith(allowed_config_dir + os.sep):
        print(f"[ERROR] Security: Configuration path '{config_path}' is restricted.")
        return

    # Load configuration
    import json
    with open(abs_config_path, 'r') as f:
        config = json.load(f)

    import shutil
    # Wipe previous data to prevent cross-asset pollution
    shutil.rmtree('data/train', ignore_errors=True)
    shutil.rmtree('data/test', ignore_errors=True)

    # Ensure directories exist
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    os.makedirs('models/matrices', exist_ok=True)

    # Use configuration with fallbacks
    tickers = config.get('tickers', ['NVDA', 'AAPL', 'MSFT', 'AMD', 'INTC'])
    data_window_days = config.get('data_window_days', 730)

    print(f"Fetching intraday data for {tickers}...")
    try:
        # Fetch data for all tickers at once
        data = yf.download(tickers, period=f"{data_window_days}d", interval='1h', group_by='ticker')
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    for ticker in tickers:
        print(f"Processing {ticker}...")

        # Sanitize ticker to prevent path traversal
        clean_ticker = os.path.basename(ticker)
        # Allow standard alphanumeric characters, dot, hyphen, underscore, and caret for index tickers
        if not re.match(r'^[\^a-zA-Z0-9_.-]+$', clean_ticker):
             print(f"Skipping invalid ticker: {ticker}")
             continue
        if clean_ticker != ticker:
             print(f"Warning: Ticker '{ticker}' sanitized to '{clean_ticker}'")
             # Proceed with sanitized ticker for file saving purposes, 
             # but we still need to access data using the original key if applicable.
        
        # However, yfinance download uses the original ticker list. 
        # So we should probably use the original ticker to access data, 
        # but the sanitized ticker for filenames.

        try:
            # Extract dataframe for specific ticker
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    df = data[ticker].copy()
                except KeyError:
                    print(f"No data found for {ticker} in bulk download.")
                    continue
            else:
                # Fallback if only one ticker or flat structure returned
                df = data.copy()

        except Exception as e:
            print(f"Error extracting data for {ticker}: {e}")
            continue

        if df.empty:
            print(f"No data fetched for {ticker}.")
            continue

        # Ensure 'Close' column exists
        if 'Close' not in df.columns:
            print(f"Error: 'Close' column not found in dataframe for {ticker}. Columns: {df.columns}")
            continue

        # Make index tz-naive and compress to Dollar Bars
        df.index = df.index.tz_localize(None)
        df = construct_dollar_bars(df)

        if df.empty:
            print(f"Not enough data to construct Dollar Bars for {ticker}.")
            continue

        # Calculate RSI (14-day)
        print(f"Calculating RSI for {ticker}...")
        delta = df['Close'].diff()

        # Gain/Loss
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))

        # Average Gain/Loss using Wilder's Smoothing (alpha=1/14)
        # com = 13 corresponds to alpha = 1/14
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate MACD (12, 26, 9)
        print(f"Calculating MACD for {ticker}...")
        # EMA 12
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        # EMA 26
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()

        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Calculate Bollinger Bands (20-day)
        print(f"Calculating Bollinger Bands for {ticker}...")
        rolling_20 = df['Close'].rolling(window=20)
        sma_20 = rolling_20.mean()
        std_20 = rolling_20.std()
        df['BB_Upper'] = sma_20 + 2 * std_20
        df['BB_Lower'] = sma_20 - 2 * std_20

        # Calculate ATR (14-day)
        print(f"Calculating ATR for {ticker}...")
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()

        print(f"Calculating Dynamic Barriers for {ticker}...")
        barriers_df = get_rolling_barriers(df['Close'], window=60, step=20)
        df['Optimal_PT'] = barriers_df['Optimal_PT'].fillna(2.0)
        df['Optimal_SL'] = barriers_df['Optimal_SL'].fillna(2.0)

        # Add Simulated Sentiment
        print(f"Calculating Simulated Sentiment for {ticker}...")
        # Generating sentiment for all rows including those that might have NaNs (which are dropped later)
        df['Sentiment_Score'] = get_mock_sentiment_batch(len(df), sia)

        # Apply Fractional Differentiation to Close price
        print(f"Applying Fractional Differentiation to {ticker}...")
        optimal_d = 1.0
        best_ffd = None
        # Loop d from 0.1 to 0.9 in increments of 0.1
        for d in np.arange(0.1, 1.0, 0.1):
            df_ffd = frac_diff_ffd(df[['Close']], d)
            clean_ffd = df_ffd['Close'].dropna()
            
            # Safety check: ADF test requires a sufficient number of valid rows (e.g., > 30 days)
            if len(clean_ffd) > 30:
                p_val = adfuller(clean_ffd)[1]
                if p_val < 0.05:
                    optimal_d = d
                    best_ffd = df_ffd['Close']
                    break

        if best_ffd is not None:
            print(f"Optimal d for {ticker} found: {optimal_d:.1f}")
            df['Close_FFD'] = best_ffd
        else:
            print(f"Warning: Could not find stationary series for {ticker} with d < 1.0. Using d=1.0")
            df['Close_FFD'] = frac_diff_ffd(df[['Close']], 1.0)['Close']

        tech_cols = ['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'ATR']
        
        # 1. Drop NaNs FIRST so the split calculations are accurate
        df = df.dropna()

        # 2. Dynamically calculate the 80% split date on healthy data
        split_idx = int(len(df) * 0.8)
        split_date = df.index[split_idx]

        # 3. Create the Train index for PCA
        train_clean_idx = df[df.index < split_date].index
        all_clean_idx = df.index

        # 4. Apply Point-in-Time PCA
        scaler = StandardScaler()
        scaler.fit(df.loc[train_clean_idx, tech_cols])
        scaled_tech = scaler.transform(df.loc[all_clean_idx, tech_cols])
        
        pca = PCA(n_components=5)
        scaled_train_tech = scaler.transform(df.loc[train_clean_idx, tech_cols])
        pca.fit(scaled_train_tech)
        pca_features = pca.transform(scaled_tech)

        # Save matrices for Live Inference
        import joblib
        joblib.dump(scaler, f'models/matrices/scaler_{clean_ticker}.pkl')
        joblib.dump(pca, f'models/matrices/pca_{clean_ticker}.pkl')

        pca_cols = ['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5']
        df_pca = pd.DataFrame(pca_features, index=all_clean_idx, columns=pca_cols)
        df = pd.concat([df, df_pca], axis=1)
        df.drop(columns=tech_cols, inplace=True)

        # 5. Split Data
        try:
            train_df = df[df.index < split_date]
            raw_test_df = df[df.index >= split_date]
            # Hotfix: Dynamic Train/Test Split applied here
            
            # IMPLEMENT 60-PERIOD EMBARGO (Drop first 60 rows of Test Set to prevent feature leakage)
            embargo_size = 60 # Maximum feature lookback window.
            if len(raw_test_df) > embargo_size:
                test_df = raw_test_df.iloc[embargo_size:]
            else:
                test_df = raw_test_df # Fallback if test set is too small
                
            # Save to CSV
            train_file = f'data/train/{clean_ticker}_data.csv'
            train_df.to_csv(train_file)
            print(f"Train data saved to {train_file}")

            test_file = f'data/test/{clean_ticker}_data.csv'
            test_df.to_csv(test_file)
            print(f"Test data saved to {test_file}")

        except Exception as e:
            print(f"Error splitting/saving data for {ticker}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_phase1.json')
    args = parser.parse_args()
    fetch_data(args.config)
