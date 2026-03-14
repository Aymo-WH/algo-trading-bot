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
from utils import load_config
from statsmodels.tsa.stattools import adfuller
import scipy.stats as ss

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

def get_weights(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thres=1e-5):
    w = get_weights(d, series.shape[0])
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    df = {}
    for name in series.columns:
        seriesF = series[[name]].ffill().dropna()
        df_ = pd.Series(dtype=float, index=seriesF.index)
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]): continue
            df_.loc[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[seriesF.index[:iloc + 1], name])[0]
        df[name] = df_.copy(deep=True)
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
    scores = _get_cached_scores(sia)

    # Vectorized sampling
    selected_scores = np.random.choice(scores, size=n)
    noise = np.random.uniform(-0.1, 0.1, size=n)
    final_scores = selected_scores + noise
    return np.clip(final_scores, -1.0, 1.0)

def get_mock_sentiment(sia):
    """
    Simulates fetching daily news headlines and returns a sentiment score.
    """
    scores = _get_cached_scores(sia)
    score = np.random.choice(scores)
    # Add some noise to make it less discrete
    score += random.uniform(-0.1, 0.1)
    return max(-1.0, min(1.0, score)) # Clip to [-1, 1]

def fetch_data():
    download_nltk_data()
    sia = SentimentIntensityAnalyzer()

    # Load configuration
    config = load_config()

    # Ensure data directory exists
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)

    # Use configuration with fallbacks
    tickers = config.get('tickers', ['NVDA', 'AAPL', 'MSFT', 'AMD', 'INTC'])
    start_date = config.get('start_date', '2016-01-01')
    end_date = config.get('end_date', '2026-01-01')

    train_start_date = config.get('train_start_date', '2016-01-01')
    train_end_date = config.get('train_end_date', '2022-12-31')
    test_start_date = config.get('test_start_date', '2023-01-01')

    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    try:
        # Fetch data for all tickers at once
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
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
        
        # Apply PCA
        clean_idx = df[tech_cols].dropna().index
        scaler = StandardScaler()
        scaled_tech = scaler.fit_transform(df.loc[clean_idx, tech_cols])
        
        pca = PCA(n_components=5)
        pca_features = pca.fit_transform(scaled_tech)
        pca_cols = ['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5']
        
        # BULLETPROOF PANDAS ASSIGNMENT
        df_pca = pd.DataFrame(pca_features, index=clean_idx, columns=pca_cols)
        df = pd.concat([df, df_pca], axis=1)
        
        # Drop the old highly-correlated columns
        df.drop(columns=tech_cols, inplace=True)

        # Drop NaN rows
        df = df.dropna()

        # Split Data
        try:
            # Slicing with .loc using strings works if the index is DatetimeIndex or strings.
            # yfinance returns DatetimeIndex, so string slicing is supported.
            train_df = df.loc[train_start_date:train_end_date]
            test_df = df.loc[test_start_date:]

            # Save to CSV
            # USE CLEAN TICKER HERE
            train_file = f'data/train/{clean_ticker}_data.csv'
            train_df.to_csv(train_file)
            print(f"Train data saved to {train_file}")

            test_file = f'data/test/{clean_ticker}_data.csv'
            test_df.to_csv(test_file)
            print(f"Test data saved to {test_file}")

        except Exception as e:
            print(f"Error splitting/saving data for {ticker}: {e}")

if __name__ == "__main__":
    fetch_data()
