import yfinance as yf
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
import os
from utils import load_config

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

def extract_ticker_data(data, ticker):
    """
    Extracts a DataFrame for a specific ticker from a bulk download.
    Handles MultiIndex columns if present.
    """
    if isinstance(data.columns, pd.MultiIndex):
        try:
            df = data[ticker].copy()
        except KeyError:
            print(f"No data found for {ticker} in bulk download.")
            return None
    else:
        # Fallback if only one ticker or flat structure returned
        df = data.copy()

    if df.empty:
        print(f"No data fetched for {ticker}.")
        return None

    return df

def calculate_rsi(df, window=14):
    """Calculates RSI (14-day)"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    # Average Gain/Loss using Wilder's Smoothing (alpha=1/14)
    avg_gain = gain.ewm(com=window-1, adjust=False).mean()
    avg_loss = loss.ewm(com=window-1, adjust=False).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df):
    """Calculates MACD (12, 26, 9)"""
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()

    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def calculate_bollinger_bands(df, window=20):
    """Calculates Bollinger Bands (20-day)"""
    sma_20 = df['Close'].rolling(window=window).mean()
    std_20 = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = sma_20 + 2 * std_20
    df['BB_Lower'] = sma_20 - 2 * std_20
    return df

def calculate_atr(df, window=14):
    """Calculates ATR (14-day)"""
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=window).mean()
    return df

def add_sentiment(df, sia):
    """Adds Simulated Sentiment score"""
    df['Sentiment_Score'] = get_mock_sentiment_batch(len(df), sia)
    return df

def save_ticker_data(df, ticker, train_start, train_end, test_start):
    """Splits and saves ticker data to CSV."""
    try:
        train_df = df.loc[train_start:train_end]
        test_df = df.loc[test_start:]

        train_file = f'data/train/{ticker}_data.csv'
        train_df.to_csv(train_file)
        print(f"Train data saved to {train_file}")

        test_file = f'data/test/{ticker}_data.csv'
        test_df.to_csv(test_file)
        print(f"Test data saved to {test_file}")
    except Exception as e:
        print(f"Error splitting/saving data for {ticker}: {e}")

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
    start_date = config.get('start_date', '2018-01-01')
    end_date = config.get('end_date', '2026-01-01')

    train_start_date = config.get('train_start_date', '2018-01-01')
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

        try:
            df = extract_ticker_data(data, ticker)
            if df is None:
                continue

            # Ensure 'Close' column exists
            if 'Close' not in df.columns:
                print(f"Error: 'Close' column not found in dataframe for {ticker}. Columns: {df.columns}")
                continue

            print(f"Calculating RSI for {ticker}...")
            df = calculate_rsi(df)

            print(f"Calculating MACD for {ticker}...")
            df = calculate_macd(df)

            print(f"Calculating Bollinger Bands for {ticker}...")
            df = calculate_bollinger_bands(df)

            print(f"Calculating ATR for {ticker}...")
            df = calculate_atr(df)

            print(f"Calculating Simulated Sentiment for {ticker}...")
            df = add_sentiment(df, sia)

            # Drop NaN rows
            df = df.dropna()

            # Split Data and Save
            save_ticker_data(df, ticker, train_start_date, train_end_date, test_start_date)

        except Exception as e:
            print(f"Error extracting data for {ticker}: {e}")
            continue

if __name__ == "__main__":
    fetch_data()
