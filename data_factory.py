import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
import os
import json

def load_config():
    """Loads configuration from config.json, returns empty dict if not found."""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def download_nltk_data():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

def get_mock_sentiment(sia):
    """
    Simulates fetching daily news headlines and returns a sentiment score.
    """
    headlines = [
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

    headline = random.choice(headlines)
    score = sia.polarity_scores(headline)['compound']
    # Add some noise to make it less discrete
    score += random.uniform(-0.1, 0.1)
    return max(-1.0, min(1.0, score)) # Clip to [-1, 1]

def fetch_data():
    download_nltk_data()
    sia = SentimentIntensityAnalyzer()

    # Load configuration
    config = load_config()

    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
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
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
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
        df['Sentiment_Score'] = [get_mock_sentiment(sia) for _ in range(len(df))]

        # Drop NaN rows
        df = df.dropna()

        # Split Data
        try:
            # Slicing with .loc using strings works if the index is DatetimeIndex or strings.
            # yfinance returns DatetimeIndex, so string slicing is supported.
            train_df = df.loc[train_start_date:train_end_date]
            test_df = df.loc[test_start_date:]

            # Save to CSV
            train_file = f'data/train/{ticker}_data.csv'
            train_df.to_csv(train_file)
            print(f"Train data saved to {train_file}")

            test_file = f'data/test/{ticker}_data.csv'
            test_df.to_csv(test_file)
            print(f"Test data saved to {test_file}")

        except Exception as e:
            print(f"Error splitting/saving data for {ticker}: {e}")

if __name__ == "__main__":
    fetch_data()
