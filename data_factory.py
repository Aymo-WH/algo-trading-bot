import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
import os

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

    tickers = ['NVDA', 'AAPL', 'MSFT', 'AMD', 'INTC']
    start_date = '2020-01-01'
    end_date = '2026-02-21' # Exclusive, so includes 2026-02-20

    for ticker in tickers:
        print(f"Fetching {ticker} data from {start_date} to {end_date}...")

        # Fetch data
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue

        if df.empty:
            print(f"No data fetched for {ticker}.")
            continue

        # Check if MultiIndex columns (common in new yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            # We assume the first level is Price and second is Ticker
            # We can drop the Ticker level if it's just one ticker
            try:
                df.columns = df.columns.droplevel(1)
            except Exception as e:
                print(f"Warning: Could not drop level from MultiIndex columns for {ticker}: {e}")
                pass

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

        # Add Simulated Sentiment
        print(f"Calculating Simulated Sentiment for {ticker}...")
        # Generating sentiment for all rows including those that might have NaNs (which are dropped later)
        df['Sentiment_Score'] = [get_mock_sentiment(sia) for _ in range(len(df))]

        # Drop NaN rows
        df = df.dropna()

        # Save to CSV
        output_file = f'data/{ticker}_data.csv'
        df.to_csv(output_file)
        print(f"Data saved to {output_file}")

if __name__ == "__main__":
    fetch_data()
