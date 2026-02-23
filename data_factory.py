import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os

def fetch_data():
    tickers = ['NVDA', 'AAPL', 'MSFT', 'AMD', 'INTC']
    start_date = '2020-01-01'
    end_date = '2026-02-21' # Exclusive, so includes 2026-02-20

    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)

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

        # Calculate Bollinger Bands
        print(f"Calculating Bollinger Bands for {ticker}...")
        # Standard settings: length=20, std=2
        bb = df.ta.bbands(length=20, std=2)
        if bb is not None:
             # Keep only Lower, Mid, Upper. Usually first 3 columns.
             bb = bb.iloc[:, :3]
             bb.columns = ['BBL', 'BBM', 'BBU']
             df = pd.concat([df, bb], axis=1)

        # Calculate ATR
        print(f"Calculating ATR for {ticker}...")
        # ATR needs High, Low, Close. Assuming they exist. yfinance usually provides them.
        atr = df.ta.atr(length=14)
        if atr is not None:
            atr.name = 'ATR'
            df = pd.concat([df, atr], axis=1)

        # Drop NaN rows
        df = df.dropna()

        # Save to CSV
        output_file = f'data/{ticker}_data.csv'
        df.to_csv(output_file)
        print(f"Data saved to {output_file}")

if __name__ == "__main__":
    fetch_data()
