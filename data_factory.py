import yfinance as yf
import pandas as pd

def fetch_data():
    ticker = 'NVDA'
    start_date = '2020-01-01'
    end_date = '2026-02-21' # Exclusive, so includes 2026-02-20

    print(f"Fetching {ticker} data from {start_date} to {end_date}...")

    # Fetch data
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if df.empty:
        print("No data fetched.")
        return

    # Check if MultiIndex columns (common in new yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        # We assume the first level is Price and second is Ticker
        # We can drop the Ticker level if it's just one ticker
        try:
            df.columns = df.columns.droplevel(1)
        except Exception as e:
            print(f"Warning: Could not drop level from MultiIndex columns: {e}")
            pass

    # Ensure 'Close' column exists
    if 'Close' not in df.columns:
        print(f"Error: 'Close' column not found in dataframe. Columns: {df.columns}")
        return

    # Calculate RSI (14-day)
    print("Calculating RSI...")
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
    print("Calculating MACD...")
    # EMA 12
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    # EMA 26
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()

    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Save to CSV
    output_file = 'nvda_data.csv'
    df.to_csv(output_file)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    fetch_data()
