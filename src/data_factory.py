import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import re
import joblib
import ccxt
import time
from statsmodels.tsa.stattools import adfuller
from core.optimize_barriers import get_rolling_barriers
from numba import njit, prange

TRAIN_SPLIT_RATIO = 0.8

@njit
def _sadf_inner(y_sub):
    n = len(y_sub)
    dy = y_sub[1:]
    yx = y_sub[:-1]

    sum_y = np.sum(yx)
    sum_y2 = np.sum(yx**2)

    mean_x2 = sum_y / (n - 1)
    mean_y = np.sum(dy) / (n - 1)

    var_x2 = sum_y2 - (sum_y**2)/(n-1)
    if var_x2 < 1e-10:
        return 0.0

    cov_x2_y = np.sum(yx * dy) - sum_y * np.sum(dy) / (n-1)

    beta = cov_x2_y / var_x2
    alpha = mean_y - beta * mean_x2

    res = dy - (alpha + beta * yx)
    ssr = np.sum(res**2)

    if ssr < 1e-10:
        return 0.0

    sigma2 = ssr / (n - 3)
    se_beta = np.sqrt(sigma2 / var_x2)

    if se_beta < 1e-10:
        return 0.0

    return beta / se_beta

@njit(parallel=True)
def rolling_sadf_np(prices, min_len, window):
    n = len(prices)
    sadf = np.full(n, np.nan)

    for t in prange(window, n):
        max_tstat = -np.inf
        for s in range(t - window, t - min_len + 1):
            y_sub = prices[s:t+1]
            tstat = _sadf_inner(y_sub)
            if tstat > max_tstat:
                max_tstat = tstat
        sadf[t] = max_tstat
    return sadf


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


def calculate_microstructural_features(df, window=50):
    '''Calculates VPIN, Amihud Illiquidity, Kyle's Lambda, and SADF for a dataframe.'''
    # Calculate VPIN
    dp = df['Close'].diff()
    buy_vol = np.where(dp > 0, df['Volume'], np.where(dp == 0, df['Volume'] / 2, 0))
    sell_vol = np.where(dp < 0, df['Volume'], np.where(dp == 0, df['Volume'] / 2, 0))
    v_imb = np.abs(buy_vol - sell_vol)

    rolling_v_imb = pd.Series(v_imb, index=df.index).rolling(window=window).sum()
    rolling_v = df['Volume'].rolling(window=window).sum()
    df['VPIN'] = rolling_v_imb / rolling_v

    # Calculate Liquidity Proxies
    dollar_volume = df['Close'] * df['Volume']
    abs_return = df['Close'].pct_change().abs()

    illiquidity = abs_return / (dollar_volume + 1e-8)
    df['Amihud_Illiq'] = illiquidity.rolling(window=window).mean()

    ret = df['Close'].pct_change()
    lambda_proxy = ret / (dollar_volume + 1e-8)
    df['Kyles_Lambda'] = lambda_proxy.rolling(window=window).mean()

    # Calculate SADF
    prices_vals = df['Close'].values
    sadf_vals = rolling_sadf_np(prices_vals, min_len=30, window=100)
    df['SADF'] = sadf_vals

    return df

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
    # Security Fix: Prevent Path Traversal
    # 1. Resolve project root and allowed config directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    allowed_config_dir = os.path.realpath(os.path.join(project_root, "config"))

    # 2. Resolve path: if simple filename, assume in config/
    if os.path.dirname(config_path) == "":
        target_path = os.path.join(allowed_config_dir, config_path)
    else:
        if not os.path.isabs(config_path):
            target_path = os.path.join(project_root, config_path)
        else:
            target_path = config_path

    # 3. Final Security Validation
    abs_config_path = os.path.realpath(target_path)
    if not abs_config_path.startswith(allowed_config_dir + os.sep) and abs_config_path != allowed_config_dir:
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

    crypto_tickers = [t for t in tickers if t in ['BTC-USD', 'ETH-USD']]
    yfinance_tickers = [t for t in tickers if t not in crypto_tickers]

    data = None
    if yfinance_tickers:
        print(f"Fetching intraday data from yfinance for {yfinance_tickers}...")
        try:
            # Fetch data for all yfinance tickers at once
            data = yf.download(yfinance_tickers, period=f"{data_window_days}d", interval='1h', group_by='ticker')
        except Exception as e:
            print(f"Error fetching data from yfinance: {e}")
            data = None

    exchange = None
    if crypto_tickers:
        exchange = ccxt.binanceus({
            'enableRateLimit': True,
        })

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

        df = None
        if ticker in crypto_tickers:
            # Fetch from ccxt
            symbol_map = {'BTC-USD': 'BTC/USDT', 'ETH-USD': 'ETH/USDT'}
            symbol = symbol_map[ticker]
            print(f"Fetching {data_window_days} days of 1h data for {symbol} from Binance...")

            # 730 days * 24 hours = 17520 bars
            # ccxt limit per fetch is usually 1000 for binance.
            timeframe = '1h'
            limit = 1000

            now = exchange.milliseconds()
            since = now - (data_window_days * 24 * 60 * 60 * 1000)

            all_ohlcv = []

            while since < now:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                    if not ohlcv:
                        break
                    all_ohlcv.extend(ohlcv)
                    # Next fetch starts from the last candle's timestamp + 1 ms to avoid duplicates
                    since = ohlcv[-1][0] + 1
                    time.sleep(exchange.rateLimit / 1000) # Respect rate limit
                except Exception as e:
                    print(f"Error fetching {symbol} from Binance: {e}")
                    break

            if not all_ohlcv:
                print(f"No data fetched for {ticker} from Binance.")
                continue

            df = pd.DataFrame(all_ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['Date'], unit='ms')
            df.set_index('Date', inplace=True)

        else:
            if data is None:
                continue
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

        if df is None or df.empty:
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

        # Refactored microstructural feature generation to separate function
        print(f"Calculating Microstructural Features for {ticker}...")
        df = calculate_microstructural_features(df)

        print(f"Calculating Dynamic Barriers for {ticker}...")
        barriers_df = get_rolling_barriers(df['Close'], window=60, step=20)
        df['Optimal_PT'] = barriers_df['Optimal_PT'].fillna(2.0)
        df['Optimal_SL'] = barriers_df['Optimal_SL'].fillna(2.0)

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

        tech_cols = ['VPIN', 'Amihud_Illiq', 'Kyles_Lambda', 'SADF']
        
        # 1. Drop NaNs FIRST so the split calculations are accurate
        df = df.dropna()

        # 2. Dynamically calculate the split date on healthy data
        split_idx = int(len(df) * TRAIN_SPLIT_RATIO)
        split_date = df.index[split_idx]

        # 3. Create the Train index for PCA
        train_clean_idx = df[df.index < split_date].index
        all_clean_idx = df.index

        # 4. Apply Point-in-Time PCA
        scaler = StandardScaler()
        scaler.fit(df.loc[train_clean_idx, tech_cols])
        scaled_tech = scaler.transform(df.loc[all_clean_idx, tech_cols])
        
        pca = PCA(n_components=4) # Changed from 5 to 4 because we now have 4 features
        scaled_train_tech = scaler.transform(df.loc[train_clean_idx, tech_cols])
        pca.fit(scaled_train_tech)
        pca_features = pca.transform(scaled_tech)

        # Save matrices for Live Inference
        joblib.dump(scaler, f'models/matrices/scaler_{clean_ticker}.pkl')
        joblib.dump(pca, f'models/matrices/pca_{clean_ticker}.pkl')

        pca_cols = ['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4']
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
