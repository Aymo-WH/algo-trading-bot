import pandas as pd
import json
import os

def load_config() -> dict:
    """
    Loads configuration from config.json, returns empty dict if not found.
    """
    try:
        # Assuming config.json is in the root directory relative to execution
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if a DataFrame has MultiIndex columns (common in new yfinance)
    and drops the second level (Ticker) if so.

    Args:
        df (pd.DataFrame): The DataFrame to check and modify.

    Returns:
        pd.DataFrame: The modified DataFrame with flattened columns if applicable.
    """
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # We assume the first level is Price and second is Ticker
            # We can drop the Ticker level if it's just one ticker
            df.columns = df.columns.droplevel(1)
        except Exception:
            pass
    return df
