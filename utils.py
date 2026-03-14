import pandas as pd
import json
import os

_CONFIG_CACHE = None

def load_config() -> dict:
    """
    Loads configuration from config.json, returns empty dict if not found.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    try:
        # Assuming config.json is in the root directory relative to execution
        with open('config.json', 'r') as f:
            _CONFIG_CACHE = json.load(f)
            return _CONFIG_CACHE
    except FileNotFoundError:
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE

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
        except (IndexError, ValueError):
            pass
    return df
