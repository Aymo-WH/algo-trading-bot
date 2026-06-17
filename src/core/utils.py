import pandas as pd
import json
import os

_CONFIG_CACHE = {}

def load_config(config_path: str = 'config/config_phase1.json') -> dict:
    """
    Loads configuration settings from the given config JSON, cached per path.

    The config controls universal simulation parameters such as the transaction
    fee percentage. Results are cached per path in `_CONFIG_CACHE` to prevent
    repetitive disk reads while still allowing different configs (phase1, crypto,
    macro) to be loaded independently within the same process.

    Args:
        config_path (str): Path to the config JSON. Defaults to config/config_phase1.json.

    Returns:
        dict: A dictionary of configuration parameters. Returns an empty dict if the file is missing.
    """

    if config_path in _CONFIG_CACHE:
        return _CONFIG_CACHE[config_path]

    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cfg = {}

    _CONFIG_CACHE[config_path] = cfg
    return cfg

def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens a multi-index column structure into a single level.

    Often, the yfinance library returns DataFrames with MultiIndex columns (e.g., when
    downloading multiple tickers). This function safely attempts to drop the second level
    or flatten the tuples into a single string to ensure compatibility with standard pipelines.

    Args:
        df (pd.DataFrame): The DataFrame with potential MultiIndex columns.

    Returns:
        pd.DataFrame: A DataFrame with flattened, single-level columns.
    """

    if isinstance(df.columns, pd.MultiIndex):
        try:
            # We assume the first level is Price and second is Ticker
            # We can drop the Ticker level if it's just one ticker
            df.columns = df.columns.droplevel(1)
        except (IndexError, ValueError):
            pass
    return df

def load_agent(model_path: str):
    """
    Loads a trained model from disk. Supports Stable-Baselines3 (PPO) and XGBoost.

    Args:
        model_path (str): The path to the saved model file (.zip, .pkl, or .json).

    Returns:
        The loaded model instance.

    Raises:
        ValueError: If the model type is unknown.
    """
    if "ppo" in model_path.lower():
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    elif "xgb" in model_path.lower():
        import joblib
        return joblib.load(model_path)
    else:
        raise ValueError(f"Unknown model type for {model_path}")
