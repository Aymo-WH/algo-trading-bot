import pandas as pd
import json
import os

_CONFIG_CACHE = None

def load_config() -> dict:
    """
    Loads configuration settings from config/config_phase1.json with a module-level cache.

    The config.json file controls universal simulation parameters such as the transaction
    fee percentage. Once loaded during execution, the result is cached in `_CONFIG_CACHE`
    to prevent repetitive disk reads.

    Returns:
        dict: A dictionary of configuration parameters. Returns an empty dict if the file is missing.
    """

    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    try:
        # Assuming config.json is in the root directory relative to execution
        with open('config/config_phase1.json', 'r') as f:
            _CONFIG_CACHE = json.load(f)
            return _CONFIG_CACHE
    except (FileNotFoundError, json.JSONDecodeError):
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE

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
        model_path (str): The path to the saved model file (.zip or .json).

    Returns:
        The loaded model instance.

    Raises:
        ValueError: If the model type is unknown.
    """
    if "ppo" in model_path.lower():
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    elif "xgb" in model_path.lower():
        import xgboost as xgb
        # Assuming we save the XGBClassifier model using native save_model to .json
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model
    else:
        raise ValueError(f"Unknown model type for {model_path}")
