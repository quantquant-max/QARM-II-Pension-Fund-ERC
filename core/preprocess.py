import numpy as np
import pandas as pd

def clip_date_range(prices: pd.DataFrame, start: str = None, end: str = None) -> pd.DataFrame:
    p = prices
    if start:
        p = p[p.index >= pd.to_datetime(start)]
    if end:
        p = p[p.index <= pd.to_datetime(end)]
    return p.sort_index()

def align_on_intersection(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.dropna(how="any")

def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna()

def sanity_check_returns(returns: pd.DataFrame, min_rows: int = 50, min_cols: int = 2):
    if returns.shape[0] < min_rows or returns.shape[1] < min_cols:
        raise ValueError(f"Not enough data: rows={returns.shape[0]}, cols={returns.shape[1]}")
