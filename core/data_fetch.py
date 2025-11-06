from pathlib import Path
from typing import List, Optional
import pandas as pd
from .logging import get_logger

logger = get_logger("core.data_fetch")

def load_prices_from_csv(csv_path: Path, date_col: str = "date") -> pd.DataFrame:
    """
    CSV wide:
      - colonne 'date'
      - colonnes = tickers
      - valeurs = prix (Adj Close)
    Retourne DataFrame index datetime, colonnes=tickers.
    """
    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"CSV must contain a '{date_col}' column.")
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.set_index(date_col).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    logger.info(f"Loaded {csv_path} with shape {df.shape}")
    return df

def select_tickers(prices: pd.DataFrame, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    if tickers is None:
        return prices
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        logger.warning(f"Tickers not found and ignored: {missing}")
    return prices[[t for t in tickers if t in prices.columns]]
