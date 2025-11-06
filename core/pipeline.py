from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .logging import get_logger
from .data_fetch import load_prices_from_csv, select_tickers
from .preprocess import clip_date_range, align_on_intersection, to_log_returns, sanity_check_returns
from .erc import erc_weights

logger = get_logger("core.pipeline")

def optimize_erc_from_csv(
    csv_path: str,
    tickers: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    annualize_factor: float = 52.0,
) -> Dict[str, Any]:
    prices = load_prices_from_csv(Path(csv_path))
    if tickers:
        prices = select_tickers(prices, tickers)
    prices = clip_date_range(prices, start, end)
    prices = align_on_intersection(prices)
    rets = to_log_returns(prices)
    sanity_check_returns(rets)

    lw = LedoitWolf().fit(rets.values)
    Sigma = lw.covariance_ * annualize_factor

    w = erc_weights(Sigma)  # placeholder = égal
    tickers_final = list(prices.columns)

    sigma_p = float(np.sqrt(w @ Sigma @ w))
    mrc = (Sigma @ w) / sigma_p
    rc = w * mrc
    rc_pct = rc / rc.sum()

    logger.info(f"OK optimize placeholder | n={len(tickers_final)} | sigma={sigma_p:.4f}")

    return {
        "tickers": tickers_final,
        "weights": w,
        "sigma": sigma_p,
        "rc_pct": rc_pct,
        "Sigma": Sigma,
        "returns": rets,
    }
