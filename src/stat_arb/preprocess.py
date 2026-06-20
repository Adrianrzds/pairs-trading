from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log prices for each series."""
    if (prices <= 0).any().any():
        raise ValueError("Log prices require strictly positive inputs.")
    return np.log(prices.astype(float))


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price series."""
    return compute_log_prices(prices).diff().dropna()


def compute_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix for return series."""
    return returns.corr()
