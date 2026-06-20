from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf


def download_price_data(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1d",
    adjust: bool = True,
) -> pd.DataFrame:
    """Download adjusted close prices for one or more tickers."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=adjust,
        progress=False,
        actions=False,
    )
    if raw.empty:
        raise ValueError("No price data was downloaded for the requested tickers.")

    field = "Close" if adjust else "Adj Close"
    if field not in raw.columns:
        raise ValueError(f"Downloaded data is missing the expected {field!r} field.")

    prices = raw[field].copy()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.reindex(columns=tickers)
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices = prices.sort_index()
    return prices


def validate_price_data(
    prices: pd.DataFrame,
    expected_tickers: list[str] | None = None,
    max_missing_fraction: float = 0.05,
    min_observations: int = 3,
) -> pd.DataFrame:
    """Validate that price series are aligned, complete, and free of duplicate indices."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Price data must be a non-empty DataFrame.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price index must be a DatetimeIndex.")
    if prices.index.duplicated().any():
        raise ValueError("Price index contains duplicate dates.")
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()
    if expected_tickers and set(prices.columns) != set(expected_tickers):
        raise ValueError(f"Expected columns {expected_tickers}, received {list(prices.columns)}.")

    if prices.isna().all(axis=1).any():
        raise ValueError("Price data contains rows with all missing values.")

    if prices.isna().any().any():
        cleaned = prices.dropna(how="any")
        missing_rows = prices.isna().any(axis=1).sum()
        missing_pct = missing_rows / len(prices)
        if missing_pct > max_missing_fraction:
            raise ValueError("Price data contains too many missing values.")
        prices = cleaned

    numeric = prices.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Price data contains non-finite or non-numeric values.")
    if (numeric <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")
    if len(numeric) < min_observations:
        raise ValueError(f"At least {min_observations} aligned observations are required.")

    return numeric


def save_price_data(prices: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)
