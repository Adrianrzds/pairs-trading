from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import duckdb
import pandas as pd

_SAFE_TICKER = re.compile(r"[^A-Z0-9._-]+")


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


@dataclass(frozen=True)
class PriceCache:
    """Local DuckDB/Parquet cache for adjusted close price data."""

    data_dir: Path = Path("data")
    duckdb_path: Path = Path("data/research.duckdb")

    @property
    def parquet_dir(self) -> Path:
        return self.data_dir / "raw"

    def ensure(self) -> None:
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    def _ticker_path(self, ticker: str) -> Path:
        safe = _SAFE_TICKER.sub("_", ticker.upper())
        return self.parquet_dir / f"{safe}.parquet"

    def write_prices(self, prices: pd.DataFrame, source: str = "yfinance") -> None:
        """Persist wide adjusted-close data as one Parquet file per ticker."""
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("Price cache requires a DatetimeIndex.")
        self.ensure()
        downloaded_at = pd.Timestamp.now("UTC")
        for ticker in prices.columns:
            frame = (
                prices[[ticker]]
                .dropna()
                .rename(columns={ticker: "adj_close"})
                .reset_index(names="date")
            )
            frame["ticker"] = str(ticker).upper()
            frame["source"] = source
            frame["downloaded_at"] = downloaded_at
            frame = frame[["date", "ticker", "adj_close", "source", "downloaded_at"]]
            frame.to_parquet(self._ticker_path(str(ticker)), index=False)
        self.refresh_catalog()

    def refresh_catalog(self) -> None:
        """Create or refresh the DuckDB view over cached Parquet files."""
        self.ensure()
        parquet_glob = _sql_literal(str(self.parquet_dir / "*.parquet"))
        with duckdb.connect(str(self.duckdb_path)) as connection:
            connection.execute(f"""
                CREATE OR REPLACE VIEW prices AS
                SELECT
                    CAST(date AS DATE) AS date,
                    ticker,
                    CAST(adj_close AS DOUBLE) AS adj_close,
                    source,
                    downloaded_at
                FROM read_parquet({parquet_glob})
                """)

    def available_tickers(self) -> list[str]:
        self.ensure()
        files = sorted(self.parquet_dir.glob("*.parquet"))
        if not files:
            return []
        self.refresh_catalog()
        with duckdb.connect(str(self.duckdb_path), read_only=True) as connection:
            return [
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT ticker FROM prices ORDER BY ticker"
                ).fetchall()
            ]

    def load_prices(
        self,
        tickers: list[str] | tuple[str, ...],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Load cached adjusted closes as a validated wide price table."""
        if not tickers:
            raise ValueError("At least one ticker is required.")
        self.refresh_catalog()
        params: list[object] = [[ticker.upper() for ticker in tickers]]
        where = ["ticker IN (SELECT UNNEST(?))"]
        if start is not None:
            where.append("date >= ?")
            params.append(pd.Timestamp(start).date())
        if end is not None:
            where.append("date <= ?")
            params.append(pd.Timestamp(end).date())
        query = f"""
            SELECT date, ticker, adj_close
            FROM prices
            WHERE {' AND '.join(where)}
            ORDER BY date, ticker
        """
        with duckdb.connect(str(self.duckdb_path), read_only=True) as connection:
            long_prices = connection.execute(query, params).fetchdf()
        if long_prices.empty:
            return pd.DataFrame()
        wide = long_prices.pivot(index="date", columns="ticker", values="adj_close")
        wide.index = pd.DatetimeIndex(wide.index).tz_localize(None).normalize()
        return wide.reindex(columns=[ticker.upper() for ticker in tickers]).sort_index()
