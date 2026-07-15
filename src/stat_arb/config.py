from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

DEFAULT_UNIVERSE = [
    "SPY",
    "IVV",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "LQD",
    "HYG",
    "GLD",
    "SLV",
    "USO",
    "UNG",
    "XLE",
    "XOP",
    "XLF",
    "XLK",
    "XLV",
    "XLI",
    "XLP",
    "XLY",
    "XLU",
]


@dataclass(frozen=True)
class CaseStudy:
    name: str
    y_asset: str
    x_asset: str
    theme: str
    thesis: str


CASE_STUDIES = [
    CaseStudy(
        name="ETF near-clones: SPY / IVV",
        y_asset="SPY",
        x_asset="IVV",
        theme="Expected cointegration",
        thesis=(
            "Both ETFs track the S&P 500, so this is the cleanest example of a spread "
            "that should be tightly anchored by the same underlying benchmark."
        ),
    ),
    CaseStudy(
        name="Energy sector: XLE / XOP",
        y_asset="XLE",
        x_asset="XOP",
        theme="Same sector, different exposures",
        thesis=(
            "Both ETFs are tied to the energy sector, but XLE is weighted toward large "
            "integrated energy companies while XOP is more exploration and production focused."
        ),
    ),
    CaseStudy(
        name="Credit risk: LQD / HYG",
        y_asset="LQD",
        x_asset="HYG",
        theme="Related fixed-income risk, different credit quality",
        thesis=(
            "Investment-grade and high-yield corporate bond ETFs share rate and credit-cycle "
            "drivers, but their spread can move differently during risk-off regimes."
        ),
    ),
    CaseStudy(
        name="Metals regime case: GLD / SLV",
        y_asset="GLD",
        x_asset="SLV",
        theme="Related but regime-dependent",
        thesis=(
            "Gold and silver are related metals, but gold often behaves like a safe-haven "
            "asset while silver has more industrial demand exposure. Cointegration can break."
        ),
    ),
]


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path = Path("data")
    duckdb_path: Path = Path("data/research.duckdb")
    default_universe: tuple[str, ...] = tuple(DEFAULT_UNIVERSE)


def load_app_config() -> AppConfig:
    """Load dashboard defaults from environment variables."""
    data_dir = Path(os.getenv("STAT_ARB_DATA_DIR", "data"))
    duckdb_path = Path(os.getenv("STAT_ARB_DUCKDB_PATH", str(data_dir / "research.duckdb")))
    raw_universe = os.getenv("STAT_ARB_DEFAULT_UNIVERSE", ",".join(DEFAULT_UNIVERSE))
    universe = tuple(ticker.strip().upper() for ticker in raw_universe.split(",") if ticker.strip())
    return AppConfig(data_dir=data_dir, duckdb_path=duckdb_path, default_universe=universe)
