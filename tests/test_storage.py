import pandas as pd

from stat_arb.storage import PriceCache


def test_price_cache_round_trip(tmp_path) -> None:
    prices = pd.DataFrame(
        {
            "GLD": [100.0, 101.0, 102.0],
            "SLV": [20.0, 20.5, 20.25],
        },
        index=pd.date_range("2024-01-02", periods=3, freq="B"),
    )
    cache = PriceCache(data_dir=tmp_path / "data", duckdb_path=tmp_path / "research.duckdb")

    cache.write_prices(prices)
    loaded = cache.load_prices(["SLV", "GLD"], start="2024-01-03")

    assert cache.available_tickers() == ["GLD", "SLV"]
    assert list(loaded.columns) == ["SLV", "GLD"]
    assert loaded.index.min() == pd.Timestamp("2024-01-03")
    assert loaded.loc[pd.Timestamp("2024-01-04"), "GLD"] == 102.0
