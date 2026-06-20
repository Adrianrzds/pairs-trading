import pandas as pd
import pytest

from stat_arb.data import validate_price_data


def test_validate_price_data() -> None:
    prices = pd.DataFrame(
        {"GLD": [1.0, 1.1, 1.2], "SLV": [2.0, 2.1, 2.2]},
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    validated = validate_price_data(prices)
    assert validated.equals(prices)


def test_validate_price_data_removes_small_missing_blocks() -> None:
    prices = pd.DataFrame(
        {
            "GLD": [1.0] * 19 + [None],
            "SLV": [2.0 + i * 0.1 for i in range(20)],
        },
        index=pd.date_range("2020-01-01", periods=20, freq="D"),
    )
    validated = validate_price_data(prices)
    assert validated.shape == (19, 2)


def test_validate_price_data_rejects_nonpositive_prices() -> None:
    prices = pd.DataFrame(
        {"GLD": [1.0, 0.0, 1.2], "SLV": [2.0, 2.1, 2.2]},
        index=pd.date_range("2020-01-01", periods=3),
    )
    with pytest.raises(ValueError, match="strictly positive"):
        validate_price_data(prices)
