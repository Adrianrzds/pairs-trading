import pandas as pd
import numpy as np

from stat_arb.stats import (
    estimate_hedge_ratio_ols,
    construct_spread,
    compute_zscore,
    walk_forward_ols,
    compute_walk_forward_spread_zscore,
)


def test_ols_hedge_ratio_and_spread() -> None:
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    y = pd.Series([2.0, 4.0, 6.0, 8.0])
    hedge_ratio = estimate_hedge_ratio_ols(y, x)
    assert abs(hedge_ratio - 2.0) < 1e-8
    spread = construct_spread(y, x, hedge_ratio)
    assert all(abs(spread) < 1e-8)


def test_compute_zscore() -> None:
    series = pd.Series(np.arange(1.0, 11.0))
    z = compute_zscore(series, window=5)
    assert z.shape == series.shape
    assert z.isna().sum() == 4


def test_walk_forward_ols_does_not_use_current_or_future_data() -> None:
    index = pd.date_range("2020-01-01", periods=80)
    x = pd.Series(np.linspace(1, 5, 80), index=index)
    y = 1.0 + 2.0 * x
    original = walk_forward_ols(y, x, trading_start=index[40], refit_frequency=10)

    changed_y = y.copy()
    changed_y.loc[index[40] :] += 1000
    changed = walk_forward_ols(changed_y, x, trading_start=index[40], refit_frequency=10)
    assert np.isclose(original.loc[index[40], "hedge_ratio"], 2.0)
    assert original.loc[index[40]].equals(changed.loc[index[40]])


def test_walk_forward_zscore_recomputes_window_with_current_parameters() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=index)
    y = pd.Series([2.0, 4.2, 5.8, 8.5, 9.7, 12.4], index=index)
    parameters = pd.DataFrame(
        {
            "intercept": [0.0] * 6,
            "hedge_ratio": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        },
        index=index,
    )

    result = compute_walk_forward_spread_zscore(y, x, parameters, window=3)
    expected_window = y.iloc[1:4] - 2.0 * x.iloc[1:4]
    expected = (expected_window.iloc[-1] - expected_window.mean()) / expected_window.std(ddof=1)

    assert np.isclose(result.loc[index[3], "zscore"], expected)
    assert result["zscore"].isna().sum() == 2
