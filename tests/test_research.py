import numpy as np
import pandas as pd

from stat_arb.research import (
    PairResearchConfig,
    build_pair_research,
    estimate_walk_forward_parameters,
    rolling_sharpe,
)


def _sample_pair(rows: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2022-01-03", periods=rows)
    x_log = 3.0 + np.cumsum(rng.normal(0.0, 0.01, rows))
    residual = rng.normal(0.0, 0.01, rows)
    y_log = 1.0 + 0.8 * x_log + residual
    return pd.DataFrame({"AAA": np.exp(y_log), "BBB": np.exp(x_log)}, index=index)


def test_walk_forward_parameters_use_only_prior_observations() -> None:
    prices = _sample_pair()
    logs = np.log(prices)
    original = estimate_walk_forward_parameters(
        logs["AAA"],
        logs["BBB"],
        method="expanding",
        min_observations=80,
    )
    changed = logs.copy()
    changed.loc[changed.index[80] :, "AAA"] += 100.0
    contaminated = estimate_walk_forward_parameters(
        changed["AAA"],
        changed["BBB"],
        method="expanding",
        min_observations=80,
    )

    first_estimate_date = logs.index[80]
    assert original.loc[first_estimate_date].equals(contaminated.loc[first_estimate_date])


def test_build_pair_research_outputs_current_signal_and_metrics() -> None:
    prices = _sample_pair()
    result = build_pair_research(
        prices,
        PairResearchConfig(
            y_asset="AAA",
            x_asset="BBB",
            min_observations=80,
            zscore_window=20,
            entry_threshold=1.5,
            exit_threshold=0.25,
        ),
    )

    assert result.current_signal["signal"] in {"long_spread", "short_spread", "flat"}
    assert "max_drawdown" in result.backtest.metrics
    assert result.backtest.ledger["turnover"].sum() >= 0.0


def test_rolling_sharpe_shape() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, 0.0], index=pd.date_range("2024-01-01", periods=4))
    sharpe = rolling_sharpe(returns, window=2)
    assert sharpe.shape == returns.shape
    assert sharpe.isna().sum() == 1
