import pandas as pd
import numpy as np

from stat_arb.backtest import generate_signals, simulate_pair_backtest


def test_generate_signals() -> None:
    zscore = pd.Series([-3.0, -2.5, -1.5, -0.2, 0.1, 2.1, 1.0, 0.0], index=pd.date_range("2020-01-01", periods=8))
    signals = generate_signals(zscore)
    assert signals.loc["2020-01-01", "position"] == 1
    assert signals.loc["2020-01-06", "position"] == -1


def test_simulate_pair_backtest() -> None:
    prices = pd.DataFrame({"GLD": [100.0, 101.0, 102.0], "SLV": [20.0, 20.2, 20.4]}, index=pd.date_range("2020-01-01", periods=3))
    hedge_ratio = pd.Series(5.0, index=prices.index)
    zscore = pd.Series([3.0, 1.0, -3.0], index=prices.index)
    signals = generate_signals(zscore)
    result = simulate_pair_backtest(prices, hedge_ratio, signals)
    assert "cagr" in result.metrics
    assert result.equity.shape[0] == 3


def test_signal_is_applied_to_next_period_and_costs_cover_both_legs() -> None:
    index = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"GLD": [100.0, 110.0, 110.0], "SLV": [100.0, 100.0, 100.0]}, index=index)
    signals = pd.DataFrame({"position": [1, 1, 0]}, index=index)
    result = simulate_pair_backtest(
        prices,
        pd.Series(1.0, index=index),
        signals,
        tx_cost=0.001,
    )
    # Entry at the first close costs 10 bps on gross-one exposure; the 10% GLD move
    # earns 5% the following day because each leg has 50% absolute weight.
    assert np.isclose(result.ledger.iloc[0]["transaction_cost"], 0.001)
    assert np.isclose(result.ledger.iloc[1]["gross_return"], 0.05)
    assert result.metrics["trade_count"] == 1
    assert np.isclose(result.metrics["turnover"], 2.0)
