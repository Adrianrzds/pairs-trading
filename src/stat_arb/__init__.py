"""Statistical arbitrage utilities for cointegrated pair trading."""

from .data import download_price_data, validate_price_data
from .preprocess import compute_log_prices, compute_log_returns, compute_correlation
from .stats import (
    engle_granger_test,
    estimate_ols_parameters,
    estimate_hedge_ratio_ols,
    construct_spread,
    rolling_hedge_ratio,
    walk_forward_ols,
    compute_zscore,
    compute_walk_forward_spread_zscore,
)
from .backtest import (
    generate_signals,
    simulate_pair_backtest,
    calculate_performance_metrics,
    BacktestResult,
)
from .research import (
    PairResearchConfig,
    PairResearchResult,
    build_pair_research,
    estimate_walk_forward_parameters,
    rolling_sharpe,
)

__all__ = [
    "download_price_data",
    "validate_price_data",
    "compute_log_prices",
    "compute_log_returns",
    "compute_correlation",
    "engle_granger_test",
    "estimate_ols_parameters",
    "estimate_hedge_ratio_ols",
    "construct_spread",
    "rolling_hedge_ratio",
    "walk_forward_ols",
    "compute_zscore",
    "compute_walk_forward_spread_zscore",
    "generate_signals",
    "simulate_pair_backtest",
    "calculate_performance_metrics",
    "BacktestResult",
    "PairResearchConfig",
    "PairResearchResult",
    "build_pair_research",
    "estimate_walk_forward_parameters",
    "rolling_sharpe",
]
