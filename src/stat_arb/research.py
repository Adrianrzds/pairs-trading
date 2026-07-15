from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Literal

import numpy as np
import pandas as pd

from .backtest import BacktestResult, generate_signals, simulate_pair_backtest
from .data import validate_price_data
from .preprocess import compute_log_prices
from .stats import (
    compute_walk_forward_spread_zscore,
    engle_granger_test,
    estimate_ols_parameters,
)

LOGGER = logging.getLogger(__name__)
HedgeMethod = Literal["expanding", "rolling"]


@dataclass(frozen=True)
class PairResearchConfig:
    y_asset: str
    x_asset: str
    hedge_method: HedgeMethod = "expanding"
    lookback: int = 252
    min_observations: int = 60
    zscore_window: int = 60
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    transaction_cost_bps: float = 5.0
    position_size: float = 1.0


@dataclass(frozen=True)
class PairResearchResult:
    config: PairResearchConfig
    prices: pd.DataFrame
    log_prices: pd.DataFrame
    parameters: pd.DataFrame
    spread: pd.Series
    zscore: pd.Series
    signals: pd.DataFrame
    cointegration: dict[str, float]
    backtest: BacktestResult
    current_signal: dict[str, object]


def estimate_walk_forward_parameters(
    y: pd.Series,
    x: pd.Series,
    method: HedgeMethod = "expanding",
    lookback: int = 252,
    min_observations: int = 60,
) -> pd.DataFrame:
    """Estimate OLS parameters for each date using observations strictly before that date."""
    if method not in {"expanding", "rolling"}:
        raise ValueError("Hedge method must be 'expanding' or 'rolling'.")
    if min_observations < 2:
        raise ValueError("At least two observations are required for OLS.")
    if method == "rolling" and lookback < min_observations:
        raise ValueError("Rolling lookback must be at least min_observations.")

    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    parameters = pd.DataFrame(
        index=aligned.index, columns=["intercept", "hedge_ratio"], dtype=float
    )
    for date in aligned.index:
        history = aligned.loc[aligned.index < date]
        if method == "rolling":
            history = history.tail(lookback)
        if len(history) < min_observations:
            continue
        intercept, beta = estimate_ols_parameters(history["y"], history["x"])
        parameters.loc[date] = [intercept, beta]
    return parameters


def build_pair_research(prices: pd.DataFrame, config: PairResearchConfig) -> PairResearchResult:
    """Run cointegration, signal construction, and backtest for one selected pair."""
    tickers = [config.y_asset.upper(), config.x_asset.upper()]
    if tickers[0] == tickers[1]:
        raise ValueError("Select two different assets.")

    prices = validate_price_data(prices[tickers], expected_tickers=tickers, min_observations=100)
    log_prices = compute_log_prices(prices)
    LOGGER.info("Running pair research for %s/%s over %d rows", *tickers, len(prices))

    parameters = estimate_walk_forward_parameters(
        log_prices[tickers[0]],
        log_prices[tickers[1]],
        method=config.hedge_method,
        lookback=config.lookback,
        min_observations=config.min_observations,
    ).dropna()
    if len(parameters) < config.zscore_window:
        raise ValueError("Not enough post-warmup observations for the requested z-score window.")

    aligned_log_prices = log_prices.reindex(parameters.index)
    spread_stats = compute_walk_forward_spread_zscore(
        aligned_log_prices[tickers[0]],
        aligned_log_prices[tickers[1]],
        parameters,
        window=config.zscore_window,
    )
    usable = spread_stats.dropna().index
    if usable.empty:
        raise ValueError("No usable z-score observations were produced.")

    parameters = parameters.loc[usable]
    spread = spread_stats.loc[usable, "spread"].rename("spread")
    zscore = spread_stats.loc[usable, "zscore"].rename("zscore")
    signals = generate_signals(
        zscore,
        entry_threshold=config.entry_threshold,
        exit_threshold=config.exit_threshold,
    )
    cointegration = engle_granger_test(log_prices[tickers[0]], log_prices[tickers[1]])
    backtest = simulate_pair_backtest(
        prices.loc[usable, tickers],
        parameters["hedge_ratio"],
        signals,
        tx_cost=config.transaction_cost_bps / 10_000,
        position_size=config.position_size,
    )
    current_signal = describe_current_signal(zscore, signals, parameters)
    return PairResearchResult(
        config=config,
        prices=prices.loc[usable, tickers],
        log_prices=aligned_log_prices.loc[usable, tickers],
        parameters=parameters,
        spread=spread,
        zscore=zscore,
        signals=signals,
        cointegration=cointegration,
        backtest=backtest,
        current_signal=current_signal,
    )


def describe_current_signal(
    zscore: pd.Series,
    signals: pd.DataFrame,
    parameters: pd.DataFrame,
) -> dict[str, object]:
    """Describe the latest close-based signal without assuming future execution."""
    latest_date = zscore.dropna().index[-1]
    position = int(signals.loc[latest_date, "position"])
    if position > 0:
        label = "long_spread"
    elif position < 0:
        label = "short_spread"
    else:
        label = "flat"
    return {
        "as_of": latest_date,
        "signal": label,
        "position": position,
        "zscore": float(zscore.loc[latest_date]),
        "hedge_ratio": float(parameters.loc[latest_date, "hedge_ratio"]),
        "execution_note": (
            "Signal is observed at the latest close and would execute on the next session."
        ),
    }


def rolling_sharpe(returns: pd.Series, window: int = 63, annual_factor: int = 252) -> pd.Series:
    """Annualized rolling Sharpe ratio."""
    if window < 2:
        raise ValueError("Rolling Sharpe window must be at least two observations.")
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=1).replace(0, np.nan)
    return (np.sqrt(annual_factor) * mean / std).rename("rolling_sharpe")
