from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import pandas as pd

from .backtest import generate_signals, simulate_pair_backtest
from .data import download_price_data, save_price_data, validate_price_data
from .plot import plot_equity_curve, plot_price_series, plot_spread_and_zscore
from .preprocess import compute_correlation, compute_log_prices, compute_log_returns
from .stats import (
    construct_spread,
    engle_granger_test,
    estimate_ols_parameters,
    walk_forward_ols,
    compute_zscore,
)


@dataclass(frozen=True)
class ResearchConfig:
    start: str = "2010-01-01"
    end: str = "2026-01-01"
    formation_end: str = "2017-12-31"
    zscore_window: int = 60
    refit_frequency: int = 21
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    transaction_cost_bps: float = 5.0


def run_research(
    prices: pd.DataFrame,
    config: ResearchConfig,
    output_dir: str | Path = "outputs",
) -> dict[str, object]:
    """Run and persist the fixed-parameter GLD/SLV out-of-sample study."""
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    prices = validate_price_data(prices, expected_tickers=["GLD", "SLV"], min_observations=100)
    log_prices = compute_log_prices(prices)
    returns = compute_log_returns(prices)
    cutoff = pd.Timestamp(config.formation_end)
    formation = log_prices.loc[:cutoff]
    trading = log_prices.loc[log_prices.index > cutoff]
    if len(formation) < config.zscore_window or len(trading) < config.zscore_window:
        raise ValueError("Formation and trading periods must each exceed the z-score window.")
    trading_start = trading.index[0]

    formation_coint = engle_granger_test(formation["GLD"], formation["SLV"])
    # This second test is reported only as an ex-post stability diagnostic, never as a trading filter.
    trading_coint = engle_granger_test(trading["GLD"], trading["SLV"])
    intercept, beta = estimate_ols_parameters(formation["GLD"], formation["SLV"])
    parameters = walk_forward_ols(
        log_prices["GLD"],
        log_prices["SLV"],
        trading_start=trading_start,
        refit_frequency=config.refit_frequency,
    )
    trading_spread = construct_spread(
        trading["GLD"],
        trading["SLV"],
        parameters["hedge_ratio"],
        parameters["intercept"],
    ).rename("spread")
    zscore = compute_zscore(trading_spread, config.zscore_window).rename("zscore")
    signals = generate_signals(zscore, config.entry_threshold, config.exit_threshold)
    result = simulate_pair_backtest(
        prices.loc[trading.index],
        parameters["hedge_ratio"],
        signals,
        tx_cost=config.transaction_cost_bps / 10_000,
    )

    formation_returns = returns.loc[returns.index <= cutoff]
    trading_returns = returns.loc[returns.index > cutoff]
    correlation = pd.concat(
        {
            "formation": compute_correlation(formation_returns).stack(),
            "trading_ex_post": compute_correlation(trading_returns).stack(),
        },
        axis=1,
    )
    coint_table = pd.DataFrame(
        [formation_coint, trading_coint], index=["formation", "trading_ex_post"]
    )
    formation_summary = pd.Series(
        {
            "formation_start": formation.index.min().date().isoformat(),
            "formation_end": formation.index.max().date().isoformat(),
            "trading_start": trading.index.min().date().isoformat(),
            "trading_end": trading.index.max().date().isoformat(),
            "formation_observations": len(formation),
            "trading_observations": len(trading),
            "formation_intercept": intercept,
            "formation_hedge_ratio": beta,
        },
        name="value",
    )

    pd.Series(result.metrics, name="value").to_csv(tables / "performance_metrics.csv")
    formation_summary.to_csv(tables / "sample_summary.csv")
    correlation.to_csv(tables / "return_correlations.csv")
    coint_table.to_csv(tables / "cointegration_tests.csv")
    parameters.to_csv(tables / "walk_forward_parameters.csv")
    pd.concat([trading_spread, zscore, signals["position"]], axis=1).to_csv(
        tables / "spread_and_signals.csv"
    )
    result.ledger.to_csv(tables / "daily_backtest_ledger.csv")
    result.trades.to_csv(tables / "trade_log.csv", index=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)

    plot_price_series(prices, figures / "normalized_prices.png")
    plot_spread_and_zscore(
        trading_spread,
        zscore,
        figures / "spread_and_zscore.png",
        config.entry_threshold,
        config.exit_threshold,
    )
    plot_equity_curve(result.equity, figures / "equity_and_drawdown.png")
    return {
        "metrics": result.metrics,
        "formation_cointegration": formation_coint,
        "trading_cointegration_ex_post": trading_coint,
        "result": result,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GLD/SLV statistical-arbitrage study.")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-01-01", help="Exclusive data end date")
    parser.add_argument("--formation-end", default="2017-12-31")
    parser.add_argument("--input", type=Path, help="Optional local adjusted-close CSV")
    parser.add_argument("--data-path", type=Path, default=Path("data/gld_slv_adjusted_close.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ResearchConfig(start=args.start, end=args.end, formation_end=args.formation_end)
    if args.input:
        prices = pd.read_csv(args.input, index_col=0, parse_dates=True)
    else:
        prices = download_price_data(["GLD", "SLV"], config.start, config.end)
        save_price_data(prices, args.data_path)
    study = run_research(prices, config, args.output_dir)
    metrics = pd.Series(study["metrics"])
    print(metrics.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nArtifacts saved under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
