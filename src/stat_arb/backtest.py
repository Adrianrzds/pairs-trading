from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    daily_returns: pd.Series
    ledger: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


def generate_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> pd.DataFrame:
    """Create a persistent spread position from close-observed z-scores."""
    if not 0 <= exit_threshold < entry_threshold:
        raise ValueError("Thresholds must satisfy 0 <= exit < entry.")
    position = 0
    values: list[int] = []
    for z in zscore:
        if pd.isna(z):
            values.append(position)
            continue
        if position == 0:
            if z <= -entry_threshold:
                position = 1
            elif z >= entry_threshold:
                position = -1
        elif position == 1 and z >= -exit_threshold:
            position = 0
        elif position == -1 and z <= exit_threshold:
            position = 0
        values.append(position)
    return pd.DataFrame({"zscore": zscore, "position": values}, index=zscore.index)


def _trade_log(ledger: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    position = ledger["position"]
    starts = ledger.index[(position != 0) & (position.shift(fill_value=0) == 0)]
    for start in starts:
        side = int(position.loc[start])
        later = position.loc[position.index > start]
        exits = later.index[later == 0]
        end = exits[0] if len(exits) else ledger.index[-1]
        period = ledger.loc[start:end, "net_return"]
        records.append(
            {
                "entry_date": start,
                "exit_date": end if len(exits) else pd.NaT,
                "side": "long_spread" if side == 1 else "short_spread",
                "holding_days": max(0, len(period) - 1),
                "net_return": float((1.0 + period).prod() - 1.0),
                "closed": bool(len(exits)),
            }
        )
    return pd.DataFrame.from_records(records)


def simulate_pair_backtest(
    prices: pd.DataFrame,
    hedge_ratio: pd.Series,
    signals: pd.DataFrame,
    tx_cost: float = 0.0005,
) -> BacktestResult:
    """Backtest beta-weighted, gross-one pair positions with next-return execution logic.

    A signal observed at close t sets portfolio weights for t to t+1. Costs are charged
    on changes in both leg weights. ``tx_cost`` is the one-way cost per dollar traded.
    """
    if prices.shape[1] != 2:
        raise ValueError("Pair backtests require exactly two price columns.")
    if tx_cost < 0:
        raise ValueError("Transaction cost cannot be negative.")
    index = prices.index
    beta = hedge_ratio.reindex(index).ffill()
    position = signals["position"].reindex(index).fillna(0).astype(int)
    if beta.isna().any():
        raise ValueError("Hedge ratio must be available for every backtest date.")

    gross = 1.0 + beta.abs()
    y_weight = position / gross
    x_weight = -position * beta / gross
    weights = pd.DataFrame({prices.columns[0]: y_weight, prices.columns[1]: x_weight})
    asset_returns = prices.pct_change().fillna(0.0)
    gross_return = (weights.shift(1).fillna(0.0) * asset_returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    costs = turnover * tx_cost
    net_return = gross_return - costs
    equity = (1.0 + net_return).cumprod().rename("equity")

    ledger = pd.DataFrame(
        {
            "position": position,
            "hedge_ratio": beta,
            "y_weight": y_weight,
            "x_weight": x_weight,
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": costs,
            "net_return": net_return,
            "equity": equity,
        }
    )
    trades = _trade_log(ledger)
    metrics = calculate_performance_metrics(equity, net_return, turnover, trades)
    return BacktestResult(equity, net_return.rename("net_return"), ledger, trades, metrics)


def calculate_performance_metrics(
    equity: pd.Series,
    returns: pd.Series,
    turnover: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
    annual_factor: int = 252,
) -> dict[str, float]:
    """Calculate annualized and trade-level performance statistics."""
    if returns.empty:
        return {}
    ending_equity = float(equity.iloc[-1])
    cagr = ending_equity ** (annual_factor / len(returns)) - 1 if ending_equity > 0 else -1.0
    volatility = returns.std(ddof=1)
    downside_deviation = np.sqrt(np.mean(np.minimum(returns.to_numpy(), 0.0) ** 2))
    sharpe = np.sqrt(annual_factor) * returns.mean() / volatility if volatility > 0 else np.nan
    sortino = (
        np.sqrt(annual_factor) * returns.mean() / downside_deviation
        if downside_deviation > 0
        else np.nan
    )
    max_drawdown = float((equity / equity.cummax() - 1.0).min())
    closed = trades.loc[trades["closed"]] if trades is not None and not trades.empty else None
    hit_rate = (
        float((closed["net_return"] > 0).mean()) if closed is not None and len(closed) else np.nan
    )
    return {
        "cumulative_return": ending_equity - 1.0,
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "turnover": float(turnover.sum()) if turnover is not None else np.nan,
        "trade_count": int(len(trades)) if trades is not None else 0,
    }
