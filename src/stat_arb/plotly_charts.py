from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .research import PairResearchResult, rolling_sharpe

PLOT_TEMPLATE = "plotly_white"


def price_chart(result: PairResearchResult) -> go.Figure:
    normalized = result.prices / result.prices.iloc[0]
    fig = go.Figure()
    for ticker in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[ticker],
                mode="lines",
                name=ticker,
            )
        )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Normalized adjusted closes",
        yaxis_title="Growth of $1",
        legend_orientation="h",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def spread_zscore_chart(result: PairResearchResult) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(
        go.Scatter(x=result.spread.index, y=result.spread, mode="lines", name="Spread"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=result.zscore.index, y=result.zscore, mode="lines", name="Z-score"),
        row=2,
        col=1,
    )
    entry = result.config.entry_threshold
    exit_threshold = result.config.exit_threshold
    for value, color, dash in [
        (entry, "#b91c1c", "dash"),
        (-entry, "#047857", "dash"),
        (exit_threshold, "#92400e", "dot"),
        (-exit_threshold, "#92400e", "dot"),
        (0.0, "#334155", "solid"),
    ]:
        fig.add_hline(y=value, line_color=color, line_dash=dash, row=2, col=1)
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Spread and close-based z-score",
        legend_orientation="h",
        margin=dict(l=20, r=20, t=55, b=20),
        height=560,
    )
    fig.update_yaxes(title_text="Spread", row=1, col=1)
    fig.update_yaxes(title_text="Z-score", row=2, col=1)
    return fig


def equity_drawdown_chart(result: PairResearchResult) -> go.Figure:
    equity = result.backtest.equity
    drawdown = equity / equity.cummax() - 1.0
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=equity.index, y=equity, mode="lines", name="Equity"), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#b91c1c"),
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Cumulative returns and drawdown",
        legend_orientation="h",
        margin=dict(l=20, r=20, t=55, b=20),
        height=560,
    )
    fig.update_yaxes(title_text="Growth of $1", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    return fig


def rolling_sharpe_chart(result: PairResearchResult, window: int = 63) -> go.Figure:
    sharpe = rolling_sharpe(result.backtest.daily_returns, window=window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sharpe.index, y=sharpe, mode="lines", name="Rolling Sharpe"))
    fig.add_hline(y=0.0, line_dash="dot", line_color="#334155")
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=f"{window}-day rolling Sharpe",
        yaxis_title="Annualized Sharpe",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def hedge_ratio_chart(parameters: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=parameters.index,
            y=parameters["hedge_ratio"],
            mode="lines",
            name="Hedge ratio",
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Walk-forward hedge ratio",
        yaxis_title="Beta",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig
