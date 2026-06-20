from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_price_series(prices: pd.DataFrame, path: str | None = None) -> None:
    normalized = prices / prices.iloc[0]
    ax = normalized.plot(title="GLD and SLV: Growth of $1", xlabel="Date", ylabel="Normalized price")
    ax.figure.tight_layout()
    if path:
        ax.figure.savefig(path)
    plt.close(ax.figure)


def plot_spread_and_zscore(
    spread: pd.Series,
    zscore: pd.Series,
    path: str | None = None,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(spread.index, spread, label="Spread")
    axs[0].axhline(0, color="black", lw=0.8)
    axs[0].set_title("Residual Spread")
    axs[0].legend()

    axs[1].plot(zscore.index, zscore, label="Rolling z-score")
    axs[1].axhline(0, color="black", lw=0.8)
    axs[1].axhline(entry_threshold, color="red", ls="--", label="Entry")
    axs[1].axhline(-entry_threshold, color="green", ls="--", label="Entry")
    axs[1].axhline(exit_threshold, color="orange", ls=":", label="Exit")
    axs[1].axhline(-exit_threshold, color="orange", ls=":")
    axs[1].set_title("Rolling z-score")
    axs[1].legend()

    fig.tight_layout()
    if path:
        fig.savefig(path)
    plt.close(fig)


def plot_equity_curve(equity: pd.Series, path: str | None = None) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[2, 1])
    equity.plot(ax=axes[0], title="Out-of-Sample Strategy Equity", ylabel="Growth of $1")
    drawdown = equity / equity.cummax() - 1.0
    axes[1].fill_between(drawdown.index, drawdown.to_numpy(), 0, color="firebrick", alpha=0.5)
    axes[1].set(title="Drawdown", xlabel="Date", ylabel="Drawdown")
    axes[1].set_ylim(min(-0.01, float(np.nanmin(drawdown)) * 1.1), 0.01)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    plt.close(fig)
