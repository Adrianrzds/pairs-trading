from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def engle_granger_test(
    y: pd.Series,
    x: pd.Series,
    trend: str = "c",
) -> dict[str, float]:
    """Run the Engle-Granger two-step cointegration test."""
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(aligned) < 20:
        raise ValueError("At least 20 aligned observations are required for cointegration testing.")
    score, pvalue, critical_values = coint(aligned["y"], aligned["x"], trend=trend)
    return {
        "t_stat": float(score),
        "p_value": float(pvalue),
        "critical_value_1pct": float(critical_values[0]),
        "critical_value_5pct": float(critical_values[1]),
        "critical_value_10pct": float(critical_values[2]),
    }


def estimate_ols_parameters(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Estimate intercept and slope in ``y = intercept + beta * x + error``."""
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(aligned) < 2:
        raise ValueError("At least two aligned observations are required for OLS.")
    model = sm.OLS(aligned["y"], sm.add_constant(aligned["x"])).fit()
    return float(model.params["const"]), float(model.params["x"])


def estimate_hedge_ratio_ols(y: pd.Series, x: pd.Series) -> float:
    """Estimate OLS hedge ratio y = beta * x + intercept."""
    _, beta = estimate_ols_parameters(y, x)
    return beta


def construct_spread(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
    intercept: float = 0.0,
) -> pd.Series:
    """Construct the residual spread from a hedge ratio."""
    return y - intercept - hedge_ratio * x


def walk_forward_ols(
    y: pd.Series,
    x: pd.Series,
    trading_start: str | pd.Timestamp,
    refit_frequency: int = 21,
) -> pd.DataFrame:
    """Expanding-window OLS parameters, each fit using observations before its date."""
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    trading_start = pd.Timestamp(trading_start)
    trading_dates = aligned.index[aligned.index >= trading_start]
    if not len(trading_dates):
        raise ValueError("Trading start is outside the available sample.")
    if refit_frequency < 1:
        raise ValueError("Refit frequency must be positive.")

    parameters = pd.DataFrame(index=trading_dates, columns=["intercept", "hedge_ratio"], dtype=float)
    for number, date in enumerate(trading_dates):
        if number % refit_frequency == 0:
            history = aligned.loc[aligned.index < date]
            if len(history) < 20:
                raise ValueError("At least 20 pre-trading observations are required for walk-forward OLS.")
            intercept, beta = estimate_ols_parameters(history["y"], history["x"])
            parameters.loc[date] = [intercept, beta]
    return parameters.ffill()


def rolling_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    window: int,
) -> pd.Series:
    """Estimate a rolling hedge ratio using OLS on a moving window."""
    hedge_ratio = (
        x.rolling(window)
        .apply(lambda z: sm.OLS(y.loc[z.index], sm.add_constant(z), missing="drop").fit().params[1], raw=False)
    )
    return hedge_ratio


def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score based on a rolling mean and standard deviation."""
    if window < 2:
        raise ValueError("Z-score window must be at least two observations.")
    mu = series.rolling(window, min_periods=window).mean()
    sigma = series.rolling(window, min_periods=window).std(ddof=1).replace(0, np.nan)
    return (series - mu) / sigma
