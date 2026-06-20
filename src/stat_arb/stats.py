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

    parameters = pd.DataFrame(
        index=trading_dates, columns=["intercept", "hedge_ratio"], dtype=float
    )
    for number, date in enumerate(trading_dates):
        if number % refit_frequency == 0:
            history = aligned.loc[aligned.index < date]
            if len(history) < 20:
                raise ValueError(
                    "At least 20 pre-trading observations are required for walk-forward OLS."
                )
            intercept, beta = estimate_ols_parameters(history["y"], history["x"])
            parameters.loc[date] = [intercept, beta]
    return parameters.ffill()


def rolling_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    window: int,
) -> pd.Series:
    """Estimate a rolling hedge ratio using OLS on a moving window."""
    hedge_ratio = x.rolling(window).apply(
        lambda z: sm.OLS(y.loc[z.index], sm.add_constant(z), missing="drop").fit().params[1],
        raw=False,
    )
    return hedge_ratio


def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score based on a rolling mean and standard deviation."""
    if window < 2:
        raise ValueError("Z-score window must be at least two observations.")
    mu = series.rolling(window, min_periods=window).mean()
    sigma = series.rolling(window, min_periods=window).std(ddof=1).replace(0, np.nan)
    return (series - mu) / sigma


def compute_walk_forward_spread_zscore(
    y: pd.Series,
    x: pd.Series,
    parameters: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Construct date-consistent walk-forward spreads and trailing z-scores.

    For signal date ``t``, every observation in the trailing window is transformed
    using the intercept and hedge ratio available at ``t``. This avoids comparing
    residuals produced by different model fits while retaining strictly historical
    estimation: callers must supply parameters fitted only through ``t - 1``.
    """
    if window < 2:
        raise ValueError("Z-score window must be at least two observations.")
    required_columns = {"intercept", "hedge_ratio"}
    if not required_columns.issubset(parameters.columns):
        raise ValueError(f"Parameters must contain columns {sorted(required_columns)}.")

    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).reindex(parameters.index)
    if aligned.isna().any().any() or parameters[list(required_columns)].isna().any().any():
        raise ValueError("Prices and walk-forward parameters must be complete and aligned.")

    spread = pd.Series(index=parameters.index, dtype=float, name="spread")
    zscore = pd.Series(index=parameters.index, dtype=float, name="zscore")
    for position, date in enumerate(parameters.index):
        intercept = float(parameters.at[date, "intercept"])
        beta = float(parameters.at[date, "hedge_ratio"])
        spread.at[date] = aligned.at[date, "y"] - intercept - beta * aligned.at[date, "x"]
        if position + 1 < window:
            continue
        trailing = aligned.iloc[position - window + 1 : position + 1]
        comparable_spread = trailing["y"] - intercept - beta * trailing["x"]
        sigma = comparable_spread.std(ddof=1)
        if sigma > 0 and np.isfinite(sigma):
            zscore.at[date] = (comparable_spread.iloc[-1] - comparable_spread.mean()) / sigma

    return pd.concat([spread, zscore], axis=1)
