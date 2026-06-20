# Statistical Arbitrage: Gold/Silver Pairs Trading

An end-to-end, reproducible research project testing whether the adjusted prices of the
SPDR Gold Shares ETF (`GLD`) and iShares Silver Trust (`SLV`) have a sufficiently stable
cointegrating relationship to support an out-of-sample mean-reversion strategy after costs.

This is a research baseline, not a claim of deployable alpha or investment advice.

## Research question

Gold and silver returns can be highly correlated without their price levels sharing a stable
long-run equilibrium. The project therefore reports both quantities but does not confuse them:

- Pearson correlation measures contemporaneous linear co-movement in daily **returns**.
- Engle–Granger tests whether a linear combination of the two **log-price levels** is stationary.

The pre-specified null hypothesis for Engle–Granger is no cointegration. The formation-period
p-value is the relevant research result; the trading-period test is clearly labeled as an ex-post
stability diagnostic and is never used to admit, reject, or tune trades.

## Methodology

1. Download daily adjusted prices from Yahoo Finance and validate alignment, missingness,
   duplicates, finiteness, and positivity.
2. Split the sample by date: 2010–2017 is formation and 2018–2025 is trading by default.
3. On formation data only, calculate return correlation, run Engle–Granger, and estimate
   `log(GLD) = intercept + beta * log(SLV) + residual` with `statsmodels` OLS.
4. During trading, refit that OLS monthly (every 21 observations) on an expanding window ending
   on the previous observation. No current or future price enters that day's parameter estimate.
5. For each signal date, apply that date's lagged OLS coefficients consistently to all observations
   in its trailing 60-day spread window, then calculate the z-score. Enter long spread at
   `z <= -2`, short spread at `z >= 2`, and exit inside `+/-0.5`. These fixed baseline parameters
   are not selected on trading-period results.
6. A signal observed at close *t* determines weights held over *t* to *t+1*. Pair weights are
   normalized to one dollar of gross exposure: `w_GLD = position/(1+|beta|)` and
   `w_SLV = -position*beta/(1+|beta|)`.
7. Charge 5 bps one-way on the absolute change in both leg weights. The model reports CAGR,
   Sharpe, Sortino, maximum drawdown, closed-trade hit rate, gross notional turnover, and entries.

Walk-forward refitting is used for hedge parameters. The z-score uses trailing observations only.
The first 60 trading observations are consequently a warm-up period with no position.

## Repository layout

```text
src/stat_arb/       importable package: data, statistics, signals, backtest, plots, pipeline
scripts/            thin executable wrapper
notebooks/          reserved for exploration; production logic stays in the package
tests/              deterministic unit and integration tests
data/               generated downloads, ignored by git
outputs/tables/     generated metrics, diagnostics, ledger, and trade log
outputs/figures/    generated prices, spread/z-score, equity, and drawdown plots
```

## Reproduce

Python 3.10 or later is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
pytest -q
stat-arb-backtest
```

The run downloads data to `data/gld_slv_adjusted_close.csv` and writes all research artifacts to
`outputs/`. Both directories retain tracked placeholders while generated contents remain ignored.
To rerun from an existing snapshot without a network call:

```bash
stat-arb-backtest --input data/gld_slv_adjusted_close.csv
```

The CLI also accepts `--start`, `--end` (exclusive), `--formation-end`, `--data-path`, and
`--output-dir`. Exact model assumptions are saved in `outputs/config.json`.

## Baseline result

The reproducible run through 2025-12-31 does **not** support the proposed strategy. The formation
period Engle–Granger test fails to reject no cointegration (p-value 0.216), and the ex-post trading
period diagnostic also fails to reject it (p-value 0.356). After 5 bps one-way costs, the fixed
walk-forward strategy produces a -4.57% CAGR, -0.81 Sharpe ratio, and -33.53% maximum drawdown over
2018–2025. It opens 33 trades and wins 65.62% of closed trades, illustrating why hit rate alone is
not evidence of profitability. These values are generated from Yahoo Finance data and may change
slightly if that vendor revises history; the complete run is persisted under `outputs/`.

The appropriate conclusion is not that gold/silver pairs trading can never work. It is that this
transparent GLD/SLV specification provides no evidence of a stable, cost-surviving cointegrating
relationship in the tested sample.

## Interpreting results

The core decision is not whether the backtest has a positive Sharpe in isolation. Inspect:

- `cointegration_tests.csv`: does formation reject no cointegration, and does the relationship
  remain plausible in the held-out period?
- `walk_forward_parameters.csv`: is the hedge ratio economically and statistically stable?
- `spread_and_signals.csv`: are entries genuine deviations or regime shifts?
- `performance_metrics.csv` and `trade_log.csv`: does performance survive two-leg costs, and is it
  distributed across enough independent trades?
- `equity_and_drawdown.png`: is return generation steady or concentrated in one episode?

If formation data does not reject the null, or out-of-sample diagnostics deteriorate materially,
the honest conclusion is that this particular fixed specification does not establish a tradable
cointegrating relationship.

## Assumptions and limitations

- Adjusted ETF closes are research proxies, not executable quotes. Close-to-close simulation omits
  bid/ask dynamics, slippage variation, market impact, borrow availability/fees, financing, taxes,
  and execution latency.
- A constant 5 bps per one-way dollar traded is transparent but simplified.
- GLD and SLV have different economic exposures, fund structures, and inception histories. Their
  relationship can shift with monetary regimes, industrial silver demand, and ETF-specific flows.
- Engle–Granger is asymmetric and tests one linear relation. Structural breaks, multiple-testing
  risk, and residual autocorrelation need deeper analysis before any live use.
- Expanding estimation adapts slowly after a regime break. A rolling formation window is a sensible
  robustness check, but its length should be selected without looking at the final holdout.
- CAGR and annualized ratios use 252 observations per year and a zero risk-free rate. Hit rate uses
  closed trades; turnover is total gross notional traded over the test, not annualized turnover.

## Suggested next research steps

Freeze the current trading interval as a holdout before adding robustness checks: subperiod and
structural-break tests, alternative hedge estimators, borrow/financing costs, delayed execution,
cost stress tests, and a nested train/validation/test scheme for any parameter comparison. Avoid
turning the final holdout into an optimization set.
