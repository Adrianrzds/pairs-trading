# Pairs Trading Research Dashboard

A local, portfolio-quality Python dashboard for researching ETF pairs-trading ideas with
cointegration diagnostics, walk-forward hedge ratios, delayed-execution backtests, and a
DuckDB/Parquet data cache.

This is a research tool, not investment advice or evidence of deployable alpha.

## Features

- Streamlit dashboard for selecting ETF pairs and configuring model/backtest assumptions.
- Daily adjusted close downloads via `yfinance`.
- Local Parquet snapshots queried through DuckDB.
- Engle-Granger cointegration test on log prices.
- Curated case studies for ETF near-clones, sector pairs, credit pairs, and GLD/SLV regime behavior.
- Expanding or rolling OLS hedge ratios estimated only from prior observations.
- Spread and z-score construction with configurable trailing windows.
- Long-spread, short-spread, and flat signals with configurable entry/exit thresholds.
- Backtest with one-session signal execution delay, transaction costs, position sizing, turnover,
  trade list, hit rate, rolling Sharpe, cumulative returns, and drawdown.
- Deterministic pytest coverage for storage, no-lookahead parameter estimation, signal execution,
  and backtest mechanics.

## Repository Layout

```text
app/
  streamlit_app.py          Streamlit dashboard
src/stat_arb/
  config.py                 Environment-aware app defaults
  storage.py                DuckDB and Parquet price cache
  data.py                   yfinance download and price validation
  research.py               Pair research orchestration
  stats.py                  OLS, cointegration, spread, z-score utilities
  backtest.py               Delayed-execution pair backtest
  plotly_charts.py          Dashboard chart builders
tests/                      Unit and integration tests
data/                       Generated local cache, ignored by git
```

## Quickstart

Python 3.10 or later is required.

Clone the repo, create a virtual environment, install the app, then launch Streamlit:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
streamlit run app/streamlit_app.py
```

The dashboard opens locally. Use the sidebar to download or refresh the ETF universe, then select
two cached assets and run the research view.

Optional local settings live in `.env`. To customize paths or the default ETF universe:

```bash
cp sample.env .env
```

No API secrets are required for the default Yahoo Finance data path.

## Case Studies

The dashboard includes four curated asset pairs to show what cointegration does and does not mean:

- `SPY` / `IVV`: ETF near-clones tracking the S&P 500. This is the cleanest case where a stable
  long-run spread is economically plausible.
- `XLE` / `XOP`: same energy sector, but different portfolio construction and company exposures.
  Related assets do not automatically imply a stationary spread.
- `LQD` / `HYG`: corporate bond ETFs with shared rate and credit-cycle drivers but different
  credit quality. Risk-off periods can change the relationship sharply.
- `GLD` / `SLV`: related metals with different macro drivers. Gold can behave like a safe-haven
  asset while silver has more industrial demand exposure, so their relationship can decouple.

The current strategy mode is a cointegration-spread model. If Engle-Granger does not support a
stationary spread over the selected sample, the app treats that mode as diagnostic only. This does
not rule out other pairs-trading approaches such as distance trading, rolling/regime cointegration,
or fundamental relative value.

## Developer Setup

Install test and lint tools with the development extra:

```bash
python3 -m pip install -e '.[dev]'
pytest -q
ruff check .
```

`pyproject.toml` is the source of truth for package metadata, runtime dependencies, and developer
tooling. The editable install lets `app/streamlit_app.py` import `stat_arb` without setting
`PYTHONPATH`.

## Data Flow

1. `yfinance` downloads daily adjusted closes for the configured ETF universe.
2. The cache writes one Parquet file per ticker under `data/raw/`.
3. DuckDB exposes those Parquet files as a local `prices` view in `data/research.duckdb`.
4. The dashboard loads the selected pair into a wide adjusted-close table.
5. Prices are validated for alignment, missing values, duplicate dates, finite values, and
   positivity.
6. Log prices feed cointegration, hedge-ratio estimation, spread construction, signal generation,
   and backtesting.

Generated data, DuckDB files, virtualenvs, and `.env` files are ignored by git.

Yahoo Finance data is downloaded by the person running the app in their own local environment.
This repository does not bundle or redistribute market data.

## Look-Ahead Bias Controls

- Hedge parameters for date `t` are estimated using observations strictly before `t`.
- Rolling hedge ratios use only the trailing window that ends at `t - 1`.
- Expanding hedge ratios use all available history through `t - 1`.
- The z-score at close `t` uses the spread history available through close `t`.
- The signal observed at close `t` becomes an executed position one session later.
- Returns and transaction costs are both calculated from executed, delayed weights.
- The current signal shown in the dashboard is based only on the latest completed close and is
  labeled as a next-session execution candidate.
- If cointegration is not supported, the readout marks the cointegration-spread mode as diagnostic.

## Methodology

For a selected dependent asset `Y` and hedge asset `X`, the model estimates:

```text
log(Y_t) = intercept_t + beta_t * log(X_t) + residual_t
```

The spread is the residual. A long-spread signal means long `Y` and short `beta * X`; a
short-spread signal means short `Y` and long `beta * X`. Pair weights are normalized to the
configured gross exposure:

```text
w_Y = position_size * position / (1 + abs(beta))
w_X = -position_size * position * beta / (1 + abs(beta))
```

Transaction costs are charged on absolute changes in executed leg weights. Turnover is the sum of
those absolute changes across both legs.

## Limitations

- Adjusted closes are research proxies, not executable prices.
- The backtest omits bid/ask spread variation, slippage, market impact, borrow costs, financing,
  taxes, ETF creation/redemption effects, and intraday execution uncertainty.
- Yahoo Finance data can be revised and may differ from institutional data vendors.
- Engle-Granger is asymmetric and does not guarantee a stable tradable relationship.
- Testing many pairs and parameters creates multiple-testing risk.
- Rolling/expanding OLS adapts slowly to structural breaks.
