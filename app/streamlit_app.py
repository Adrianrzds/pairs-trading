from __future__ import annotations

import logging
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stat_arb.config import CASE_STUDIES, CaseStudy, load_app_config  # noqa: E402
from stat_arb.data import download_price_data, validate_price_data  # noqa: E402
from stat_arb.plotly_charts import (  # noqa: E402
    equity_drawdown_chart,
    hedge_ratio_chart,
    price_chart,
    rolling_sharpe_chart,
    spread_zscore_chart,
)
from stat_arb.research import PairResearchConfig, build_pair_research  # noqa: E402
from stat_arb.storage import PriceCache  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _parse_universe(raw: str) -> list[str]:
    return sorted(
        {ticker.strip().upper() for ticker in raw.replace("\n", ",").split(",") if ticker.strip()}
    )


@st.cache_data(show_spinner=False)
def _download_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    return download_price_data(list(tickers), start=start, end=end)


def _format_pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2%}"


def _format_float(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2f}"


def _display_series(values: dict[str, object] | pd.Series) -> pd.DataFrame:
    series = pd.Series(values, name="value")
    return series.map(str).to_frame()


def _case_by_name(name: str) -> CaseStudy | None:
    for case in CASE_STUDIES:
        if case.name == name:
            return case
    return None


def _cointegration_label(p_value: float) -> str:
    if p_value <= 0.05:
        return "Supported"
    if p_value <= 0.10:
        return "Borderline"
    return "Not Supported"


def _drawdown_label(max_drawdown: float) -> str:
    depth = abs(max_drawdown)
    if depth < 0.10:
        return "Low"
    if depth < 0.25:
        return "Medium"
    return "High"


def _signal_label(signal: object) -> str:
    return str(signal).replace("_", " ").title()


def _render_research_summary(
    latest: dict[str, object],
    metrics: dict[str, float],
    coint: dict[str, float],
    case_study: CaseStudy | None,
) -> None:
    signal = _signal_label(latest["signal"])
    cointegration = _cointegration_label(float(coint["p_value"]))
    mode_status = "Supported" if cointegration == "Supported" else "Diagnostic"
    drawdown = _drawdown_label(float(metrics["max_drawdown"]))
    sharpe = float(metrics["sharpe"])

    if mode_status == "Diagnostic":
        signal_text = (
            f"Raw cointegration-spread signal is {signal}, but this mode lacks strong "
            "full-sample cointegration support."
        )
    elif signal == "Flat":
        signal_text = "No active spread position at the latest close."
    else:
        signal_text = f"{signal} setup at the latest close; modeled execution is next session."

    if cointegration == "Supported":
        coint_text = "Engle-Granger rejects no cointegration at the 5% level."
    elif cointegration == "Borderline":
        coint_text = "Engle-Granger is borderline; treat the relation as fragile."
    else:
        coint_text = "Engle-Granger does not reject no cointegration at the 10% level."

    if pd.isna(sharpe):
        performance_text = "Backtest Sharpe is unavailable for this run."
    elif sharpe > 1.0:
        performance_text = "Backtest Sharpe is positive after configured costs."
    elif sharpe > 0.0:
        performance_text = "Backtest Sharpe is modest after configured costs."
    else:
        performance_text = "Backtest Sharpe is negative after configured costs."

    st.subheader("Research Readout")
    if case_study is not None:
        st.info(f"Case study: {case_study.theme}. {case_study.thesis}")
    columns = st.columns(5)
    columns[0].metric("Signal", signal)
    columns[1].metric("Cointegration Mode", mode_status)
    columns[2].metric("Cointegration", cointegration)
    columns[3].metric("Drawdown Risk", drawdown)
    columns[4].metric("Latest Close", pd.Timestamp(latest["as_of"]).date().isoformat())
    st.caption(f"{signal_text} {coint_text} {performance_text}")
    st.caption(
        "This readout evaluates the cointegration-spread approach only; distance trading, "
        "rolling/regime cointegration, or fundamental relative-value methods are separate "
        "strategies."
    )


def _cointegration_table(coint: dict[str, float]) -> pd.DataFrame:
    p_value = float(coint["p_value"])
    t_stat = float(coint["t_stat"])
    rows = [
        ("Decision", _cointegration_label(p_value)),
        ("p-value", f"{p_value:.3f}"),
        ("Test statistic", f"{t_stat:.3f}"),
        ("5% critical value", f"{float(coint['critical_value_5pct']):.3f}"),
        (
            "Interpretation",
            (
                "Stationary spread supported"
                if p_value <= 0.05
                else "No strong cointegration evidence"
            ),
        ),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"]).set_index("Item")


def _active_backtest_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    active = (
        ledger["position"].ne(0)
        | ledger["turnover"].ne(0)
        | ledger["gross_return"].ne(0)
        | ledger["transaction_cost"].ne(0)
        | ledger["net_return"].ne(0)
    )
    return ledger.loc[active]


def _render_workflow_header(active_step: int) -> None:
    labels = ["1. Data", "2. Pair", "3. Review"]
    columns = st.columns(3)
    for index, label in enumerate(labels, start=1):
        state = (
            "complete" if index < active_step else "active" if index == active_step else "pending"
        )
        columns[index - 1].metric(label, state.title())


def _render_empty_data_state(
    universe: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache: PriceCache,
) -> None:
    _render_workflow_header(active_step=1)
    st.subheader("No Cached ETF Data")
    st.write("Download the configured universe to build the local research cache.")
    details = pd.DataFrame(
        {
            "value": [
                f"{len(universe)} tickers",
                f"{pd.Timestamp(start).date()} to {pd.Timestamp(end).date()}",
                str(cache.parquet_dir),
                str(cache.duckdb_path),
                "Yahoo Finance via yfinance; no API key",
            ]
        },
        index=["Universe", "Date range", "Parquet cache", "DuckDB catalog", "Data source"],
    )
    st.dataframe(details, width="stretch")
    st.info("Use the sidebar button to download data before selecting a pair.")


def main() -> None:
    load_dotenv(ROOT / ".env")
    app_config = load_app_config()
    cache = PriceCache(data_dir=app_config.data_dir, duckdb_path=app_config.duckdb_path)

    st.set_page_config(
        page_title="Pairs Research Dashboard",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Pairs Trading Research Dashboard")

    with st.sidebar:
        st.header("1. Load Data")
        universe_raw = st.text_area(
            "ETF tickers",
            value=",".join(app_config.default_universe),
            height=110,
        )
        universe = _parse_universe(universe_raw)
        start = st.date_input("Start", value=pd.Timestamp("2015-01-01"))
        end = st.date_input("End", value=pd.Timestamp.today().normalize())
        if st.button("Download / refresh cache", type="primary", width="stretch"):
            try:
                downloaded = _download_prices(
                    tuple(universe),
                    pd.Timestamp(start).date().isoformat(),
                    (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
                )
                validated = validate_price_data(
                    downloaded, expected_tickers=universe, min_observations=100
                )
                cache.write_prices(validated)
                st.success(f"Cached {len(validated):,} rows for {len(validated.columns)} tickers.")
            except Exception as exc:
                LOGGER.exception("Download failed")
                st.error(f"Download failed: {exc}")

        cached = cache.available_tickers()
        usable_tickers = [ticker for ticker in universe if ticker in cached]
        if len(usable_tickers) < 2:
            st.info(f"{len(cached)} cached ticker(s) available.")
        else:
            st.success(f"{len(usable_tickers)} cached ticker(s) ready.")

            st.header("2. Select Pair")
            case_names = ["Custom pair"] + [case.name for case in CASE_STUDIES]
            selected_case_name = st.selectbox("Case study", case_names, index=0)
            selected_case = _case_by_name(selected_case_name)
            if selected_case is not None:
                st.caption(f"{selected_case.theme}: {selected_case.thesis}")
                missing = [
                    ticker
                    for ticker in [selected_case.y_asset, selected_case.x_asset]
                    if ticker not in usable_tickers
                ]
                if missing:
                    st.warning(
                        "Download / refresh the expanded universe to use this case: "
                        + ", ".join(missing)
                    )

            default_y = (
                selected_case.y_asset
                if selected_case and selected_case.y_asset in usable_tickers
                else usable_tickers[0]
            )
            y_asset = st.selectbox(
                "Dependent asset",
                usable_tickers,
                index=usable_tickers.index(default_y),
            )
            x_options = [ticker for ticker in usable_tickers if ticker != y_asset]
            default_x = (
                selected_case.x_asset
                if selected_case and selected_case.x_asset in x_options
                else x_options[0]
            )
            x_asset = st.selectbox("Hedge asset", x_options, index=x_options.index(default_x))

            hedge_method = "expanding"
            min_observations = 120
            lookback = 252
            zscore_window = 60
            entry_threshold = 2.0
            exit_threshold = 0.5
            transaction_cost_bps = 5.0
            position_size = 1.0

            with st.expander("Advanced Settings"):
                hedge_method = st.segmented_control(
                    "Hedge ratio",
                    options=["expanding", "rolling"],
                    default=hedge_method,
                )
                min_observations = st.number_input(
                    "Minimum OLS observations", 20, 1_000, min_observations, step=10
                )
                lookback = st.number_input("Rolling lookback", 40, 1_500, lookback, step=21)
                zscore_window = st.number_input("Z-score window", 10, 500, zscore_window, step=5)
                entry_threshold = st.slider("Entry threshold", 0.5, 4.0, entry_threshold, 0.1)
                exit_threshold = st.slider("Exit threshold", 0.0, 2.0, exit_threshold, 0.1)
                transaction_cost_bps = st.number_input(
                    "Transaction cost, bps one-way", 0.0, 100.0, transaction_cost_bps, 0.5
                )
                position_size = st.number_input("Gross exposure", 0.1, 5.0, position_size, 0.1)

    if len(usable_tickers) < 2:
        _render_empty_data_state(universe, pd.Timestamp(start), pd.Timestamp(end), cache)
        st.stop()

    prices = cache.load_prices([y_asset, x_asset], start=start, end=end)
    config = PairResearchConfig(
        y_asset=y_asset,
        x_asset=x_asset,
        hedge_method=hedge_method,
        lookback=int(lookback),
        min_observations=int(min_observations),
        zscore_window=int(zscore_window),
        entry_threshold=float(entry_threshold),
        exit_threshold=float(exit_threshold),
        transaction_cost_bps=float(transaction_cost_bps),
        position_size=float(position_size),
    )

    try:
        result = build_pair_research(prices, config)
    except Exception as exc:
        LOGGER.exception("Research run failed")
        st.error(f"Research run failed: {exc}")
        st.stop()

    latest = result.current_signal
    metrics = result.backtest.metrics
    coint = result.cointegration
    active_case = (
        selected_case
        if selected_case and [y_asset, x_asset] == [selected_case.y_asset, selected_case.x_asset]
        else None
    )

    _render_workflow_header(active_step=3)
    _render_research_summary(latest, metrics, coint, active_case)

    top = st.columns(6)
    top[0].metric("Latest z-score", _format_float(float(latest["zscore"])))
    top[1].metric("Cointegration p", f"{coint['p_value']:.3f}")
    top[2].metric("Cumulative return", _format_pct(metrics["cumulative_return"]))
    top[3].metric("Sharpe", _format_float(metrics["sharpe"]))
    top[4].metric("Max drawdown", _format_pct(metrics["max_drawdown"]))
    top[5].metric("Trades", f"{int(metrics['trade_count']):,}")

    st.caption(
        f"As of {pd.Timestamp(latest['as_of']).date().isoformat()} close. "
        "Signals execute with a one-session delay in the backtest."
    )

    overview, spread, backtest, trades, data = st.tabs(
        ["Overview", "Spread & Signals", "Backtest", "Trades", "Data"]
    )

    with overview:
        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(price_chart(result), width="stretch")
        with right:
            st.subheader("Cointegration")
            st.dataframe(_cointegration_table(coint), width="stretch")
            st.caption(
                "This diagnostic describes evidence for a stationary spread; it is not a "
                "guarantee of profitability."
            )
            st.subheader("Current signal")
            st.dataframe(_display_series(latest), width="stretch")

    with spread:
        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(spread_zscore_chart(result), width="stretch")
        with right:
            st.plotly_chart(hedge_ratio_chart(result.parameters), width="stretch")
            signal_table = pd.concat(
                [result.spread, result.zscore, result.signals["position"]],
                axis=1,
            ).tail(20)
            st.dataframe(signal_table, width="stretch")

    with backtest:
        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(equity_drawdown_chart(result), width="stretch")
            st.plotly_chart(rolling_sharpe_chart(result), width="stretch")
        with right:
            metric_table = pd.Series(metrics, name="value").to_frame()
            st.dataframe(metric_table, width="stretch")
            st.subheader("Active Backtest Rows")
            active_ledger = _active_backtest_rows(result.backtest.ledger)
            if active_ledger.empty:
                st.info("No active positions or executed trades for this configuration.")
            else:
                st.dataframe(active_ledger.tail(25), width="stretch")
            st.caption(
                "Flat no-position days are hidden here because their weights and returns are zero."
            )

    with trades:
        st.dataframe(result.backtest.trades, width="stretch")

    with data:
        st.dataframe(result.prices.tail(200), width="stretch")
        st.download_button(
            "Download full ledger CSV",
            result.backtest.ledger.to_csv().encode("utf-8"),
            file_name=f"{y_asset}_{x_asset}_ledger.csv",
            mime="text/csv",
        )
        active_ledger = _active_backtest_rows(result.backtest.ledger)
        st.download_button(
            "Download active ledger CSV",
            active_ledger.to_csv().encode("utf-8"),
            file_name=f"{y_asset}_{x_asset}_active_ledger.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
