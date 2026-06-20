import numpy as np
import pandas as pd

from stat_arb.pipeline import ResearchConfig, run_research


def test_pipeline_writes_expected_artifacts(tmp_path) -> None:
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2018-01-01", periods=360)
    silver_log = 3.0 + np.cumsum(rng.normal(0, 0.01, len(index)))
    residual = np.zeros(len(index))
    for i in range(1, len(index)):
        residual[i] = 0.8 * residual[i - 1] + rng.normal(0, 0.005)
    prices = pd.DataFrame(
        {"GLD": np.exp(1.2 + 0.8 * silver_log + residual), "SLV": np.exp(silver_log)},
        index=index,
    )
    config = ResearchConfig(
        formation_end=str(index[199].date()),
        zscore_window=20,
        refit_frequency=20,
    )
    study = run_research(prices, config, tmp_path)

    assert "sharpe" in study["metrics"]
    assert (tmp_path / "tables/performance_metrics.csv").exists()
    assert (tmp_path / "tables/trade_log.csv").exists()
    assert (tmp_path / "figures/equity_and_drawdown.png").exists()
