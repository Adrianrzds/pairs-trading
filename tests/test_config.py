from stat_arb.config import CASE_STUDIES, DEFAULT_UNIVERSE


def test_case_study_assets_are_in_default_universe() -> None:
    universe = set(DEFAULT_UNIVERSE)
    for case in CASE_STUDIES:
        assert case.y_asset in universe
        assert case.x_asset in universe
