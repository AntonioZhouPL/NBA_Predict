from __future__ import annotations

import pandas as pd
import pytest

from nba_predict.data import NBADataPreprocessor


def test_load_raw_rejects_missing_file(tmp_path: pytest.TempPathFactory) -> None:
    with pytest.raises(FileNotFoundError, match="Raw data file not found"):
        NBADataPreprocessor().load_raw(tmp_path / "missing.csv")


def test_build_team_summary_validates_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        NBADataPreprocessor().build_team_summary(
            pd.DataFrame({"slugSeason": ["2022-23"]})
        )


def test_build_team_summary_aggregates_and_converts_flags(
    sample_raw_gamelogs: pd.DataFrame,
) -> None:
    summary = NBADataPreprocessor().build_team_summary(sample_raw_gamelogs)

    assert len(summary) == 6
    assert set(summary["outcomeGame"].unique()) == {0, 1}
    assert set(summary["isB2BSecond"].unique()) == {0, 1}
    assert summary["poss"].notna().all()
    assert summary["plu_min"].notna().all()


def test_build_design_matrix_creates_home_minus_away_features(
    sample_raw_gamelogs: pd.DataFrame,
) -> None:
    preprocessor = NBADataPreprocessor(rolling_window=2, drop_initial_rows=0)

    design_matrix = preprocessor.build_design_matrix(sample_raw_gamelogs)

    assert list(design_matrix.columns) == [
        "slugSeason",
        "dateGame",
        "outcomeGame",
        *preprocessor.feature_columns,
    ]
    assert len(design_matrix) == 2
    assert design_matrix["dateGame"].tolist() == ["2022-10-02", "2022-10-03"]
    assert design_matrix["outcomeGame"].tolist() == [1, 1]
    assert design_matrix["avg_win_perc"].tolist() == pytest.approx([0.0, 0.0])
    assert design_matrix[preprocessor.feature_columns].notna().all().all()


def test_save_design_matrix_writes_expected_output(
    sample_raw_gamelogs: pd.DataFrame,
    tmp_path: pytest.TempPathFactory,
) -> None:
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "design_matrix.csv"
    sample_raw_gamelogs.to_csv(raw_path, index=False)

    saved = NBADataPreprocessor(
        rolling_window=2,
        drop_initial_rows=0,
    ).save_design_matrix(raw_path, output_path)

    assert output_path.exists()
    assert pd.read_csv(output_path).shape == saved.shape
