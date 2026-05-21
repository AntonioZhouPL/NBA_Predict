from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_predict.config import ProjectPaths
from nba_predict.pipeline import NBAPredictionPipeline


def test_download_data_uses_downloader(monkeypatch) -> None:
    called = {}
    expected = pd.DataFrame({"slugSeason": ["2022-23"]})

    class FakeDownloader:
        def __init__(self, season_start: int, season_end: int) -> None:
            called["season_start"] = season_start
            called["season_end"] = season_end

        def save(self, output_path: Path) -> pd.DataFrame:
            called["output_path"] = output_path
            return expected

    monkeypatch.setattr("nba_predict.pipeline.NBAStatsDownloader", FakeDownloader)
    pipeline = NBAPredictionPipeline()
    result = pipeline.download_data(
        output_path=Path("custom.csv"),
        season_start=2015,
        season_end=2016,
    )

    assert result.equals(expected)
    assert called == {
        "season_start": 2015,
        "season_end": 2016,
        "output_path": Path("custom.csv"),
    }


def test_prepare_data_uses_preprocessor(monkeypatch, tmp_path) -> None:
    called = {}
    expected = pd.DataFrame({"slugSeason": ["2022-23"]})

    class FakePreprocessor:
        def save_design_matrix(self, raw_path: Path, output_path: Path) -> pd.DataFrame:
            called["raw_path"] = raw_path
            called["output_path"] = output_path
            return expected

    monkeypatch.setattr("nba_predict.pipeline.NBADataPreprocessor", FakePreprocessor)
    pipeline = NBAPredictionPipeline(
        paths=ProjectPaths(
            raw_data=tmp_path / "raw.csv",
            design_matrix=tmp_path / "processed" / "design_matrix.csv",
            metrics_dir=tmp_path / "metrics",
            predictions_dir=tmp_path / "predictions",
            figures_dir=tmp_path / "figures",
            models_dir=tmp_path / "models",
        )
    )

    result = pipeline.prepare_data()

    assert result.equals(expected)
    assert called["raw_path"] == tmp_path / "raw.csv"
    assert called["output_path"] == tmp_path / "processed" / "design_matrix.csv"


def test_evaluate_model_writes_metrics_and_predictions(
    sample_design_matrix: pd.DataFrame,
    tmp_path,
) -> None:
    design_matrix_path = tmp_path / "design_matrix.csv"
    sample_design_matrix.to_csv(design_matrix_path, index=False)
    paths = ProjectPaths(
        raw_data=tmp_path / "raw.csv",
        design_matrix=design_matrix_path,
        metrics_dir=tmp_path / "metrics",
        predictions_dir=tmp_path / "predictions",
        figures_dir=tmp_path / "figures",
        models_dir=tmp_path / "models",
    )
    pipeline = NBAPredictionPipeline(paths=paths)

    result = pipeline.evaluate_model("logistic", season="2022-23")

    assert result["model"] == "logistic"
    assert Path(result["metrics_path"]).exists()
    assert Path(result["predictions_path"]).exists()


def test_run_baseline_evaluates_all_models(monkeypatch) -> None:
    evaluated = []
    pipeline = NBAPredictionPipeline()

    def fake_evaluate_model(model_name: str, season: str, design_matrix_path=None):
        evaluated.append((model_name, season, design_matrix_path))
        return {"model": model_name}

    monkeypatch.setattr(pipeline, "evaluate_model", fake_evaluate_model)

    result = pipeline.run_baseline(
        season="2022-23",
        design_matrix_path=Path("snapshot.csv"),
    )

    assert result == [
        {"model": "logistic"},
        {"model": "inference-logistic"},
        {"model": "ridge"},
        {"model": "lasso"},
    ]
    assert evaluated == [
        ("logistic", "2022-23", Path("snapshot.csv")),
        ("inference-logistic", "2022-23", Path("snapshot.csv")),
        ("ridge", "2022-23", Path("snapshot.csv")),
        ("lasso", "2022-23", Path("snapshot.csv")),
    ]
