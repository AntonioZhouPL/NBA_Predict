from __future__ import annotations

import json
import sys
from pathlib import Path

from nba_predict import cli


def test_build_parser_supports_evaluate_design_matrix_argument() -> None:
    args = cli.build_parser().parse_args(
        [
            "evaluate",
            "--model",
            "ridge",
            "--season",
            "2022-23",
            "--design-matrix",
            "matrix.csv",
        ]
    )

    assert args.command == "evaluate"
    assert args.model == "ridge"
    assert args.design_matrix == Path("matrix.csv")


def test_main_dispatches_download_data(monkeypatch, capsys) -> None:
    class FakePipeline:
        def __init__(self, cv_folds: int) -> None:
            self.cv_folds = cv_folds

        def download_data(self, output_path, season_start: int, season_end: int):
            assert output_path == Path("raw.csv")
            assert season_start == 2013
            assert season_end == 2014
            return [1, 2, 3]

    monkeypatch.setattr(cli, "NBAPredictionPipeline", FakePipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nba-predict",
            "download-data",
            "--output",
            "raw.csv",
            "--season-start",
            "2013",
            "--season-end",
            "2014",
        ],
    )

    cli.main()

    assert "Saved raw game logs with 3 rows." in capsys.readouterr().out


def test_main_dispatches_evaluate(monkeypatch, capsys) -> None:
    class FakePipeline:
        def __init__(self, cv_folds: int) -> None:
            assert cv_folds == 5

        def evaluate_model(self, model_name: str, season: str, design_matrix_path):
            assert model_name == "lasso"
            assert season == "2022-23"
            assert design_matrix_path == Path("matrix.csv")
            return {"model": model_name, "season": season}

    monkeypatch.setattr(cli, "NBAPredictionPipeline", FakePipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nba-predict",
            "evaluate",
            "--model",
            "lasso",
            "--season",
            "2022-23",
            "--design-matrix",
            "matrix.csv",
            "--cv-folds",
            "5",
        ],
    )

    cli.main()

    assert json.loads(capsys.readouterr().out) == {
        "model": "lasso",
        "season": "2022-23",
    }


def test_main_dispatches_run_baseline(monkeypatch, capsys) -> None:
    class FakePipeline:
        def __init__(self, cv_folds: int) -> None:
            assert cv_folds == 4

        def run_baseline(self, season: str, design_matrix_path):
            assert season == "2022-23"
            assert design_matrix_path == Path("snapshot.csv")
            return [{"model": "logistic"}]

    monkeypatch.setattr(cli, "NBAPredictionPipeline", FakePipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nba-predict",
            "run-baseline",
            "--season",
            "2022-23",
            "--design-matrix",
            "snapshot.csv",
            "--cv-folds",
            "4",
        ],
    )

    cli.main()

    assert json.loads(capsys.readouterr().out) == [{"model": "logistic"}]
