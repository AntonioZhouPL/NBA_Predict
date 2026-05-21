from __future__ import annotations

import pandas as pd
import pytest

from nba_predict.modeling import (
    EvaluationResult,
    LogisticGamePredictor,
    RollingSeasonEvaluator,
    SeasonSplitter,
)


def test_evaluation_result_to_dict_omits_predictions() -> None:
    result = EvaluationResult(
        accuracy=0.75,
        precision=0.8,
        recall=0.7,
        confusion_matrix=[[2, 1], [0, 3]],
        cumulative_errors=[1, 2],
        predictions=pd.DataFrame({"actual": [1], "predicted": [1]}),
    )

    assert result.to_dict() == {
        "accuracy": 0.75,
        "precision": 0.8,
        "recall": 0.7,
        "confusion_matrix": [[2, 1], [0, 3]],
        "cumulative_errors": [1, 2],
    }


def test_season_splitter_rejects_missing_season(
    sample_design_matrix: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="was not found"):
        SeasonSplitter(season="1999-00").split(sample_design_matrix)


def test_invalid_penalty_is_rejected() -> None:
    with pytest.raises(ValueError, match="penalty must be one of"):
        LogisticGamePredictor(penalty="elasticnet")


def test_predictor_fits_and_predicts(sample_design_matrix: pd.DataFrame) -> None:
    predictor = LogisticGamePredictor()

    predictor.fit(sample_design_matrix.loc[sample_design_matrix["slugSeason"].eq("2021-22")])
    probabilities = predictor.predict_proba(
        sample_design_matrix.loc[sample_design_matrix["slugSeason"].eq("2022-23")]
    )

    assert len(probabilities) == 4
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_evaluator_rejects_missing_season(sample_design_matrix: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="was not found"):
        RollingSeasonEvaluator(season="1999-00").evaluate(
            sample_design_matrix,
            predictor_factory=LogisticGamePredictor,
        )


def test_evaluator_returns_predictions_and_metrics(
    sample_design_matrix: pd.DataFrame,
) -> None:
    result = RollingSeasonEvaluator(season="2022-23").evaluate(
        sample_design_matrix,
        predictor_factory=LogisticGamePredictor,
    )

    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert len(result.cumulative_errors) == 2
    assert list(result.predictions.columns) == [
        "dateGame",
        "slugSeason",
        "actual",
        "predicted",
        "probability_home_win",
    ]
    assert set(result.predictions["slugSeason"]) == {"2022-23"}
