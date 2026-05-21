from __future__ import annotations

import pandas as pd
import pytest

from nba_predict.download import NBAStatsDownloader, season_end_year_to_slug


def test_season_end_year_to_slug_formats_expected_value() -> None:
    assert season_end_year_to_slug(2023) == "2022-23"


def test_location_from_matchup_detects_home_and_away() -> None:
    assert NBAStatsDownloader._location_from_matchup("LAL @ BOS") == "A"
    assert NBAStatsDownloader._location_from_matchup("BOS vs. LAL") == "H"


def test_location_from_matchup_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="Cannot infer location"):
        NBAStatsDownloader._location_from_matchup("BOS-LAL")


def test_back_to_back_flags_follow_team_schedule(
    sample_api_game_logs: pd.DataFrame,
) -> None:
    downloader = NBAStatsDownloader()
    normalized = downloader._normalize_season(sample_api_game_logs, "2022-23")

    sharks_flags = normalized.loc[normalized["nameTeam"].eq("Sharks"), "isB2BSecond"]
    wolves_flags = normalized.loc[normalized["nameTeam"].eq("Wolves"), "isB2BSecond"]

    assert sharks_flags.tolist() == [False, True, True]
    assert wolves_flags.tolist() == [False, True, True]


def test_normalize_season_rejects_missing_columns() -> None:
    downloader = NBAStatsDownloader()
    with pytest.raises(ValueError, match="missing columns"):
        downloader._normalize_season(pd.DataFrame({"GAME_ID": ["1"]}), "2022-23")


def test_save_uses_download_and_writes_csv(tmp_path: pytest.TempPathFactory) -> None:
    output = tmp_path / "raw.csv"
    expected = pd.DataFrame({"slugSeason": ["2022-23"], "dateGame": ["2022-10-01"]})
    downloader = NBAStatsDownloader()
    downloader.download = lambda: expected  # type: ignore[method-assign]

    saved = downloader.save(output)

    assert saved.equals(expected)
    assert pd.read_csv(output).equals(expected)
