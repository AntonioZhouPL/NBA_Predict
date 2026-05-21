from __future__ import annotations

import pandas as pd
import pytest

from nba_predict.data import FEATURE_COLUMNS


@pytest.fixture
def sample_api_game_logs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["0001", "0001", "0002", "0002", "0003", "0003"],
            "GAME_DATE": [
                "2022-10-01",
                "2022-10-01",
                "2022-10-02",
                "2022-10-02",
                "2022-10-03",
                "2022-10-03",
            ],
            "TEAM_ID": [1, 2, 1, 2, 1, 2],
            "TEAM_NAME": ["Sharks", "Wolves", "Sharks", "Wolves", "Sharks", "Wolves"],
            "MATCHUP": [
                "Sharks vs. Wolves",
                "Wolves @ Sharks",
                "Sharks @ Wolves",
                "Wolves vs. Sharks",
                "Sharks vs. Wolves",
                "Wolves @ Sharks",
            ],
            "WL": ["W", "L", "L", "W", "W", "L"],
            "BLK": [5, 4, 4, 5, 6, 3],
            "TOV": [12, 14, 13, 11, 10, 12],
            "PF": [18, 20, 19, 17, 16, 18],
            "PTS": [100, 90, 95, 102, 110, 98],
            "STL": [8, 7, 6, 9, 10, 6],
            "REB": [45, 40, 42, 44, 47, 41],
            "OREB": [10, 9, 9, 10, 11, 8],
            "DREB": [35, 31, 33, 34, 36, 33],
            "FGA": [80, 78, 79, 81, 84, 80],
            "FGM": [40, 35, 37, 39, 43, 38],
            "FTM": [15, 14, 17, 16, 18, 15],
            "FTA": [20, 18, 21, 19, 22, 20],
        }
    )


@pytest.fixture
def sample_raw_gamelogs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slugSeason": ["2022-23"] * 6,
            "dateGame": [
                "2022-10-01",
                "2022-10-01",
                "2022-10-02",
                "2022-10-02",
                "2022-10-03",
                "2022-10-03",
            ],
            "idGame": ["0001", "0001", "0002", "0002", "0003", "0003"],
            "nameTeam": ["Sharks", "Wolves", "Sharks", "Wolves", "Sharks", "Wolves"],
            "isB2BSecond": [False, False, True, True, True, True],
            "locationGame": ["H", "A", "A", "H", "H", "A"],
            "slugMatchup": [
                "Sharks vs. Wolves",
                "Wolves @ Sharks",
                "Sharks @ Wolves",
                "Wolves vs. Sharks",
                "Sharks vs. Wolves",
                "Wolves @ Sharks",
            ],
            "outcomeGame": ["W", "L", "L", "W", "W", "L"],
            "blk": [5, 4, 4, 5, 6, 3],
            "tov": [12, 14, 13, 11, 10, 12],
            "pf": [18, 20, 19, 17, 16, 18],
            "pts": [100, 90, 95, 102, 110, 98],
            "stl": [8, 7, 6, 9, 10, 6],
            "treb": [45, 40, 42, 44, 47, 41],
            "oreb": [10, 9, 9, 10, 11, 8],
            "dreb": [35, 31, 33, 34, 36, 33],
            "fga": [80, 78, 79, 81, 84, 80],
            "fgm": [40, 35, 37, 39, 43, 38],
            "ftm": [15, 14, 17, 16, 18, 15],
            "fta": [20, 18, 21, 19, 22, 20],
        }
    )


@pytest.fixture
def sample_design_matrix() -> pd.DataFrame:
    rows = [
        ("2021-22", "2022-04-01", 1, [0, 5, 4, 2, -2, 8, -6, 4, 0.08, 0.8]),
        ("2021-22", "2022-04-02", 0, [1, -4, -3, -1, 3, -7, 5, -3, -0.06, 0.2]),
        ("2021-22", "2022-04-03", 1, [0, 6, 5, 3, -1, 10, -7, 5, 0.09, 0.7]),
        ("2021-22", "2022-04-04", 0, [1, -5, -4, -2, 2, -8, 6, -4, -0.05, 0.3]),
        ("2022-23", "2022-10-01", 1, [0, 5, 3, 2, -2, 9, -6, 4, 0.07, 0.8]),
        ("2022-23", "2022-10-01", 0, [1, -5, -3, -2, 2, -9, 6, -4, -0.07, 0.2]),
        ("2022-23", "2022-10-02", 1, [0, 4, 2, 1, -1, 7, -5, 3, 0.05, 0.7]),
        ("2022-23", "2022-10-02", 0, [1, -4, -2, -1, 1, -7, 5, -3, -0.05, 0.3]),
    ]
    data = []
    for season, date_game, outcome, feature_values in rows:
        row = {
            "slugSeason": season,
            "dateGame": date_game,
            "outcomeGame": outcome,
        }
        row.update(dict(zip(FEATURE_COLUMNS, feature_values, strict=True)))
        data.append(row)
    return pd.DataFrame(data)
