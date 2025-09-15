# nfl_model/modeling.py
from __future__ import annotations
import math
from typing import Dict, Iterable, Tuple
import pandas as pd
import nfl_data_py as nfl

# Normalize legacy codes to modern ones
TEAM_FIX = {
    "LA": "LAR",   # old Rams code
    "STL": "LAR",
    "SD": "LAC",   # old Chargers
    "OAK": "LV",   # old Raiders
}

def _fix_abbr_series(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_FIX)

def _expected_home_prob(elo_home: float, elo_away: float, hfa: float = 55.0) -> float:
    # Elo expectation with home-field advantage in Elo points
    diff = (elo_home + hfa) - elo_away
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

def _update_elo(
    elo_home: float, elo_away: float, home_win: int, k: float = 20.0, hfa: float = 55.0
) -> Tuple[float, float]:
    ph = _expected_home_prob(elo_home, elo_away, hfa)
    # outcome: 1 for home win, 0 for away win
    delta_home = k * (home_win - ph)
    delta_away = -delta_home
    return elo_home + delta_home, elo_away + delta_away

def _train_elo_from_schedules(years: Iterable[int], k: float = 20.0, hfa: float = 55.0) -> Dict[str, float]:
    """Train Elo team ratings using historical schedules (scores)."""
    games = nfl.import_schedules(list(years))
    # Keep rows with final scores only
    games = games.loc[(games["home_score"].notna()) & (games["away_score"].notna())].copy()

    # Use modern abbreviations for stability
    games["home_team"] = _fix_abbr_series(games["home_team"])
    games["away_team"] = _fix_abbr_series(games["away_team"])

    ratings: Dict[str, float] = {}
    def get(team: str) -> float:
        return ratings.get(team, 1500.0)

    # Sort chronologically to simulate season flow
    if "gameday" in games.columns:
        games["gameday"] = pd.to_datetime(games["gameday"], errors="coerce")
        games = games.sort_values("gameday")
    else:
        games = games.sort_values(["season", "week"])

    for _, r in games.iterrows():
        ht, at = r["home_team"], r["away_team"]
        hs, as_ = float(r["home_score"]), float(r["away_score"])
        if pd.isna(hs) or pd.isna(as_):
            continue

        eh, ea = get(ht), get(at)
        home_win = 1 if hs > as_ else 0
        nh, na = _update_elo(eh, ea, home_win, k=k, hfa=hfa)
        ratings[ht], ratings[at] = nh, na

    return ratings

def train_elo_and_predict(
    upcoming_sched: pd.DataFrame,
    train_start: int = 2018,
    train_end: int = 2024,
    k: float = 20.0,
    hfa: float = 55.0,
) -> pd.DataFrame:
    """
    Train Elo on historical seasons and return model probabilities for upcoming games.

    Returns columns: home_team, away_team, home_prob_model, away_prob_model
    """
    ratings = _train_elo_from_schedules(range(train_start, train_end + 1), k=k, hfa=hfa)

    df = upcoming_sched.copy()
    df["home_team"] = _fix_abbr_series(df["home_team"])
    df["away_team"] = _fix_abbr_series(df["away_team"])

    def row_prob(r):
        eh = ratings.get(r["home_team"], 1500.0)
        ea = ratings.get(r["away_team"], 1500.0)
        p = _expected_home_prob(eh, ea, hfa=hfa)
        return p

    df["home_prob_model"] = df.apply(row_prob, axis=1)
    df["away_prob_model"] = 1.0 - df["home_prob_model"]
    return df[["home_team", "away_team", "home_prob_model", "away_prob_model"]]
