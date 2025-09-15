# nfl_model/modeling.py
from __future__ import annotations
import pandas as pd
import nfl_data_py as nfl

TEAM_FIX = {"LA":"LAR","STL":"LAR","SD":"LAC","OAK":"LV"}

def _fix(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_FIX)

def _expected_home_prob(elo_home: float, elo_away: float, hfa: float = 55.0) -> float:
    diff = (elo_home + hfa) - elo_away
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

def _update_elo(eh: float, ea: float, home_win: int, k: float = 20.0, hfa: float = 55.0) -> tuple[float,float]:
    ph = _expected_home_prob(eh, ea, hfa)
    d  = k * (home_win - ph)
    return eh + d, ea - d

def _train_elo(years: range, k=20.0, hfa=55.0) -> dict[str,float]:
    g = nfl.import_schedules(list(years))
    g = g[(g.home_score.notna()) & (g.away_score.notna())].copy()
    g["home_team"] = _fix(g["home_team"]); g["away_team"] = _fix(g["away_team"])
    if "gameday" in g: g["gameday"] = pd.to_datetime(g["gameday"], errors="coerce"); g = g.sort_values("gameday")
    else: g = g.sort_values(["season","week"])
    r: dict[str,float] = {}
    def get(t): return r.get(t, 1500.0)
    for _, row in g.iterrows():
        ht, at = row.home_team, row.away_team
        eh, ea = get(ht), get(at)
        hw = 1 if float(row.home_score) > float(row.away_score) else 0
        r[ht], r[at] = _update_elo(eh, ea, hw, k, hfa)
    return r

def train_elo_and_predict(upcoming: pd.DataFrame, train_start=2018, train_end=2024, k=20.0, hfa=55.0) -> pd.DataFrame:
    ratings = _train_elo(range(train_start, train_end+1), k=k, hfa=hfa)
    df = upcoming.copy()
    df["home_team"] = _fix(df["home_team"]); df["away_team"] = _fix(df["away_team"])
    def row_p(r):
        eh = ratings.get(r.home_team, 1500.0); ea = ratings.get(r.away_team, 1500.0)
        return _expected_home_prob(eh, ea, hfa)
    df["home_prob_model"] = df.apply(row_p, axis=1)
    df["away_prob_model"] = 1.0 - df["home_prob_model"]
    return df[["home_team","away_team","home_prob_model","away_prob_model"]]
