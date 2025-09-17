# nfl_model/features.py
from __future__ import annotations
import pandas as pd
import numpy as np

def _team_rollups(g: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-team rolling/season-to-date stats from a game-level schedule DataFrame.
    Expected columns if available:
      season, week, gameday, home_team, away_team, home_score, away_score
    We infer team points for/against and margin. If some cols are missing, we degrade gracefully.
    """
    cols = g.columns
    if "home_score" not in cols or "away_score" not in cols:
        # If scores aren’t available in your current season yet, create safe placeholders.
        g = g.copy()
        g["home_score"] = np.nan
        g["away_score"] = np.nan

    # Build a unified team-game log
    home = g.rename(columns={
        "home_team": "team", "away_team": "opp",
        "home_score": "points_for", "away_score": "points_against"
    })[["season","week","gameday","team","opp","points_for","points_against"]].copy()
    away = g.rename(columns={
        "away_team": "team", "home_team": "opp",
        "away_score": "points_for", "home_score": "points_against"
    })[["season","week","gameday","team","opp","points_for","points_against"]].copy()
    team_games = pd.concat([home, away], ignore_index=True).sort_values(["team","gameday"])

    team_games["margin"] = team_games["points_for"] - team_games["points_against"]
    team_games["played"] = (~team_games["points_for"].isna()).astype(int)

    # Rolling windows (last 5) and season-to-date means
    def _roll(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("gameday")
        # rolling last 5 (exclude current game – shift)
        df["pf_roll5"]  = df["points_for"].shift().rolling(5, min_periods=1).mean()
        df["pa_roll5"]  = df["points_against"].shift().rolling(5, min_periods=1).mean()
        df["mar_roll5"] = df["margin"].shift().rolling(5, min_periods=1).mean()
        # season-to-date (exclude current game)
        df["pf_szn"]  = df["points_for"].shift().expanding(min_periods=1).mean()
        df["pa_szn"]  = df["points_against"].shift().expanding(min_periods=1).mean()
        df["mar_szn"] = df["margin"].shift().expanding(min_periods=1).mean()
        # games played YTD before this one
        df["gms_szn"] = df["played"].shift().expanding(min_periods=1).sum()
        return df

    team_games = team_games.groupby(["season","team"], group_keys=False).apply(_roll)

    # When no history, backfill with league means by season to avoid NaNs
    def _fill_with_league_means(df: pd.DataFrame) -> pd.DataFrame:
        for c in ["pf_roll5","pa_roll5","mar_roll5","pf_szn","pa_szn","mar_szn"]:
            mappa = df.groupby("season")[c].transform(lambda s: s.mean(skipna=True))
            df[c] = df[c].fillna(mappa)
        df["gms_szn"] = df["gms_szn"].fillna(0)
        return df

    team_games = _fill_with_league_means(team_games)
    return team_games

def build_upcoming_with_features(schedule_upcoming: pd.DataFrame) -> pd.DataFrame:
    """
    Given a future/upcoming schedule (one row per game),
    attach model features from historical rollups computed on the same DataFrame
    (i.e., using rows earlier than each game). For current-year early weeks we’ll lean
    on roll5/season means computed from prior rows; if no history exists, we backfill with league averages.
    """
    if schedule_upcoming.empty:
        return schedule_upcoming.copy()

    # We need a “history + upcoming” union to compute rolling stats.
    # In-season: your schedule.csv includes only upcoming games. For robust training we’ll
    # let the model train on past seasons (in modeling.py). Here we still compute team priors
    # from whatever we have (backfilled to league avg).
    # So we just compute rollups over the upcoming frame itself for neutral priors.
    histlike = schedule_upcoming.copy()

    # If gameday missing, create an order key so rolling still works deterministically.
    if "gameday" not in histlike.columns:
        histlike["gameday"] = pd.Timestamp.today().normalize()

    team_games = _team_rollups(histlike)

    # Join home features
    home_feats = team_games.rename(columns={
        "team":"home_team",
        "pf_roll5":"home_pf_roll5", "pa_roll5":"home_pa_roll5", "mar_roll5":"home_mar_roll5",
        "pf_szn":"home_pf_szn", "pa_szn":"home_pa_szn", "mar_szn":"home_mar_szn",
        "gms_szn":"home_gms_szn"
    })[["season","gameday","home_team","home_pf_roll5","home_pa_roll5","home_mar_roll5",
        "home_pf_szn","home_pa_szn","home_mar_szn","home_gms_szn"]]

    # Join away features
    away_feats = team_games.rename(columns={
        "team":"away_team",
        "pf_roll5":"away_pf_roll5", "pa_roll5":"away_pa_roll5", "mar_roll5":"away_mar_roll5",
        "pf_szn":"away_pf_szn", "pa_szn":"away_pa_szn", "mar_szn":"away_mar_szn",
        "gms_szn":"away_gms_szn"
    })[["season","gameday","away_team","away_pf_roll5","away_pa_roll5","away_mar_roll5",
        "away_pf_szn","away_pa_szn","away_mar_szn","away_gms_szn"]]

    out = schedule_upcoming.merge(home_feats, on=["season","gameday","home_team"], how="left")
    out = out.merge(away_feats, on=["season","gameday","away_team"], how="left")

    # Derived differentials (home - away) that the model will like
    out["roll5_pf_diff"] = out["home_pf_roll5"] - out["away_pf_roll5"]
    out["roll5_pa_diff"] = out["home_pa_roll5"] - out["away_pa_roll5"]
    out["roll5_mar_diff"] = out["home_mar_roll5"] - out["away_mar_roll5"]

    out["szn_pf_diff"] = out["home_pf_szn"] - out["away_pf_szn"]
    out["szn_pa_diff"] = out["home_pa_szn"] - out["away_pa_szn"]
    out["szn_mar_diff"] = out["home_mar_szn"] - out["away_mar_szn"]

    # Home field indicator (simple baseline)
    out["home_field"] = 1.0

    # Fill any remaining NaNs with 0 (means we backed into league averages already)
    for c in out.columns:
        if out[c].dtype.kind in "fc":
            out[c] = out[c].fillna(0.0)

    return out
