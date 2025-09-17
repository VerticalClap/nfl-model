# nfl_model/features.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

# We only import nfl_data_py here, so your other scripts can still run if it's not present.
try:
    import nfl_data_py as nfl
except Exception:
    nfl = None


def _safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:
        pass


def load_history_schedules(seasons: List[int]) -> pd.DataFrame:
    """
    Past seasons schedules (with final scores) used for training.
    """
    if nfl is None:
        _safe_print("[features] nfl_data_py not available; returning empty schedules.")
        return pd.DataFrame()

    _safe_print(f"[features] loading schedules for seasons: {seasons}")
    sch = nfl.import_schedules(seasons)
    # normalize common columns
    if "gameday" in sch.columns:
        sch["gameday"] = pd.to_datetime(sch["gameday"], errors="coerce")
    else:
        for alt in ["game_date", "start_time"]:
            if alt in sch.columns:
                sch["gameday"] = pd.to_datetime(sch[alt], errors="coerce")
                break

    keep = [c for c in [
        "game_id", "season", "week", "gameday",
        "home_team", "away_team",
        "home_score", "away_score"
    ] if c in sch.columns]
    sch = sch[keep].dropna(subset=["home_team", "away_team"]).reset_index(drop=True)
    return sch


def load_history_pbp(seasons: List[int]) -> pd.DataFrame:
    """
    Play-by-play for seasons (offense/defense EPA per game).
    """
    if nfl is None:
        _safe_print("[features] nfl_data_py not available; returning empty pbp.")
        return pd.DataFrame()

    _safe_print(f"[features] loading pbp for seasons: {seasons}")
    pbp = nfl.import_pbp_data(seasons, downcast=False)

    # keep offensive plays with EPA defined
    pbp = pbp.loc[pbp["epa"].notna() & pbp["posteam"].notna()].copy()
    # per-team, per-game offense epa/play & success rate
    off = (
        pbp.groupby(["game_id", "posteam"], as_index=False)
           .agg(plays=("epa", "size"),
                epa_sum=("epa", "sum"),
                success=("epa", lambda s: np.mean(s > 0)))
           .rename(columns={"posteam": "team"})
    )
    off["off_epa_per_play"] = off["epa_sum"] / off["plays"]

    # defense is just the opponent of offense on those plays
    pbp_def = pbp.copy()
    pbp_def["defteam"] = pbp_def["defteam"].fillna("")  # just in case
    deff = (
        pbp_def.groupby(["game_id", "defteam"], as_index=False)
               .agg(plays_allowed=("epa", "size"),
                    epa_allowed=("epa", "sum"),
                    success_allowed=("epa", lambda s: np.mean(s > 0)))
               .rename(columns={"defteam": "team"})
    )
    deff["def_epa_per_play_allowed"] = deff["epa_allowed"] / deff["plays_allowed"]

    # merge offense+defense per team-game
    team_game = pd.merge(off, deff, on=["game_id", "team"], how="outer")
    return team_game


def build_team_week_table(sched: pd.DataFrame, team_game: pd.DataFrame) -> pd.DataFrame:
    """
    Convert game-level into team-week rows with rolling features.
    """
    if sched.empty:
        return pd.DataFrame()

    # explode schedule into 2 rows per game: (home team) and (away team)
    home_rows = sched[["game_id", "season", "week", "gameday", "home_team", "away_team"]].copy()
    home_rows = home_rows.rename(columns={"home_team": "team", "away_team": "opp"})
    home_rows["is_home"] = 1

    away_rows = sched[["game_id", "season", "week", "gameday", "home_team", "away_team"]].copy()
    away_rows = away_rows.rename(columns={"away_team": "team", "home_team": "opp"})
    away_rows["is_home"] = 0

    tw = pd.concat([home_rows, away_rows], ignore_index=True)

    # attach team-game metrics (off/def epa/success)
    tw = tw.merge(team_game, on=["game_id", "team"], how="left")

    # rolling windows per team within season ordered by week
    tw = tw.sort_values(["team", "season", "week"]).reset_index(drop=True)

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        # previous N games (no leakage) -> shift before rolling
        for col in [
            "off_epa_per_play", "success", "def_epa_per_play_allowed", "success_allowed"
        ]:
            if col in g.columns:
                g[f"{col}_roll8"] = g[col].shift(1).rolling(8, min_periods=3).mean()
                g[f"{col}_roll4"] = g[col].shift(1).rolling(4, min_periods=2).mean()
        return g

    tw = tw.groupby(["team", "season"], group_keys=False).apply(_roll)
    return tw


def make_game_feature_rows(sched_any: pd.DataFrame, tw: pd.DataFrame) -> pd.DataFrame:
    """
    For each game row in `sched_any` create model-ready features by joining
    home & away team rolling stats from tw (team-week table).
    """
    if sched_any.empty or tw.empty:
        return pd.DataFrame()

    key_cols = ["season", "week", "home_team", "away_team", "game_id", "gameday"]
    base = sched_any[key_cols].copy()

    # Last known rolling stats for each team BEFORE this week
    sel_cols = [
        "team", "season", "week",
        "off_epa_per_play_roll8", "off_epa_per_play_roll4",
        "success_roll8", "success_roll4",
        "def_epa_per_play_allowed_roll8", "def_epa_per_play_allowed_roll4",
        "success_allowed_roll8", "success_allowed_roll4",
    ]
    for c in sel_cols:
        if c not in tw.columns and c not in ["team", "season", "week"]:
            tw[c] = np.nan
    tw_sel = tw[sel_cols].copy()

    # For week W game, use team's stats from week W-1 (already rolled with shift)
    # Join home
    home = tw_sel.rename(columns={
        "team": "home_team",
        "off_epa_per_play_roll8": "home_off_epp8",
        "off_epa_per_play_roll4": "home_off_epp4",
        "success_roll8": "home_off_sr8",
        "success_roll4": "home_off_sr4",
        "def_epa_per_play_allowed_roll8": "home_def_eppa8",
        "def_epa_per_play_allowed_roll4": "home_def_eppa4",
        "success_allowed_roll8": "home_def_sr8",
        "success_allowed_roll4": "home_def_sr4",
    })
    away = tw_sel.rename(columns={
        "team": "away_team",
        "off_epa_per_play_roll8": "away_off_epp8",
        "off_epa_per_play_roll4": "away_off_epp4",
        "success_roll8": "away_off_sr8",
        "success_roll4": "away_off_sr4",
        "def_epa_per_play_allowed_roll8": "away_def_eppa8",
        "def_epa_per_play_allowed_roll4": "away_def_eppa4",
        "success_allowed_roll8": "away_def_sr8",
        "success_allowed_roll4": "away_def_sr4",
    })

    feat = (
        base.merge(home.drop(columns=["season", "week"]), on="home_team", how="left")
            .merge(away.drop(columns=["season", "week"]), on="away_team", how="left")
    )

    # Simple differentials — these drive the spread model
    feat["x_epa_off"] = feat["home_off_epp8"] - feat["away_off_epp8"]
    feat["x_sr_off"]  = feat["home_off_sr8"]  - feat["away_off_sr8"]
    feat["x_epa_def"] = feat["away_def_eppa8"] - feat["home_def_eppa8"]  # lower allowed is better
    feat["x_sr_def"]  = feat["away_def_sr8"]   - feat["home_def_sr8"]

    # Where 8-game windows are missing early season, fall back to 4-game
    for a, b in [("home_off_epp8", "home_off_epp4"),
                 ("away_off_epp8", "away_off_epp4"),
                 ("home_off_sr8",  "home_off_sr4"),
                 ("away_off_sr8",  "away_off_sr4"),
                 ("home_def_eppa8","home_def_eppa4"),
                 ("away_def_eppa8","away_def_eppa4"),
                 ("home_def_sr8",  "home_def_sr4"),
                 ("away_def_sr8",  "away_def_sr4"),
                 ("x_epa_off","x_epa_off"), ("x_sr_off","x_sr_off"),
                 ("x_epa_def","x_epa_def"), ("x_sr_def","x_sr_def")]:
        if a in feat.columns and b in feat.columns:
            feat[a] = feat[a].fillna(feat[b])

    return feat


def build_training_and_upcoming(
    seasons_train: List[int],
    sched_upcoming: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      train_df: past games with target 'home_margin' and feature columns.
      up_df:    upcoming games (rows aligned to sched_upcoming) with same features.
    """
    if nfl is None:
        _safe_print("[features] nfl_data_py not available; returning empty frames.")
        return pd.DataFrame(), pd.DataFrame()

    # TRAIN: past schedules + pbp features
    sch_hist = load_history_schedules(seasons_train)
    pbp_hist = load_history_pbp(seasons_train)
    tw_hist  = build_team_week_table(sch_hist, pbp_hist)
    feat_hist = make_game_feature_rows(sch_hist, tw_hist)

    # training target: home margin
    if {"home_score", "away_score"}.issubset(sch_hist.columns):
        results = sch_hist[["game_id", "home_score", "away_score"]].copy()
        results["home_margin"] = results["home_score"] - results["away_score"]
    else:
        results = pd.DataFrame({"game_id": [], "home_margin": []})
    train_df = feat_hist.merge(results[["game_id", "home_margin"]], on="game_id", how="inner")
    train_df = train_df.dropna(subset=["home_margin"])

    # UPCOMING: build features using the same pipeline (uses season/week to pick last rolling stats)
    # We need historical pbp to form rolling stats; use the same tw_hist table (latest known).
    up_df = make_game_feature_rows(sched_upcoming, tw_hist)

    _safe_print(f"[features] training rows: {len(train_df)} / upcoming rows: {len(up_df)}")
    return train_df, up_df
