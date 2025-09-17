# nfl_model/modeling.py
from __future__ import annotations
import os
import math
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import nfl_data_py as nfl


# ----------------------------
# Helpers: rolling team metrics
# ----------------------------
def _team_week_aggs(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-team per-week offensive and defensive efficiency from nflfastR pbp.
    Outputs one row per (season, week, team) with offensive & defensive metrics.
    """
    # Ensure required columns exist
    need = {"season", "week", "posteam", "defteam", "epa", "pass", "rush", "success"}
    missing = [c for c in need if c not in pbp.columns]
    if missing:
        raise ValueError(f"nflfastR pbp missing columns: {missing}")

    # Coerce dtypes
    for c in ["pass", "rush", "success"]:
        pbp[c] = pbp[c].astype("float64")

    # Offensive side (by posteam)
    off = (
        pbp.groupby(["season", "week", "posteam"])
           .agg(
                plays=("epa", "size"),
                epa_per_play=("epa", "mean"),
                pass_plays=("pass", "sum"),
                rush_plays=("rush", "sum"),
                pass_epa=("epa", lambda s: s[pbp.loc[s.index, "pass"] == 1].mean() if len(s) else np.nan),
                rush_epa=("epa", lambda s: s[pbp.loc[s.index, "rush"] == 1].mean() if len(s) else np.nan),
                success_rate=("success", "mean"),
           )
           .reset_index()
           .rename(columns={"posteam": "team"})
    )

    # Defensive side (by defteam) – what they allowed
    deff = (
        pbp.groupby(["season", "week", "defteam"])
           .agg(
                plays_allowed=("epa", "size"),
                epa_allowed=("epa", "mean"),
                pass_epa_allowed=("epa", lambda s: s[pbp.loc[s.index, "pass"] == 1].mean() if len(s) else np.nan),
                rush_epa_allowed=("epa", lambda s: s[pbp.loc[s.index, "rush"] == 1].mean() if len(s) else np.nan),
                success_allowed=("success", "mean"),
           )
           .reset_index()
           .rename(columns={"defteam": "team"})
    )

    # Merge off & def
    tw = pd.merge(off, deff, on=["season", "week", "team"], how="outer")
    # Fill NaNs where no plays with 0’s where sensible (rates can be left NaN and handled in rolling)
    fill0 = ["plays", "pass_plays", "rush_plays", "plays_allowed"]
    for c in fill0:
        if c in tw.columns:
            tw[c] = tw[c].fillna(0)

    return tw


def _rolling_lastN(tw: pd.DataFrame, N: int = 4) -> pd.DataFrame:
    """
    For each team-season, compute rolling last N averages up to week-1 (so we don't use future).
    """
    metrics = [
        "epa_per_play", "pass_epa", "rush_epa", "success_rate",
        "epa_allowed", "pass_epa_allowed", "rush_epa_allowed", "success_allowed"
    ]
    tw = tw.sort_values(["team", "season", "week"]).copy()
    frames = []
    for (team, season), g in tw.groupby(["team", "season"], sort=False):
        g = g.sort_values("week")
        # shift by 1 so week k uses info through (k-1)
        rolled = g[metrics].rolling(N, min_periods=1).mean().shift(1)
        for c in rolled.columns:
            g[f"{c}_last{N}"] = rolled[c]
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    return out


# -----------------------------------------
# Build training table & fit linear model
# -----------------------------------------
def _make_training_table(sched: pd.DataFrame, tw_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rolling team features into completed games to train on margin (home - away).
    """
    need_sched = {"season", "week", "home_team", "away_team", "home_score", "away_score"}
    if not need_sched.issubset(sched.columns):
        raise ValueError(f"Schedule missing columns for training: {sorted(need_sched - set(sched.columns))}")

    # Completed games only (has scores)
    hist = sched.dropna(subset=["home_score", "away_score"]).copy()
    hist["margin"] = hist["home_score"] - hist["away_score"]

    # Join home features
    hf = tw_feat.rename(columns=lambda c: f"home_{c}" if c not in ["season", "week", "team"] else c)
    af = tw_feat.rename(columns=lambda c: f"away_{c}" if c not in ["season", "week", "team"] else c)

    hist = hist.merge(
        hf, left_on=["season", "week", "home_team"], right_on=["season", "week", "team"], how="left"
    ).drop(columns=["team"])

    hist = hist.merge(
        af, left_on=["season", "week", "away_team"], right_on=["season", "week", "team"], how="left"
    ).drop(columns=["team"])

    # Build feature diffs (home minus away)
    def pick(cols): return [c for c in hist.columns if any(c.endswith(x) for x in cols)]
    last_cols = [
        "_last4"
    ]
    base_names = [
        "epa_per_play", "pass_epa", "rush_epa", "success_rate",
        "epa_allowed", "pass_epa_allowed", "rush_epa_allowed", "success_allowed"
    ]
    feats = {}
    for base in base_names:
        hcol = f"home_{base}_last4"
        acol = f"away_{base}_last4"
        if hcol in hist.columns and acol in hist.columns:
            feats[f"diff_{base}_last4"] = hist[hcol] - hist[acol]

    X = pd.DataFrame(feats)
    y = hist["margin"]

    # Basic cleanup
    X = X.fillna(0.0).astype(float)
    y = y.astype(float)

    # Drop rows with all-zero features (rare, early-season)
    mask = (X.abs().sum(axis=1) > 0)
    X = X[mask]
    y = y[mask]
    hist = hist.loc[mask].reset_index(drop=True)

    hist["_X"] = [row for _, row in X.iterrows()]
    hist["_y"] = y.values
    return hist


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 5.0) -> Tuple[np.ndarray, float]:
    """
    Solve (X^T X + alpha*I) w = X^T y  ; return w and residual std (sigma).
    """
    n_feat = X.shape[1]
    A = X.T @ X + alpha * np.eye(n_feat)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    residuals = y - (X @ w)
    sigma = residuals.std(ddof=max(1, n_feat))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 13.5  # fallback typical NFL margin stdev
    return w, sigma


# -----------------------------------------
# Public API
# -----------------------------------------
def train_and_predict_upcoming(cache_dir: str, season_from: int = 2015) -> pd.DataFrame:
    """
    Train on prior seasons + YTD using nflfastR pbp (via nfl_data_py), predict upcoming games in cache/schedule.csv.
    Returns: DataFrame with columns [game_id, model_spread, model_home_prob]
    """
    sched_path = os.path.join(cache_dir, "schedule.csv")
    if not os.path.exists(sched_path):
        raise FileNotFoundError(f"schedule.csv not found in {cache_dir}")

    sched = pd.read_csv(sched_path, low_memory=False)
    if "season" not in sched.columns or "week" not in sched.columns:
        raise ValueError("schedule.csv missing season/week")

    # Load pbp for training window
    seasons = list(range(season_from, int(sched["season"].max()) + 1))
    pbp = nfl.import_pbp_data(years=seasons, downcast=False)

    # Build & roll team features
    tw = _team_week_aggs(pbp)
    tw = _rolling_lastN(tw, N=4)

    # Get a schedule with scores for training (historical)
    all_sched = nfl.import_schedules(seasons)
    train_tab = _make_training_table(all_sched, tw)

    # Assemble X, y
    diff_cols = [c for c in train_tab.columns if c.startswith("diff_") and c.endswith("_last4")]
    X = train_tab[diff_cols].to_numpy(dtype=float)
    y = train_tab["_y"].to_numpy(dtype=float)

    # Fit
    w, sigma = _fit_ridge(X, y, alpha=5.0)

    # Predict for upcoming games in our cached schedule
    upc = sched.copy()
    upc["gameday"] = pd.to_datetime(upc.get("gameday", pd.NaT), errors="coerce")
    today = pd.Timestamp.today().normalize()
    upc = upc[(upc["gameday"].isna()) | (upc["gameday"] >= today)].copy()

    # Merge rolling features for current season (use last4 up to week-1)
    hf = tw.rename(columns=lambda c: f"home_{c}" if c not in ["season", "week", "team"] else c)
    af = tw.rename(columns=lambda c: f"away_{c}" if c not in ["season", "week", "team"] else c)

    upc = upc.merge(
        hf, left_on=["season", "week", "home_team"], right_on=["season", "week", "team"], how="left"
    ).drop(columns=["team"])
    upc = upc.merge(
        af, left_on=["season", "week", "away_team"], right_on=["season", "week", "team"], how="left"
    ).drop(columns=["team"])

    # Build diffs for prediction
    feats = {}
    for base in [
        "epa_per_play", "pass_epa", "rush_epa", "success_rate",
        "epa_allowed", "pass_epa_allowed", "rush_epa_allowed", "success_allowed"
    ]:
        hcol = f"home_{base}_last4"
        acol = f"away_{base}_last4"
        if hcol in upc.columns and acol in upc.columns:
            feats[f"diff_{base}_last4"] = upc[hcol] - upc[acol]
        else:
            feats[f"diff_{base}_last4"] = 0.0

    X_pred = pd.DataFrame(feats).fillna(0.0).astype(float).to_numpy(dtype=float)
    pred_margin = X_pred @ w  # home - away (points)

    # Convert to win probability (home)
    # P(home wins) = 1 - CDF(0; mean=pred_margin, sd=sigma)
    # Use logistic approx to avoid SciPy dependency:
    z = pred_margin / max(1e-6, sigma)
    # Normal CDF approx (logistic with pi/sqrt(3) scale)
    home_prob = 1 / (1 + np.exp(-1.702 * z))  # calibration factor ~1.7

    out = upc[["game_id"]].copy()
    out["model_spread"] = pred_margin
    out["model_home_prob"] = home_prob.clip(0.01, 0.99)

    return out
