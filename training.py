# nfl_model/training.py
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import joblib

import nfl_data_py as nfl
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from .features import build_upcoming_with_features

# Where models live
ART_DIR = os.path.join("cache", "models")
WIN_ART = os.path.join(ART_DIR, "win_clf.pkl")
ATS_ART = os.path.join(ART_DIR, "ats_clf.pkl")
META_ART = os.path.join(ART_DIR, "meta.json")
os.makedirs(ART_DIR, exist_ok=True)

TEAM_FIX = {"LA":"LAR","STL":"LAR","SD":"LAC","OAK":"LV"}
def _fix(s: pd.Series) -> pd.Series: return s.replace(TEAM_FIX)

def _label_home_win(row) -> int:
    return 1 if float(row["home_score"]) > float(row["away_score"]) else 0

def _label_home_cover(row, line_col="home_line") -> int:
    # True if home team covers given (home) closing spread: margin + line > 0
    margin = float(row["home_score"]) - float(row["away_score"])
    line = float(row[line_col]) if pd.notna(row[line_col]) else 0.0
    return 1 if margin + line > 0 else 0

def _fit_iso_logit(X: np.ndarray, y: np.ndarray) -> CalibratedClassifierCV:
    base = LogisticRegression(max_iter=300, solver="lbfgs")
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    return clf.fit(X, y)

def _prep_history(seasons: list[int]) -> pd.DataFrame:
    # Completed games only, normalized teams
    sched = nfl.import_schedules(seasons).copy()
    sched["home_team"] = _fix(sched["home_team"])
    sched["away_team"] = _fix(sched["away_team"])
    sched = sched[(sched["home_score"].notna()) & (sched["away_score"].notna())]
    sched["game_id"] = sched["game_id"].astype(str)

    # Try to pull closing lines (some seasons/books may be missing)
    sched["home_line"] = np.nan
    try:
        lines = nfl.import_lines(seasons)
        lines["team"] = _fix(lines["team"])
        # Home-rows → home closing line per game
        hl = (lines[lines["home_away"] == "home"]
                    .loc[:, ["game_id","closing_line"]]
                    .rename(columns={"closing_line":"home_line"}))
        sched = sched.merge(hl, on="game_id", how="left")
    except Exception:
        # If lines endpoint hiccups, we still train the win model
        pass

    # Keep minimal columns trainer needs
    keep = ["season","week","gameday","home_team","away_team","game_id","home_score","away_score","home_line"]
    return sched[keep]

def train_models(train_years: list[int] | None = None) -> dict:
    """
    Trains:
      - WIN model: P(home wins)
      - ATS model (if home_line available): P(home covers closing spread)
    Persists artifacts into cache/models/*.pkl
    """
    if train_years is None:
        # 2018–2024 gives modern era with plenty of data
        train_years = list(range(2018, 2025))

    hist = _prep_history(train_years)

    # Build features by treating history rows as "upcoming" and using history as context
    X_df, feat_cols = build_upcoming_with_features(
        upcoming=hist[["season","week","gameday","home_team","away_team","game_id"]],
        past_sched=hist
    )
    # Attach labels
    df = X_df.merge(hist, on=["season","week","gameday","home_team","away_team","game_id"], how="left")
    df = df.dropna(subset=["home_score","away_score"]).copy()

    # Features matrix
    X = df[feat_cols].fillna(0.0).to_numpy()

    # --- WIN model ---
    y_win = df.apply(_label_home_win, axis=1).astype(int).to_numpy()
    win_clf = _fit_iso_logit(X, y_win)
    joblib.dump({"clf": win_clf, "feat_cols": feat_cols}, WIN_ART)

    # --- ATS model (optional if we have lines) ---
    ats_trained = False
    if "home_line" in df.columns and df["home_line"].notna().any():
        y_cov = df.apply(_label_home_cover, axis=1).astype(int).to_numpy()
        ats_clf = _fit_iso_logit(X, y_cov)
        joblib.dump({"clf": ats_clf, "feat_cols": feat_cols}, ATS_ART)
        ats_trained = True

    meta = {"train_years": train_years, "features": feat_cols, "ats_trained": ats_trained, "n_samples": int(df.shape[0])}
    with open(META_ART, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Training complete:", meta)
    return meta

if __name__ == "__main__":
    train_models()
