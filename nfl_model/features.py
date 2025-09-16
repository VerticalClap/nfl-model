# nfl_model/features.py
from __future__ import annotations
import pandas as pd

# Try to import the scaffold; if anything goes wrong, fall back to a no-op
try:
    from .rest_travel import add_rest_and_travel
except Exception:
    def add_rest_and_travel(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in [
            "home_rest_days","away_rest_days",
            "home_travel_miles","away_travel_miles"
        ]:
            if c not in out.columns:
                out[c] = 0.0
        return out

def _basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, always-available features so the pipeline runs end-to-end.
    You can extend this later (Elo, QB flags, injuries, etc.).
    """
    out = df.copy()
    if "home_team" in out.columns and "away_team" in out.columns:
        out["home_field"] = 1.0  # simple constant HFA placeholder
    else:
        out["home_field"] = 1.0
    return out

def build_upcoming_with_features(upcoming: pd.DataFrame, past_sched: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns (upcoming_with_features, feature_columns)
    Keeps columns needed by the pipeline and adds a minimal, working feature set.
    """
    cols = ["season","week","gameday","home_team","away_team","game_id"]
    base = upcoming.copy()
    for c in cols:
        if c not in base.columns:
            base[c] = None

    # Add simple features
    feat = _basic_features(base)

    # Add rest/travel scaffold (won’t break even if logic is minimal)
    feat = add_rest_and_travel(feat)

    # Feature column list for models (expand later)
    feature_cols = ["home_field", "home_rest_days", "away_rest_days", "home_travel_miles", "away_travel_miles"]

    return feat, feature_cols
