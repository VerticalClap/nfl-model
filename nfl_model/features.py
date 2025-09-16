# nfl_model/features.py
from __future__ import annotations
import pandas as pd
from .rest_travel import add_rest_and_travel

def attach_schedule_columns(upcoming: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize/ensure schedule columns exist with standard names.
    """
    out = upcoming.copy()
    # Make sure expected columns exist (some sources vary)
    needed = ["season","week","gameday","home_team","away_team"]
    for c in needed:
        if c not in out.columns:
            out[c] = pd.NA
    # Ensure proper types
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["gameday"] = pd.to_datetime(out["gameday"], errors="coerce")
    return out

def build_upcoming_with_features(
    upcoming: pd.DataFrame,
    past_sched: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Return upcoming games augmented with feature columns.
    Currently adds:
      - home_rest_days, away_rest_days, rest_delta
      - travel_km, travel_dir_km
    """
    base = attach_schedule_columns(upcoming)
    # Add rest + travel
    ft, ft_cols = add_rest_and_travel(base, past_sched)
    return ft, ft_cols
