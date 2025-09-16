# nfl_model/rest_travel.py
from __future__ import annotations
import pandas as pd

def add_rest_and_travel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal no-op scaffold so the pipeline never fails.
    Adds columns with zeros if missing.
    You can replace this later with real rest-days and travel-miles logic.
    """
    out = df.copy()
    for c in [
        "home_rest_days","away_rest_days",
        "home_travel_miles","away_travel_miles"
    ]:
        if c not in out.columns:
            out[c] = 0.0
    return out

# Back-compat alias (some earlier code called add_rest_travel)
add_rest_travel = add_rest_and_travel
