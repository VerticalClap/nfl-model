# nfl_model/pipeline.py
from __future__ import annotations
import os, json
import pandas as pd
from .config import DATA_CACHE_DIR
from .odds import extract_moneylines, add_implied_probs

# Normalize old/ambiguous team codes in schedules to current abbreviations
TEAM_FIX = {
    "LA": "LAR",     # legacy Rams code in some schedules
    "SD": "LAC",     # old Chargers
    "OAK": "LV",     # old Raiders
    "STL": "LAR",    # old Rams
    # (add more if you ever see them)
}

def _fix_abbr(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_FIX)

def _load_schedule(cache: str) -> pd.DataFrame:
    p = os.path.join(cache, "schedule.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p} — run scripts/fetch_and_build.py first.")
    df = pd.read_csv(p, low_memory=False)

    # Ensure upcoming/current season only
    if "season" in df.columns:
        df = df[df["season"] == pd.Timestamp.today().year]
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
        today = pd.Timestamp.today().normalize()
        df = df[df["gameday"] >= today]

    # Fix any legacy abbreviations before merging with odds
    if "home_team" in df.columns and "away_team" in df.columns:
        df["home_team"] = _fix_abbr(df["home_team"])
        df["away_team"] = _fix_abbr(df["away_team"])

    keep = [c for c in ["season","week","gameday","home_team","away_team","game_id"] if c in df.columns]
    return df[keep].sort_values(["week","gameday","home_team","away_team"]).reset_index(drop=True)

def _load_odds(cache: str) -> pd.DataFrame:
    p = os.path.join(cache, "odds_raw.json")
    if not os.path.exists(p):
        return pd.DataFrame()
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return extract_moneylines(raw)  # returns home_ml/away_ml + fair probs

def kelly_fraction(p: float | None, ml: float | int | None, cap: float = 0.05) -> float:
    if p is None or ml is None:
        return 0.0
    ml = float(ml)
    b = (ml / 100.0) if ml >= 0 else (100.0 / (-ml))
    f = (p * (b + 1) - 1) / b
    if f <= 0:
        return 0.0
    return min(f, cap)

def build_pick_sheet(cache: str = DATA_CACHE_DIR) -> pd.DataFrame:
    sched = _load_schedule(cache)
    odds = _load_odds(cache)

    if odds.empty:
        merged = sched.copy()
        merged["home_ml"] = None
        merged["away_ml"] = None
        merged["home_prob"] = None
        merged["away_prob"] = None
    else:
        # Left join so we keep schedule rows even if some games lack lines yet
        merged = pd.merge(
            sched, odds,
            on=["home_team","away_team"],
            how="left",
            suffixes=("","_odds")
        )

    # Add raw implied + fair probabilities (no-op if MLs missing)
    merged = add_implied_probs(merged)

    # Kelly sizing (cap 5%)
    merged["home_kelly_5pct"] = merged.apply(lambda r: kelly_fraction(r.get("home_prob"), r.get("home_ml"), 0.05), axis=1)
    merged["away_kelly_5pct"] = merged.apply(lambda r: kelly_fraction(r.get("away_prob"), r.get("away_ml"), 0.05), axis=1)

    outp = os.path.join(cache, "pick_sheet.csv")
    merged.to_csv(outp, index=False)
    print(f"[pick_sheet] wrote {outp} ({len(merged)} rows)")
    return merged
