# nfl_model/pipeline.py
from __future__ import annotations
import os, json
import pandas as pd
from .odds import extract_consensus_moneylines, extract_consensus_spreads

TEAM_FIX = {"LA":"LAR", "SD":"LAC", "OAK":"LV"}

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("home_team","away_team"):
        df[col] = df[col].astype(str).str.upper().str.strip().replace(TEAM_FIX)
    return df

def build_pick_sheet(cache_dir: str = "./cache") -> pd.DataFrame:
    sched_path = os.path.join(cache_dir, "schedule.csv")
    odds_path  = os.path.join(cache_dir, "odds_raw.json")

    sched = pd.read_csv(sched_path, low_memory=False)
    _norm(sched)

    if os.path.exists(odds_path):
        raw = json.load(open(odds_path, "r", encoding="utf-8"))
        ml  = extract_consensus_moneylines(raw, prefer_books=("draftkings",))
        sp  = extract_consensus_spreads(raw,   prefer_books=("draftkings",))
    else:
        ml = pd.DataFrame(columns=["home_team","away_team","home_ml","away_ml","home_prob","away_prob"])
        sp = pd.DataFrame(columns=["home_team","away_team","home_line","home_spread_odds","away_spread_odds"])

    # Merge in two stages: schedule + ML, then + spreads
    out = sched.merge(ml, on=["home_team","away_team"], how="left")
    out = out.merge(sp, on=["home_team","away_team"], how="left")

    # Fill friendly columns if missing (so Streamlit never crashes)
    for c in ["home_ml","away_ml","home_prob","away_prob","home_line","home_spread_odds","away_spread_odds"]:
        if c not in out.columns:
            out[c] = pd.NA

    pick_path = os.path.join(cache_dir, "pick_sheet.csv")
    out.to_csv(pick_path, index=False)
    print(f"[pick_sheet] wrote {pick_path} ({len(out)} rows)")
    return out
