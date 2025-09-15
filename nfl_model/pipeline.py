from __future__ import annotations
import os, json
import pandas as pd
from .config import DATA_CACHE_DIR
from .odds import extract_moneylines, add_implied_probs

def _load_schedule(cache: str) -> pd.DataFrame:
    p = os.path.join(cache, "schedule.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p} — run scripts/fetch_and_build.py first.")
    return pd.read_csv(p, low_memory=False)

def _load_odds(cache: str) -> pd.DataFrame:
    p = os.path.join(cache, "odds_raw.json")
    if not os.path.exists(p):
        return pd.DataFrame()
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return extract_moneylines(raw)

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

    # ✅ Only keep current season
    current_year = pd.Timestamp.today().year
    if "season" in sched.columns:
        sched = sched[sched["season"] == current_year]

    # ✅ Only keep games today or later
    if "gameday" in sched.columns:
        sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
        today = pd.Timestamp.today().normalize()
        sched = sched[sched["gameday"] >= today]

    # Keep only needed columns
    cols_needed = [c for c in ["season","week","gameday","home_team","away_team","game_id"] if c in sched.columns]
    sched_small = sched[cols_needed].copy() if cols_needed else sched.copy()

    odds = _load_odds(cache)
    if odds.empty:
        merged = sched_small.copy()
        merged["home_ml"] = None
        merged["away_ml"] = None
    else:
        merged = pd.merge(
            sched_small, odds,
            on=["home_team","away_team"],
            how="left",
            suffixes=("","_odds")
        )

    merged = add_implied_probs(merged)

    # Kelly bet sizing
    merged["home_kelly_5pct"] = merged.apply(lambda r: kelly_fraction(r["home_prob"], r["home_ml"], 0.05), axis=1)
    merged["away_kelly_5pct"] = merged.apply(lambda r: kelly_fraction(r["away_prob"], r["away_ml"], 0.05), axis=1)

    outp = os.path.join(cache, "pick_sheet.csv")
    merged.to_csv(outp, index=False)
    print(f"Wrote {outp} ({len(merged)} rows)")
    return merged
