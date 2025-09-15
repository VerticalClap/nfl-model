from __future__ import annotations
import os
import json
import pandas as pd

from .config import DATA_CACHE_DIR
from .modeling import baseline_probs_from_odds

def _load_schedule(cache: str) -> pd.DataFrame:
    p = os.path.join(cache, "schedule.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p} — run scripts/fetch_and_build.py first.")
    df = pd.read_csv(p, low_memory=False)
    return df

def _load_odds(cache: str) -> pd.DataFrame:
    """Load odds if present and coerce to a simple home/away moneyline table."""
    p = os.path.join(cache, "odds_raw.json")
    if not os.path.exists(p):
        return pd.DataFrame()  # ok; we’ll just skip odds-based probs
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Shape odds into a flat table with home/away ML if available.
    rows = []
    for game in raw if isinstance(raw, list) else []:
        # Provider formats vary; we extract best-effort moneylines
        home, away = game.get("home_team"), game.get("away_team")
        home_ml, away_ml = None, None
        for b in game.get("bookmakers", []):
            for mk in b.get("markets", []):
                if mk.get("key") == "h2h":
                    # Two outcomes: teams matched to prices
                    for o in mk.get("outcomes", []):
                        if o.get("name") == home and "price" in o:
                            home_ml = o["price"]
                        if o.get("name") == away and "price" in o:
                            away_ml = o["price"]
        rows.append({"home_team": home, "away_team": away, "home_ml": home_ml, "away_ml": away_ml})
    return pd.DataFrame(rows)

def build_pick_sheet(cache: str = DATA_CACHE_DIR) -> pd.DataFrame:
    sched = _load_schedule(cache)
    odds  = _load_odds(cache)

    # Minimal join on home/away team strings if odds available
    if not odds.empty:
        merged = pd.merge(
            sched, odds,
            on=["home_team", "away_team"],
            how="left",
            suffixes=("", "_odds")
        )
    else:
        merged = sched.copy()
        merged["home_ml"] = None
        merged["away_ml"] = None

    probs = baseline_probs_from_odds(merged[["home_ml", "away_ml"]])
    merged = pd.concat([merged.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)

    outp = os.path.join(cache, "pick_sheet.csv")
    merged.to_csv(outp, index=False)
    print(f"Wrote {outp} ({len(merged)} rows)")
    return merged
