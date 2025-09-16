# nfl_model/pipeline.py
from __future__ import annotations
import os, json, pandas as pd
from .config import DATA_CACHE_DIR
from .odds import extract_moneylines, extract_spreads, extract_totals
from .features import build_upcoming_with_features

TEAM_FIX = {"LA":"LAR","STL":"LAR","SD":"LAC","OAK":"LV"}
def _fix(s: pd.Series) -> pd.Series: return s.replace(TEAM_FIX)

# Set to ["draftkings"] to use ONLY DraftKings; set to [] to use median across all books.
BOOKS: list[str] = ["draftkings"]   # change to [] for consensus

def _load_schedule(cache: str) -> pd.DataFrame:
    p = os.path.join(cache, "schedule.csv")
    df = pd.read_csv(p, low_memory=False)
    # normalize teams
    df["home_team"] = _fix(df["home_team"]); df["away_team"] = _fix(df["away_team"])
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    return df

def _load_odds_raw(cache: str):
    p = os.path.join(cache, "odds_raw.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def build_pick_sheet(cache: str = DATA_CACHE_DIR) -> pd.DataFrame:
    cache = cache or DATA_CACHE_DIR
    sched = _load_schedule(cache)

    # Upcoming (current & future)
    up = sched.copy()
    if "gameday" in up.columns:
        up = up[up["gameday"] >= pd.Timestamp.today().normalize()]

    # Features (rest/travel for now) using the full season schedule as "history"
    feats, feat_cols = build_upcoming_with_features(
        upcoming=up[["season","week","gameday","home_team","away_team","game_id"]],
        past_sched=sched
    )

    # Odds
    raw = _load_odds_raw(cache)
    if raw is not None:
        mls = extract_moneylines(raw, bookmakers=BOOKS if BOOKS else None)
        spr = extract_spreads(raw,    bookmakers=BOOKS if BOOKS else None)
        tot = extract_totals(raw,     bookmakers=BOOKS if BOOKS else None)
    else:
        mls = spr = tot = pd.DataFrame()

    out = feats.copy()

    if not mls.empty:
        out = out.merge(mls, on=["home_team","away_team"], how="left")
    else:
        for c in ["home_ml","away_ml","home_prob","away_prob","home_prob_raw","away_prob_raw"]:
            out[c] = None

    if not spr.empty:
        out = out.merge(spr, on=["home_team","away_team"], how="left")

    if not tot.empty:
        out = out.merge(tot, on=["home_team","away_team"], how="left")

    # Arrange columns
    order_front = ["season","week","gameday","home_team","away_team","game_id"]
    ml_cols = ["home_ml","away_ml","home_prob","away_prob","home_prob_raw","away_prob_raw"]
    sp_cols = ["home_spread","away_spread","home_spread_price","away_spread_price","home_cover_prob","away_cover_prob"]
    tot_cols= ["total_points","over_price","under_price","over_prob","under_prob"]
    feat_cols = [c for c in out.columns if c in ("home_rest_days","away_rest_days","rest_delta","travel_km","travel_dir_km")]
    other = [c for c in out.columns if c not in (order_front + ml_cols + sp_cols + tot_cols + feat_cols)]
    cols = [c for c in order_front if c in out.columns] + ml_cols + sp_cols + tot_cols + feat_cols + other
    cols = [c for c in cols if c in out.columns]  # guard
    out = out[cols]

    out_path = os.path.join(cache, "pick_sheet.csv")
    out.to_csv(out_path, index=False)
    print(f"[pick_sheet] wrote {out_path} ({len(out)} rows)")
    return out
