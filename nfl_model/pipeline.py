# nfl_model/pipeline.py
from __future__ import annotations
import os, json
import pandas as pd
from .config import DATA_CACHE_DIR
from .odds import extract_moneylines, extract_spreads, add_implied_probs
from .modeling import train_elo_and_predict

TEAM_FIX = {"LA":"LAR","STL":"LAR","SD":"LAC","OAK":"LV"}
def _fix(s: pd.Series) -> pd.Series: return s.replace(TEAM_FIX)

BOOKS = ["draftkings"]     # <- change to [] to use all books’ median

def _load_schedule(cache: str) -> pd.DataFrame:
    p = os.path.join(cache, "schedule.csv")
    df = pd.read_csv(p, low_memory=False)
    if "season" in df: df = df[df["season"] == pd.Timestamp.today().year]
    if "gameday" in df:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
        df = df[df["gameday"] >= pd.Timestamp.today().normalize()]
    df["home_team"] = _fix(df["home_team"]); df["away_team"] = _fix(df["away_team"])
    keep = [c for c in ["season","week","gameday","home_team","away_team","game_id"] if c in df.columns]
    return df[keep].sort_values(["week","gameday","home_team","away_team"]).reset_index(drop=True)

def _load_odds(cache: str):
    p = os.path.join(cache, "odds_raw.json")
    if not os.path.exists(p): return pd.DataFrame(), pd.DataFrame()
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # If BOOKS is [], we use all books. Otherwise only those keys (e.g. draftkings)
    mls = extract_moneylines(raw, bookmakers=BOOKS if BOOKS else None)
    spr = extract_spreads(raw, bookmakers=BOOKS if BOOKS else None)
    return mls, spr

def kelly_fraction(p: float | None, ml: float | int | None, cap: float = 0.05) -> float:
    if p is None or ml is None or pd.isna(p) or pd.isna(ml): return 0.0
    ml = float(ml)
    b = (ml/100.0) if ml >= 0 else (100.0/(-ml))
    f = (p*(b+1)-1)/b
    return 0.0 if f <= 0 else min(f, cap)

def build_pick_sheet(cache: str = DATA_CACHE_DIR) -> pd.DataFrame:
    sched = _load_schedule(cache)
    model = train_elo_and_predict(sched, train_start=2018, train_end=2024)

    mls, spr = _load_odds(cache)

    out = sched.merge(model, on=["home_team","away_team"], how="left")

    # Moneyline merge + implied/fair probs
    if not mls.empty:
        out = out.merge(mls, on=["home_team","away_team"], how="left")
        out = add_implied_probs(out)
    else:
        out["home_ml"]=out["away_ml"]=out["home_prob"]=out["away_prob"]=None

    # Spreads merge (includes cover probs if book prices present)
    if not spr.empty:
        out = out.merge(spr, on=["home_team","away_team"], how="left")

    # Edges vs market (moneyline fair)
    out["home_edge"] = out["home_prob_model"] - out["home_prob"]
    out["away_edge"] = out["away_prob_model"] - out["away_prob"]

    # Kelly vs moneyline using YOUR model
    out["home_kelly_5pct"] = out.apply(lambda r: kelly_fraction(r.get("home_prob_model"), r.get("home_ml")), axis=1)
    out["away_kelly_5pct"] = out.apply(lambda r: kelly_fraction(r.get("away_prob_model"), r.get("away_ml")), axis=1)

    # ATS: model spread vs book spread
    # If model_spread_home < book home_spread (more fav), suggests home ATS value, etc.
    out["spread_edge_pts"] = out["model_spread_home"] - out.get("home_spread")
    # We keep book’s cover probs (vig-removed) if present for quick sanity display.
    # Later we’ll replace with model cover probs from a trained ATS classifier.

    outp = os.path.join(cache, "pick_sheet.csv")
    out.to_csv(outp, index=False)
    print(f"[pick_sheet] wrote {outp} ({len(out)} rows)")
    return out
