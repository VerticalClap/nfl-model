# nfl_model/pipeline.py
from __future__ import annotations
import os, json
import pandas as pd
from .config import DATA_CACHE_DIR
from .odds import extract_moneylines, extract_spreads, add_implied_probs
from .modeling import train_elo_and_predict

TEAM_FIX = {"LA":"LAR","STL":"LAR","SD":"LAC","OAK":"LV"}
def _fix(s: pd.Series) -> pd.Series: return s.replace(TEAM_FIX)

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
    return extract_moneylines(raw), extract_spreads(raw)

def kelly(p: float | None, ml: float | int | None, cap: float = 0.05) -> float:
    if p is None or ml is None or pd.isna(p) or pd.isna(ml): return 0.0
    ml = float(ml)
    b = (ml/100.0) if ml >= 0 else (100.0/(-ml))
    f = (p*(b+1)-1)/b
    return 0.0 if f <= 0 else min(f, cap)

def build_pick_sheet(cache: str = DATA_CACHE_DIR) -> pd.DataFrame:
    sched = _load_schedule(cache)

    # model probs (Elo trained 2018–2024)
    model = train_elo_and_predict(sched, train_start=2018, train_end=2024)

    # odds: ML + spreads
    mls, spr = _load_odds(cache)

    out = sched.merge(model, on=["home_team","away_team"], how="left")

    if not mls.empty:
        out = out.merge(mls, on=["home_team","away_team"], how="left")
        out = add_implied_probs(out)
    else:
        out["home_ml"]=out["away_ml"]=out["home_prob"]=out["away_prob"]=None

    if not spr.empty:
        out = out.merge(spr, on=["home_team","away_team"], how="left")

    # edges vs market-fair
    out["home_edge"] = out["home_prob_model"] - out["home_prob"]
    out["away_edge"] = out["away_prob_model"] - out["away_prob"]

    # Kelly vs book using YOUR model prob
    out["home_kelly_5pct"] = out.apply(lambda r: kelly(r.get("home_prob_model"), r.get("home_ml")), axis=1)
    out["away_kelly_5pct"] = out.apply(lambda r: kelly(r.get("away_prob_model"), r.get("away_ml")), axis=1)

    out_path = os.path.join(cache, "pick_sheet.csv")
    out.to_csv(out_path, index=False)
    print(f"[pick_sheet] wrote {out_path} ({len(out)} rows)")
    return out
