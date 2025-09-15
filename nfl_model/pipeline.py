# nfl_model/pipeline.py
from __future__ import annotations
import os, json
import pandas as pd
from .config import DATA_CACHE_DIR
from .odds import extract_moneylines, add_implied_probs
from .modeling import train_elo_and_predict

TEAM_FIX = {"LA":"LAR","STL":"LAR","SD":"LAC","OAK":"LV"}

def _fix_abbr(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_FIX)

def _load_schedule(cache: str) -> pd.DataFrame:
    p = os.path.join(cache, "schedule.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}. Run scripts/fetch_and_build.py first.")
    df = pd.read_csv(p, low_memory=False)
    # Keep current season and upcoming games
    if "season" in df.columns:
        df = df[df["season"] == pd.Timestamp.today().year]
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
        today = pd.Timestamp.today().normalize()
        df = df[df["gameday"] >= today]
    if "home_team" in df and "away_team" in df:
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
    return extract_moneylines(raw)

def kelly_fraction(p: float | None, ml: float | int | None, cap: float = 0.05) -> float:
    if p is None or ml is None:
        return 0.0
    ml = float(ml)
    # b is decimal profit per $1
    b = (ml / 100.0) if ml >= 0 else (100.0 / (-ml))
    f = (p * (b + 1.0) - 1.0) / b
    if f <= 0:
        return 0.0
    return min(f, cap)

def build_pick_sheet(cache: str = DATA_CACHE_DIR) -> pd.DataFrame:
    sched = _load_schedule(cache)
    odds = _load_odds(cache)

    # 1) model probabilities (Elo), trained on 2018–2024 results
    model_probs = train_elo_and_predict(sched, train_start=2018, train_end=2024)

    # 2) merge schedule + odds + model probs
    out = pd.merge(sched, model_probs, on=["home_team","away_team"], how="left")
    if not odds.empty:
        out = pd.merge(out, odds, on=["home_team","away_team"], how="left")
    else:
        out["home_ml"] = None
        out["away_ml"] = None

    # 3) compute implied/fair from odds (no-op if MLs missing)
    out = add_implied_probs(out)

    # 4) Kelly using YOUR model probs vs the book’s ML
    out["home_kelly_5pct"] = out.apply(lambda r: kelly_fraction(r.get("home_prob_model"), r.get("home_ml"), 0.05), axis=1)
    out["away_kelly_5pct"] = out.apply(lambda r: kelly_fraction(r.get("away_prob_model"), r.get("away_ml"), 0.05), axis=1)

    # 5) Optional: show edge vs market fair (model - market)
    out["home_edge"] = out["home_prob_model"] - out.get("home_prob", 0.0)
    out["away_edge"] = out["away_prob_model"] - out.get("away_prob", 0.0)

    outp = os.path.join(cache, "pick_sheet.csv")
    out.to_csv(outp, index=False)
    print(f"[pick_sheet] wrote {outp} ({len(out)} rows)")
    return out
