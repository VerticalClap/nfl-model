# nfl_model/pipeline.py
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import joblib

from .config import DATA_CACHE_DIR
from .odds import extract_consensus_moneylines  # DK or consensus
from .features import build_upcoming_with_features

TEAM_FIX = {"LA":"LAR","STL":"LAR","SD":"LAC","OAK":"LV"}
def _fix(s: pd.Series) -> pd.Series: return s.replace(TEAM_FIX)

# [] = consensus across all available books; ["draftkings"] = DK-only
BOOKS: list[str] = []   # start with consensus to maximize fill

def _load_schedule(cache: str) -> pd.DataFrame:
    p = os.path.join(cache, "schedule.csv")
    df = pd.read_csv(p, low_memory=False)
    df["home_team"] = _fix(df["home_team"]); df["away_team"] = _fix(df["away_team"])
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    return df

def _load_odds_raw(cache: str):
    p = os.path.join(cache, "odds_raw.json")
    if not os.path.exists(p): return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_models():
    win_art = os.path.join("cache","models","win_clf.pkl")
    ats_art = os.path.join("cache","models","ats_clf.pkl")
    win_pack = joblib.load(win_art) if os.path.exists(win_art) else None
    ats_pack = joblib.load(ats_art) if os.path.exists(ats_art) else None
    return win_pack, ats_pack

def _kelly_fraction(p: float | None, american_odds: float | None, cap=0.05) -> float:
    if p is None or american_odds is None or pd.isna(p) or pd.isna(american_odds):
        return 0.0
    ml = float(american_odds)
    b = (ml/100.0) if ml >= 0 else (100.0/(-ml))  # net odds
    f = (p*(b+1) - 1) / b
    return 0.0 if f <= 0 else min(float(f), cap)

def build_pick_sheet(cache: str = DATA_CACHE_DIR) -> pd.DataFrame:
    cache = cache or DATA_CACHE_DIR
    os.makedirs(cache, exist_ok=True)

    sched = _load_schedule(cache)

    # Upcoming window (dashboard shows future games)
    up = sched.copy()
    if "gameday" in up.columns:
        up = up[up["gameday"] >= pd.Timestamp.today().normalize()]

    # Features (uses full schedule as context)
    feats, feat_cols = build_upcoming_with_features(
        upcoming=up[["season","week","gameday","home_team","away_team","game_id"]],
        past_sched=sched
    )
    out = feats.copy()

    # Moneylines + (vig-removed) fair probs
    raw = _load_odds_raw(cache)
    if raw is not None:
        mls = extract_consensus_moneylines(raw, books=BOOKS if BOOKS else None)
        if not mls.empty:
            out = out.merge(mls, on=["home_team","away_team"], how="left")
        else:
            for c in ["home_ml","away_ml","home_prob","away_prob","home_prob_raw","away_prob_raw"]:
                out[c] = np.nan
    else:
        for c in ["home_ml","away_ml","home_prob","away_prob","home_prob_raw","away_prob_raw"]:
            out[c] = np.nan

    # Learned models (optional)
    win_pack, ats_pack = _load_models()
    if win_pack is not None and set(win_pack["feat_cols"]).issubset(out.columns):
        X = out[win_pack["feat_cols"]].fillna(0.0).to_numpy()
        p_home = win_pack["clf"].predict_proba(X)[:,1]
        out["home_prob_model"] = p_home
        out["away_prob_model"] = 1.0 - p_home

        # crude mapping to model spread
        logit = np.log(np.clip(p_home, 1e-6, 1-1e-6) / np.clip(1-p_home, 1e-6, 1))
        out["model_spread_home"] = (6.8 * logit).round(1)

        if "home_prob" in out.columns:
            out["home_edge"] = out["home_prob_model"] - out["home_prob"]
            out["away_edge"] = out["away_prob_model"] - out["away_prob"]

        out["home_kelly_5pct"] = out.apply(lambda r: _kelly_fraction(r.get("home_prob_model"), r.get("home_ml")), axis=1)
        out["away_kelly_5pct"] = out.apply(lambda r: _kelly_fraction(r.get("away_prob_model"), r.get("away_ml")), axis=1)

    # Column order
    front = ["season","week","gameday","home_team","away_team","game_id"]
    ml = ["home_ml","away_ml","home_prob","away_prob","home_prob_raw","away_prob_raw"]
    model = ["home_prob_model","away_prob_model","model_spread_home","home_edge","away_edge","home_kelly_5pct","away_kelly_5pct"]
    keep = [c for c in front + ml + model if c in out.columns] + [c for c in out.columns if c not in front + ml + model]
    out = out[keep]

    out_path = os.path.join(cache, "pick_sheet.csv")
    out.to_csv(out_path, index=False)
    print(f"[pick_sheet] wrote {out_path} ({len(out)} rows)")
    return out
