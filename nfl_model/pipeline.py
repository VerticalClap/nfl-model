# nfl_model/pipeline.py
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import joblib

from .config import DATA_CACHE_DIR
from .odds import extract_moneylines, extract_spreads, extract_totals
from .features import build_upcoming_with_features

TEAM_FIX = {"LA":"LAR","STL":"LAR","SD":"LAC","OAK":"LV"}
def _fix(s: pd.Series) -> pd.Series: return s.replace(TEAM_FIX)

# Set to ["draftkings"] for DK-only; set to [] for median-of-books consensus.
BOOKS: list[str] = ["draftkings"]

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
    b = (ml/100.0) if ml >= 0 else (100.0/(-ml))
    f = (p*(b+1) - 1) / b
    return 0.0 if f <= 0 else min(float(f), cap)

def build_pick_sheet(cache: str = DATA_CACHE_DIR) -> pd.DataFrame:
    cache = cache or DATA_CACHE_DIR
    sched = _load_schedule(cache)

    # limit to today+ for dashboard
    up = sched.copy()
    if "gameday" in up.columns:
        up = up[up["gameday"] >= pd.Timestamp.today().normalize()]

    # features for upcoming
    feats, feat_cols = build_upcoming_with_features(
        upcoming=up[["season","week","gameday","home_team","away_team","game_id"]],
        past_sched=sched
    )
    out = feats.copy()

    # add book odds (ML, spreads, totals)
    raw = _load_odds_raw(cache)
    if raw is not None:
        mls = extract_moneylines(raw, bookmakers=BOOKS if BOOKS else None)
        spr = extract_spreads(raw,    bookmakers=BOOKS if BOOKS else None)
        tot = extract_totals(raw,     bookmakers=BOOKS if BOOKS else None)
        if not mls.empty: out = out.merge(mls, on=["home_team","away_team"], how="left")
        if not spr.empty: out = out.merge(spr, on=["home_team","away_team"], how="left")
        if not tot.empty: out = out.merge(tot, on=["home_team","away_team"], how="left")
    else:
        for c in ["home_ml","away_ml","home_prob","away_prob","home_prob_raw","away_prob_raw"]:
            out[c] = None

    # try learned models
    win_pack, ats_pack = _load_models()
    if win_pack is not None:
        X = out[win_pack["feat_cols"]].fillna(0.0).to_numpy()
        p_home = win_pack["clf"].predict_proba(X)[:,1]
        out["home_prob_model"] = p_home
        out["away_prob_model"] = 1.0 - p_home

        # crude mapping from prob → model spread (logit scale * constant)
        logit = np.log(np.clip(p_home, 1e-6, 1-1e-6) / np.clip(1-p_home, 1e-6, 1))
        C = 6.8  # typical NFL conversion factor (pts per logit)
        out["model_spread_home"] = (C * logit).round(1)

        # edges vs book fair probs (vig-removed)
        if "home_prob" in out.columns:
            out["home_edge"] = out["home_prob_model"] - out["home_prob"]
            out["away_edge"] = out["away_prob_model"] - out["away_prob"]

        # Kelly (moneyline)
        out["home_kelly_5pct"] = out.apply(lambda r: _kelly_fraction(r.get("home_prob_model"), r.get("home_ml")), axis=1)
        out["away_kelly_5pct"] = out.apply(lambda r: _kelly_fraction(r.get("away_prob_model"), r.get("away_ml")), axis=1)

    if ats_pack is not None and set(ats_pack["feat_cols"]).issubset(out.columns):
        X = out[ats_pack["feat_cols"]].fillna(0.0).to_numpy()
        out["home_cover_model"] = ats_pack["clf"].predict_proba(X)[:,1]
        # if book spread exists: show spread edge (model spread vs book)
        if "home_spread" in out.columns:
            out["spread_edge_pts"] = out["model_spread_home"] - out["home_spread"]

    # tidy columns
    order_front = ["season","week","gameday","home_team","away_team","game_id"]
    ml_cols = ["home_ml","away_ml","home_prob","away_prob","home_prob_raw","away_prob_raw"]
    sp_cols = ["home_spread","away_spread","home_spread_price","away_spread_price","home_cover_prob","away_cover_prob"]
    tot_cols= ["total_points","over_price","under_price","over_prob","under_prob"]
    model_cols = ["home_prob_model","away_prob_model","model_spread_home","home_cover_model","home_kelly_5pct","away_kelly_5pct","home_edge","away_edge","spread_edge_pts"]
    feat_keep = ["home_rest_days","away_rest_days","rest_delta","travel_km","travel_dir_km"]

    all_cols = order_front + ml_cols + sp_cols + tot_cols + model_cols + feat_keep
    cols = [c for c in all_cols if c in out.columns] + [c for c in out.columns if c not in all_cols]
    out = out[cols]

    out_path = os.path.join(cache, "pick_sheet.csv")
    out.to_csv(out_path, index=False)
    print(f"[pick_sheet] wrote {out_path} ({len(out)} rows)")
    return out
