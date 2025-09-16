# nfl_model/pipeline.py
from __future__ import annotations
import os, json, pandas as pd
from .features import build_upcoming_with_features
from .odds import extract_consensus_moneylines
from .utils import ensure_dir

CACHE = os.environ.get("DATA_CACHE_DIR", "./cache")

def _load_upcoming() -> pd.DataFrame:
    """Load the schedule CSV (downloaded by scripts/fetch_and_build.py)."""
    p = os.path.join(CACHE, "schedule.csv")
    return pd.read_csv(p, low_memory=False)

def _load_history() -> pd.DataFrame:
    """
    For now we reuse the same season schedule as 'history' (placeholder).
    In a later step we'll hydrate multi-season history for training.
    """
    p = os.path.join(CACHE, "schedule.csv")
    return pd.read_csv(p, low_memory=False)

def _load_odds_df() -> pd.DataFrame:
    """Turn odds_raw.json into a tidy DF of consensus moneylines + implied probs."""
    p = os.path.join(CACHE, "odds_raw.json")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["home_team","away_team","home_ml","away_ml","home_prob","away_prob"])
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return extract_consensus_moneylines(raw)

def build_pick_sheet(cache_dir: str | None = None) -> str:
    """Create cache/pick_sheet.csv with features + odds columns."""
    if cache_dir:
        global CACHE
        CACHE = cache_dir

    ensure_dir(CACHE)

    upcoming = _load_upcoming()
    history  = _load_history()
    odds     = _load_odds_df()

    # === New: add rest/travel features ===
    feats, feat_cols = build_upcoming_with_features(upcoming, history)

    # Merge odds by (home_team, away_team)
    out = feats.merge(odds, on=["home_team","away_team"], how="left")

    # Order/clean columns
    cols_front = ["season","week","gameday","home_team","away_team"]
    metric_cols = ["home_ml","away_ml","home_prob","away_prob"] + feat_cols
    other_cols = [c for c in out.columns if c not in cols_front + metric_cols]
    final_cols = cols_front + metric_cols + other_cols
    out = out[final_cols]

    # Write
    out_path = os.path.join(CACHE, "pick_sheet.csv")
    out.to_csv(out_path, index=False)
    print(f"[pick_sheet] wrote {out_path} ({len(out)} rows)")
    return out_path
