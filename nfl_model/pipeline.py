# nfl_model/pipeline.py  (also paste as top-level pipeline.py if you have one)
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from .odds import extract_consensus_moneylines  # builds home_ml/away_ml + prob + vig-removed

TEAM_FIX = {"LA": "LAR", "SD": "LAC", "OAK": "LV"}


def _standardize_teams(df: pd.DataFrame, cols=("home_team", "away_team")) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].replace(TEAM_FIX)
    return out


def build_pick_sheet(cache_dir: str | Path = "./cache") -> Path:
    """
    Build pick_sheet.csv with at least the columns the app expects:
      - home_ml, away_ml (consensus)
      - home_prob_raw, away_prob_raw (pre-normalization, no-vig)
      - home_prob, away_prob (vig-removed, normalized)
      - home_prob_model, away_prob_model (fallback == market until model is added)
      - home_kelly_5pct, away_kelly_5pct (placeholder 0)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) schedule
    sched_path = cache_dir / "schedule.csv"
    if not sched_path.exists():
        raise FileNotFoundError(f"Missing {sched_path}. Run scripts/fetch_and_build.py first.")
    sched = pd.read_csv(sched_path, low_memory=False)
    sched = _standardize_teams(sched)

    keep = [c for c in ["season", "week", "gameday", "home_team", "away_team", "game_id"] if c in sched.columns]
    sched_small = sched[keep].copy()

    # 2) odds (consensus across all books)
    odds_path = cache_dir / "odds_raw.json"
    if odds_path.exists():
        raw = json.load(odds_path.open("r", encoding="utf-8"))
        odds = extract_consensus_moneylines(raw, books=[])  # [] => all books consensus
    else:
        odds = pd.DataFrame(columns=[
            "home_team","away_team","home_ml","away_ml",
            "home_prob","away_prob","home_prob_raw","away_prob_raw"
        ])

    odds = _standardize_teams(odds)

    # 3) merge
    merged = pd.merge(
        sched_small, odds,
        on=["home_team", "away_team"],
        how="left",
        suffixes=("", "_odds"),
    )

    # 4) ensure model columns (fallback to market)
    if "home_prob" in merged.columns:
        merged["home_prob_model"] = merged["home_prob"]
        merged["away_prob_model"] = 1.0 - merged["home_prob_model"]
    else:
        merged["home_prob_model"] = pd.NA
        merged["away_prob_model"] = pd.NA

    # 5) placeholder Kelly columns
    if "home_kelly_5pct" not in merged.columns:
        merged["home_kelly_5pct"] = 0.0
    if "away_kelly_5pct" not in merged.columns:
        merged["away_kelly_5pct"] = 0.0

    # 6) write
    out_path = cache_dir / "pick_sheet.csv"
    merged.to_csv(out_path, index=False)
    print(f"[pick_sheet] wrote {out_path} ({len(merged)} rows)")
    return out_path
