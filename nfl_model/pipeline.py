# nfl_model/pipeline.py
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
    Build pick_sheet.csv with at least the columns the Streamlit app expects:
      - home_ml, away_ml (consensus)
      - home_prob_raw, away_prob_raw (no-vig before normalization)
      - home_prob, away_prob (vig-removed and normalized to 1.0)
      - home_prob_model, away_prob_model (fallback model == market for now)
      - home_kelly_5pct, away_kelly_5pct (0 placeholders for now)

    Returns: Path to the written pick_sheet.csv
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) schedule
    sched_path = cache_dir / "schedule.csv"
    if not sched_path.exists():
        raise FileNotFoundError(f"Missing {sched_path}. Run scripts/fetch_and_build.py first.")
    sched = pd.read_csv(sched_path, low_memory=False)
    sched = _standardize_teams(sched)
    # Keep just the essentials the app filters on
    keep = [c for c in ["season", "week", "gameday", "home_team", "away_team", "game_id"] if c in sched.columns]
    sched_small = sched[keep].copy()

    # 2) odds
    odds_file = cache_dir / "odds_raw.json"
    if odds_file.exists():
        raw = json.load(odds_file.open("r", encoding="utf-8"))
        # books=[] means consensus across all available books
        odds = extract_consensus_moneylines(raw, books=[])
    else:
        # No odds yet → create empty columns so the app still loads
        odds = pd.DataFrame(columns=[
            "home_team", "away_team", "home_ml", "away_ml",
            "home_prob", "away_prob", "home_prob_raw", "away_prob_raw"
        ])

    odds = _standardize_teams(odds)

    # 3) merge schedule + odds
    merged = pd.merge(
        sched_small, odds,
        on=["home_team", "away_team"],
        how="left",
        suffixes=("", "_odds")
    )

    # 4) ensure model columns exist (fallback == market)
    if "home_prob" in merged.columns:
        merged["home_prob_model"] = merged["home_prob"]
        merged["away_prob_model"] = 1.0 - merged["home_prob_model"]
    else:
        merged["home_prob_model"] = pd.NA
        merged["away_prob_model"] = pd.NA

    # 5) placeholder Kelly (0 = no bet) so Streamlit can render
    if "home_kelly_5pct" not in merged.columns:
        merged["home_kelly_5pct"] = 0.0
    if "away_kelly_5pct" not in merged.columns:
        merged["away_kelly_5pct"] = 0.0

    # 6) write
    out_path = cache_dir / "pick_sheet.csv"
    merged.to_csv(out_path, index=False)
    print(f"[pick_sheet] wrote {out_path} ({len(merged)} rows)")
    return out_path
