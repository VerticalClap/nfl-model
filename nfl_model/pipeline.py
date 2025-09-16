# --- BEGIN REPLACEMENT BLOCK (put near top of pipeline.py) ---
from __future__ import annotations
import os, json, pandas as pd
from nfl_model.rest_travel import add_rest_travel
from nfl_model.features import add_basic_features
from nfl_model.modeling import model_probs_from_features
from nfl_model.odds import extract_consensus_moneylines  # we’ll pass books=['draftkings']

# Force DraftKings only.  To use consensus across all available books, set BOOKS = [].
BOOKS = ["draftkings"]

def build_pick_sheet(cache_dir: str = "./cache") -> pd.DataFrame:
    cache_dir = os.path.abspath(cache_dir)
    sched_p = os.path.join(cache_dir, "schedule.csv")
    odds_p  = os.path.join(cache_dir, "odds_raw.json")

    sched = pd.read_csv(sched_p, low_memory=False)

    # Normalize team codes in schedule so they match Odds API (HOU/TB/LV/LAC/LAR etc.)
    sched["home_team"] = sched["home_team"].replace({"LA": "LAR", "SD": "LAC", "OAK": "LV"})
    sched["away_team"] = sched["away_team"].replace({"LA": "LAR", "SD": "LAC", "OAK": "LV"})

    # Add rest/travel + any basic features you’ve built
    sched = add_rest_travel(sched)
    sched = add_basic_features(sched)

    # Load odds and extract DK moneylines + fair probs
    if os.path.exists(odds_p):
        with open(odds_p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        odds = extract_consensus_moneylines(raw, books=BOOKS)  # DraftKings only
    else:
        odds = pd.DataFrame(columns=["home_team","away_team","home_ml","away_ml","home_prob","away_prob","home_prob_raw","away_prob_raw"])

    # Merge by (home_team, away_team). We deliberately do not include date to keep it simple.
    merged = pd.merge(
        sched,
        odds,
        on=["home_team","away_team"],
        how="left",
        suffixes=("","_odds")
    )

    # Compute model probabilities (from your features)
    model = model_probs_from_features(merged)
    merged["home_prob_model"] = model["home_prob_model"]
    merged["away_prob_model"] = 1.0 - merged["home_prob_model"]

    # Kelly at 5% fraction only when market probs exist
    def kelly(p, b):
        # b is net odds (decimal-1), but for moneyline we can derive with American odds
        return max(p - (1-p)/b, 0) if (b is not None and b != 0) else 0

    def american_to_decimal(ml):
        if pd.isna(ml):
            return None
        ml = float(ml)
        if ml > 0:
            return 1 + ml/100.0
        else:
            return 1 + 100.0/abs(ml)

    merged["home_kelly_5pct"] = 0.0
    merged["away_kelly_5pct"] = 0.0
    # only compute Kelly if we actually have a moneyline
    for i, r in merged.iterrows():
        if pd.notna(r.get("home_ml")) and pd.notna(r.get("away_ml")):
            dh = american_to_decimal(r["home_ml"])
            da = american_to_decimal(r["away_ml"])
            if dh and da:
                b_h = dh - 1.0
                b_a = da - 1.0
                merged.at[i, "home_kelly_5pct"] = 0.05 * kelly(r["home_prob_model"], b_h)
                merged.at[i, "away_kelly_5pct"] = 0.05 * kelly(1.0 - r["home_prob_model"], b_a)

    out_cols = [
        "season","week","gameday","home_team","away_team","game_id",
        "home_ml","away_ml","home_prob","away_prob","home_prob_raw","away_prob_raw",
        "home_prob_model","away_prob_model",
        "home_kelly_5pct","away_kelly_5pct",
        # include any rest/travel columns you want visible:
        "home_rest_days","away_rest_days","home_travel_miles","away_travel_miles"
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    out = merged[out_cols].copy()

    out_path = os.path.join(cache_dir, "pick_sheet.csv")
    out.to_csv(out_path, index=False)
    print(f"[pick_sheet] wrote {out_path} ({len(out)} rows)")
    return out
# --- END REPLACEMENT BLOCK ---
