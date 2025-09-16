# nfl_model/pipeline.py
from __future__ import annotations
import os, json, pandas as pd
from nfl_model.odds import extract_consensus_moneylines, extract_consensus_spreads

def build_pick_sheet(cache_dir: str = "./cache", books: list[str] | None = None) -> pd.DataFrame:
    sch_path = os.path.join(cache_dir, "schedule.csv")
    odds_path = os.path.join(cache_dir, "odds_raw.json")

    sched = pd.read_csv(sch_path, low_memory=False)
    if "gameday" in sched.columns:
        sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")

    # Load odds
    with open(odds_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    ml = extract_consensus_moneylines(raw, books=books or [])
    sp = extract_consensus_spreads(raw, books=books or [])

    # Merge moneylines
    out = sched.merge(ml, on=["home_team","away_team"], how="left")

    # Merge spreads
    out = out.merge(sp, on=["home_team","away_team"], how="left")

    # Optional model placeholders (kept for dashboard safety)
    for c in ["home_prob_model", "away_prob_model", "model_spread", "edge_points", "edge_pct"]:
        if c not in out.columns:
            out[c] = None

    out = out.sort_values(["week","gameday","home_team","away_team"]).reset_index(drop=True)
    out_path = os.path.join(cache_dir, "pick_sheet.csv")
    out.to_csv(out_path, index=False)
    print(f"[pick_sheet] wrote {out_path} ({len(out)} rows)")
    return out
