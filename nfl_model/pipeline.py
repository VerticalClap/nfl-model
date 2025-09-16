from .odds import extract_consensus_spreads

def build_pick_sheet(cache_dir: str):
    import json, pandas as pd
    sched = pd.read_csv(f"{cache_dir}/schedule.csv")
    raw = json.load(open(f"{cache_dir}/odds_raw.json", "r", encoding="utf-8"))

    # moneyline
    from .odds import extract_consensus_moneylines
    ml = extract_consensus_moneylines(raw, books=[])

    # spreads
    sp = extract_consensus_spreads(raw, books=[])

    # merge into schedule
    df = sched.merge(ml, on=["home_team", "away_team"], how="left")
    df = df.merge(sp, on=["home_team", "away_team"], how="left")

    # TODO: model spread calculation (placeholder for now)
    df["model_spread"] = (df["home_prob_model"] - 0.5) * -10  # crude example
    df["spread_edge"] = df["model_spread"] - df["home_spread"]

    df.to_csv(f"{cache_dir}/pick_sheet.csv", index=False)
    print(f"[pick_sheet] wrote {cache_dir}/pick_sheet.csv ({len(df)} rows)")
