# scripts/build_model_outputs.py
import os
import pandas as pd

def build_model_outputs(cache: str):
    """
    Add baseline model outputs to pick_sheet.csv
    """
    sheet_path = os.path.join(cache, "pick_sheet.csv")
    df = pd.read_csv(sheet_path)

    # --- Baseline model (dummy math, replace later with real ML/statistics) ---
    # Model win probability: logistic transform of moneyline odds
    if {"home_ml", "away_ml"}.issubset(df.columns):
        df["model_home_prob"] = df["home_ml"].apply(
            lambda x: 1 / (1 + 10 ** (x / 400)) if pd.notna(x) else None
        )
        df["model_away_prob"] = 1 - df["model_home_prob"]

    # Model spread: derive a fake spread from difference in probs
    if {"model_home_prob"}.issubset(df.columns):
        df["model_spread"] = (df["model_home_prob"] - 0.5) * -14  # scale to NFL-like spreads

    out = os.path.join(cache, "pick_sheet.csv")
    df.to_csv(out, index=False)
    print(f"[model] wrote {out} with model columns: {', '.join([c for c in df.columns if c.startswith('model_')])}")
    return df

if __name__ == "__main__":
    cache = os.environ.get("DATA_CACHE_DIR", "./cache")
    build_model_outputs(cache)
