import numpy as np
import pandas as pd

def moneyline_to_prob(ml: float) -> float:
    """
    Convert American moneyline to implied win probability (vig not removed).
    """
    ml = float(ml)
    if ml >= 0:
        return 100.0 / (ml + 100.0)
    return (-ml) / ((-ml) + 100.0)

def baseline_probs_from_odds(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given an odds table with home/away moneylines, compute implied probs.
    Expected columns (if present): home_team, away_team, home_ml, away_ml
    Returns a dataframe with columns: home_prob, away_prob
    """
    df = odds_df.copy()
    if "home_ml" in df and "away_ml" in df:
        df["home_prob_raw"] = df["home_ml"].apply(moneyline_to_prob)
        df["away_prob_raw"] = df["away_ml"].apply(moneyline_to_prob)
        # Simple de-vig: normalize so the two probs sum to 1
        s = df["home_prob_raw"] + df["away_prob_raw"]
        df["home_prob"] = df["home_prob_raw"] / s
        df["away_prob"] = df["away_prob_raw"] / s
    else:
        # If moneylines aren’t present yet, return empty columns
        df["home_prob"] = np.nan
        df["away_prob"] = np.nan
    return df[["home_prob", "away_prob"]]
