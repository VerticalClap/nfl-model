import math
import pandas as pd

def moneyline_to_prob(ml: float | int | None) -> float | None:
    if ml is None:
        return None
    ml = float(ml)
    if ml >= 0:
        return 100.0 / (ml + 100.0)
    return (-ml) / ((-ml) + 100.0)

def normalize_two_way(p_home: float | None, p_away: float | None) -> tuple[float | None, float | None]:
    if p_home is None or p_away is None:
        return (None, None)
    s = p_home + p_away
    if s <= 0:
        return (None, None)
    return (p_home / s, p_away / s)

def extract_moneylines(odds_raw: list) -> pd.DataFrame:
    """
    Flattens The Odds API JSON to one row per game with home/away ML (best available).
    """
    rows = []
    for game in odds_raw or []:
        home = game.get("home_team")
        away = game.get("away_team")
        best_home_ml = None
        best_away_ml = None
        for b in game.get("bookmakers", []):
            for mk in b.get("markets", []):
                if mk.get("key") == "h2h":
                    for o in mk.get("outcomes", []):
                        name = o.get("name")
                        price = o.get("price")
                        if name == home:
                            # choose *best* ML from multiple books (most favorable for the team)
                            if best_home_ml is None:
                                best_home_ml = price
                            else:
                                # for favorites (negative), larger (closer to zero) is better; for dogs (positive), larger is better
                                if (price < 0 and price > best_home_ml) or (price >= 0 and price > best_home_ml):
                                    best_home_ml = price
                        if name == away:
                            if best_away_ml is None:
                                best_away_ml = price
                            else:
                                if (price < 0 and price > best_away_ml) or (price >= 0 and price > best_away_ml):
                                    best_away_ml = price
        rows.append({"home_team": home, "away_team": away, "home_ml": best_home_ml, "away_ml": best_away_ml})
    return pd.DataFrame(rows)

def add_implied_probs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["home_prob_raw"] = out["home_ml"].apply(moneyline_to_prob)
    out["away_prob_raw"] = out["away_ml"].apply(moneyline_to_prob)
    norm = out.apply(
        lambda r: normalize_two_way(r["home_prob_raw"], r["away_prob_raw"]),
        axis=1, result_type="expand"
    )
    out["home_prob"], out["away_prob"] = norm[0], norm[1]
    return out
