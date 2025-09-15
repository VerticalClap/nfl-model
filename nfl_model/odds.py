import math
import pandas as pd
from statistics import median

def moneyline_to_prob(ml):
    if ml is None or (isinstance(ml, float) and math.isnan(ml)): return None
    ml = float(ml)
    return 100.0/(ml+100.0) if ml >= 0 else (-ml)/((-ml)+100.0)

def prob_to_moneyline(p):
    if p is None or p <= 0 or p >= 1: return None
    return -round(100*p/(1-p)) if p >= 0.5 else round(100*(1-p)/p)

def remove_vig_two_way(p_home, p_away):
    if p_home is None or p_away is None: return (None, None)
    s = p_home + p_away
    if not s or s <= 0: return (None, None)
    return (p_home/s, p_away/s)

def extract_consensus_moneylines(odds_raw: list) -> pd.DataFrame:
    rows = []
    for game in odds_raw or []:
        home = game.get("home_team")
        away = game.get("away_team")
        home_prices, away_prices = [], []
        for b in game.get("bookmakers", []):
            for mk in b.get("markets", []):
                if mk.get("key") != "h2h": continue
                for o in mk.get("outcomes", []):
                    nm, price = o.get("name"), o.get("price")
                    if nm == home and price is not None: home_prices.append(price)
                    if nm == away and price is not None: away_prices.append(price)
        home_ml = median(home_prices) if home_prices else None
        away_ml = median(away_prices) if away_prices else None
        ph_raw = moneyline_to_prob(home_ml)
        pa_raw = moneyline_to_prob(away_ml)
        ph_fair, pa_fair = remove_vig_two_way(ph_raw, pa_raw)
        rows.append({
            "home_team": home, "away_team": away,
            "home_ml": home_ml, "away_ml": away_ml,
            "home_prob": ph_fair, "away_prob": pa_fair
        })
    return pd.DataFrame(rows)

# Back-compat names used by pipeline
def extract_moneylines(raw): return extract_consensus_moneylines(raw)

def add_implied_probs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "home_ml" in out: out["home_prob_raw"] = out["home_ml"].apply(moneyline_to_prob)
    if "away_ml" in out: out["away_prob_raw"] = out["away_ml"].apply(moneyline_to_prob)
    if "home_prob_raw" in out and "away_prob_raw" in out:
        out[["home_prob","away_prob"]] = out.apply(
            lambda r: pd.Series(remove_vig_two_way(r["home_prob_raw"], r["away_prob_raw"])),
            axis=1
        )
    return out
