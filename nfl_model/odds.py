# nfl_model/odds.py
import math
import re
import pandas as pd
from statistics import median

# Map bookmaker team names -> nfl_data_py abbreviations
TEAM_MAP = {
    # AFC East
    "buffalo bills": "BUF",
    "miami dolphins": "MIA",
    "new england patriots": "NE",
    "ny jets": "NYJ", "new york jets": "NYJ",
    # AFC North
    "baltimore ravens": "BAL",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "pittsburgh steelers": "PIT",
    # AFC South
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX", "jacksonville jags": "JAX",
    "tennessee titans": "TEN",
    # AFC West
    "denver broncos": "DEN",
    "kansas city chiefs": "KC", "kansas city": "KC",
    "las vegas raiders": "LV", "oakland raiders": "LV",
    "la chargers": "LAC", "los angeles chargers": "LAC", "san diego chargers": "LAC",
    # NFC East
    "dallas cowboys": "DAL",
    "ny giants": "NYG", "new york giants": "NYG",
    "philadelphia eagles": "PHI",
    "washington commanders": "WAS", "washington football team": "WAS", "washington redskins": "WAS", "washington": "WAS",
    # NFC North
    "chicago bears": "CHI",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "minnesota vikings": "MIN",
    # NFC South
    "atlanta falcons": "ATL",
    "carolina panthers": "CAR",
    "new orleans saints": "NO", "new orleans": "NO",
    "tampa bay buccaneers": "TB", "tampa bay": "TB", "tampa bay bcs": "TB",
    # NFC West
    "arizona cardinals": "ARI",
    "la rams": "LAR", "los angeles rams": "LAR", "st. louis rams": "LAR",
    "san francisco 49ers": "SF", "san francisco": "SF",
    "seattle seahawks": "SEA",
}

# Some books omit city; try mascot-only fallbacks
MASCOT_FALLBACK = {
    "patriots": "NE", "jets": "NYJ", "bills": "BUF", "dolphins": "MIA",
    "ravens": "BAL", "bengals": "CIN", "browns": "CLE", "steelers": "PIT",
    "texans": "HOU", "colts": "IND", "jaguars": "JAX", "jags": "JAX", "titans": "TEN",
    "broncos": "DEN", "chiefs": "KC", "raiders": "LV", "chargers": "LAC",
    "cowboys": "DAL", "giants": "NYG", "eagles": "PHI", "commanders": "WAS",
    "bears": "CHI", "lions": "DET", "packers": "GB", "vikings": "MIN",
    "falcons": "ATL", "panthers": "CAR", "saints": "NO", "buccaneers": "TB", "bucs": "TB",
    "cardinals": "ARI", "rams": "LAR", "49ers": "SF", "niners": "SF", "seahawks": "SEA",
}

def _clean(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s]", "", s)  # drop punctuation
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_team(name: str) -> str | None:
    if not name:
        return None
    s = _clean(name)
    # direct map
    if s in TEAM_MAP:
        return TEAM_MAP[s]
    # try mascot last token
    parts = s.split()
    if parts:
        m = parts[-1]
        if m in MASCOT_FALLBACK:
            return MASCOT_FALLBACK[m]
    # try without words like "the"
    s2 = s.replace("the ", "")
    if s2 in TEAM_MAP:
        return TEAM_MAP[s2]
    return None  # unknown

def moneyline_to_prob(ml):
    if ml is None:
        return None
    try:
        ml = float(ml)
    except Exception:
        return None
    return 100.0/(ml+100.0) if ml >= 0 else (-ml)/((-ml)+100.0)

def prob_to_moneyline(p):
    if p is None or p <= 0 or p >= 1:
        return None
    return -round(100*p/(1-p)) if p >= 0.5 else round(100*(1-p)/p)

def remove_vig_two_way(p_home, p_away):
    if p_home is None or p_away is None:
        return (None, None)
    s = p_home + p_away
    if not s or s <= 0:
        return (None, None)
    return (p_home/s, p_away/s)

def extract_consensus_moneylines(odds_raw: list) -> pd.DataFrame:
    """
    Convert The Odds API response into median moneylines per game,
    normalize team names to nfl_data_py abbreviations for reliable merging.
    """
    rows = []
    for game in odds_raw or []:
        home_raw = game.get("home_team")
        away_raw = game.get("away_team")
        home = normalize_team(home_raw)
        away = normalize_team(away_raw)
        if not home or not away:
            # skip games we can't map
            continue

        home_prices, away_prices = [], []
        for b in game.get("bookmakers", []):
            for mk in b.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                for o in mk.get("outcomes", []):
                    nm, price = o.get("name"), o.get("price")
                    team_abbr = normalize_team(nm)
                    if team_abbr == home and price is not None:
                        home_prices.append(price)
                    elif team_abbr == away and price is not None:
                        away_prices.append(price)

        home_ml = median(home_prices) if home_prices else None
        away_ml = median(away_prices) if away_prices else None
        ph_raw = moneyline_to_prob(home_ml)
        pa_raw = moneyline_to_prob(away_ml)
        ph_fair, pa_fair = remove_vig_two_way(ph_raw, pa_raw)

        rows.append({
            "home_team": home, "away_team": away,
            "home_ml": home_ml, "away_ml": away_ml,
            "home_prob": ph_fair, "away_prob": pa_fair,
        })

    return pd.DataFrame(rows)

# Back-compat for pipeline
def extract_moneylines(raw): 
    return extract_consensus_moneylines(raw)

def add_implied_probs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "home_ml" in out:
        out["home_prob_raw"] = out["home_ml"].apply(moneyline_to_prob)
    if "away_ml" in out:
        out["away_prob_raw"] = out["away_ml"].apply(moneyline_to_prob)
    if "home_prob_raw" in out and "away_prob_raw" in out:
        fair = out.apply(
            lambda r: pd.Series(remove_vig_two_way(r["home_prob_raw"], r["away_prob_raw"]),
                                index=["home_prob", "away_prob"]),
            axis=1
        )
        out[["home_prob", "away_prob"]] = fair
    return out
