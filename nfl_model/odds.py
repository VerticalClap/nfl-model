# nfl_model/odds.py
from __future__ import annotations
import pandas as pd
import math

# ----- American odds helpers -----
def american_to_prob(ml: float | int) -> float:
    ml = float(ml)
    return (ml / (ml + 100.0)) if ml >= 0 else (100.0 / (100.0 - ml))

def prob_to_american(p: float) -> float:
    if p <= 0 or p >= 1 or pd.isna(p): 
        return float("nan")
    return 100.0 * p / (1 - p) if p < 0.5 else -100.0 * (1 - p) / p

def remove_vig(p_home: float, p_away: float) -> tuple[float,float]:
    s = p_home + p_away
    if s <= 0 or pd.isna(p_home) or pd.isna(p_away):
        return (float("nan"), float("nan"))
    return p_home / s, p_away / s

def _median_or_none(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if v is not None and not pd.isna(v)]
    if not vals:
        return None
    return float(pd.Series(vals).median())

# ----- Extractors from The Odds API v4 -----
def _iter_events(raw):
    if isinstance(raw, list):
        return raw
    return raw.get("data", [])

def extract_moneylines(raw: list | dict, bookmakers: list[str] | None = None) -> pd.DataFrame:
    """
    Returns: home_team, away_team, home_ml, away_ml
    If 'bookmakers' is provided (e.g. ['draftkings']), only those books are used.
    """
    rows = []
    for ev in _iter_events(raw):
        ht, at = ev.get("home_team"), ev.get("away_team")
        if not ht or not at: 
            continue
        home_prices, away_prices = [], []
        for bk in ev.get("bookmakers", []):
            if bookmakers and (bk.get("key") not in bookmakers):
                continue
            for m in bk.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for o in m.get("outcomes", []):
                    team, price = o.get("name"), o.get("price")
                    if team == ht: home_prices.append(price)
                    elif team == at: away_prices.append(price)
        rows.append({
            "home_team": ht, "away_team": at,
            "home_ml": _median_or_none(home_prices),
            "away_ml": _median_or_none(away_prices),
        })
    return pd.DataFrame(rows)

def extract_spreads(raw: list | dict, bookmakers: list[str] | None = None) -> pd.DataFrame:
    """
    Returns: home_team, away_team, home_spread, away_spread, home_spread_price, away_spread_price,
             home_cover_prob, away_cover_prob (vig-removed from prices if both present)
    """
    rows = []
    for ev in _iter_events(raw):
        ht, at = ev.get("home_team"), ev.get("away_team")
        if not ht or not at: 
            continue
        home_lines, away_lines, home_prices, away_prices = [], [], [], []
        for bk in ev.get("bookmakers", []):
            if bookmakers and (bk.get("key") not in bookmakers):
                continue
            for m in bk.get("markets", []):
                if m.get("key") != "spreads":
                    continue
                for o in m.get("outcomes", []):
                    team, point, price = o.get("name"), o.get("point"), o.get("price")
                    if team == ht:
                        home_lines.append(point); home_prices.append(price)
                    elif team == at:
                        away_lines.append(point); away_prices.append(price)

        hs = _median_or_none(home_lines)
        as_ = _median_or_none(away_lines)
        hp = _median_or_none(home_prices)
        ap = _median_or_none(away_prices)

        # Convert spread prices to implied cover probabilities and de-vig
        ph_raw = american_to_prob(hp) if hp is not None else float("nan")
        pa_raw = american_to_prob(ap) if ap is not None else float("nan")
        ph, pa = remove_vig(ph_raw, pa_raw) if not (pd.isna(ph_raw) or pd.isna(pa_raw)) else (float("nan"), float("nan"))

        rows.append({
            "home_team": ht, "away_team": at,
            "home_spread": hs, "away_spread": as_,
            "home_spread_price": hp, "away_spread_price": ap,
            "home_cover_prob": ph, "away_cover_prob": pa,
        })
    return pd.DataFrame(rows)

# ----- Enrich moneyline with implied & fair probs -----
def add_implied_probs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "home_ml" in out and "away_ml" in out:
        ph_raw = out["home_ml"].apply(lambda x: american_to_prob(x) if pd.notna(x) else float("nan"))
        pa_raw = out["away_ml"].apply(lambda x: american_to_prob(x) if pd.notna(x) else float("nan"))
        out["home_prob_raw"], out["away_prob_raw"] = ph_raw, pa_raw
        fair = [remove_vig(h, a) if pd.notna(h) and pd.notna(a) else (float("nan"), float("nan")) for h, a in zip(ph_raw, pa_raw)]
        out["home_prob"] = [f[0] for f in fair]
        out["away_prob"] = [f[1] for f in fair]
    return out
