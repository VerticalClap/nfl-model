# nfl_model/odds.py
from __future__ import annotations
import math
import pandas as pd

def american_to_prob(ml: float | int) -> float:
    ml = float(ml)
    return (ml / (ml + 100.0)) if ml >= 0 else (100.0 / (100.0 - ml))

def prob_to_american(p: float) -> float:
    if p <= 0 or p >= 1: return float("nan")
    return 100.0 * p / (1 - p) if p < 0.5 else -100.0 * (1 - p) / p

def remove_vig(ph: float, pa: float) -> tuple[float,float]:
    s = ph + pa
    if s <= 0: return (float("nan"), float("nan"))
    return ph/s, pa/s

def _median_or_none(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if v is not None]
    if not vals: return None
    return float(pd.Series(vals).median())

def extract_moneylines(raw: list | dict) -> pd.DataFrame:
    # The Odds API v4 payload (events with bookmakers & markets)
    events = raw if isinstance(raw, list) else raw.get("data", [])
    rows = []
    for ev in events:
        ht, at = ev.get("home_team"), ev.get("away_team")
        if not ht or not at: continue
        home_prices, away_prices = [], []
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m.get("key") != "h2h": continue
                for o in m.get("outcomes", []):
                    team, price = o.get("name"), o.get("price")
                    if team in (ht, at) and price is not None:
                        if team == ht: home_prices.append(price)
                        else: away_prices.append(price)
        hml, aml = _median_or_none(home_prices), _median_or_none(away_prices)
        rows.append({"home_team": ht, "away_team": at, "home_ml": hml, "away_ml": aml})
    return pd.DataFrame(rows)

def extract_spreads(raw: list | dict) -> pd.DataFrame:
    events = raw if isinstance(raw, list) else raw.get("data", [])
    rows = []
    for ev in events:
        ht, at = ev.get("home_team"), ev.get("away_team")
        if not ht or not at: continue
        home_lines, away_lines, home_prices, away_prices = [], [], [], []
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m.get("key") != "spreads": continue
                for o in m.get("outcomes", []):
                    team, point, price = o.get("name"), o.get("point"), o.get("price")
                    if team == ht:
                        home_lines.append(point); home_prices.append(price)
                    elif team == at:
                        away_lines.append(point); away_prices.append(price)
        rows.append({
            "home_team": ht, "away_team": at,
            "home_spread": _median_or_none(home_lines),
            "away_spread": _median_or_none(away_lines),
            "home_spread_price": _median_or_none(home_prices),
            "away_spread_price": _median_or_none(away_prices),
        })
    return pd.DataFrame(rows)

def add_implied_probs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "home_ml" in out and "away_ml" in out:
        ph_raw = out["home_ml"].apply(lambda x: american_to_prob(x) if pd.notna(x) else float("nan"))
        pa_raw = out["away_ml"].apply(lambda x: american_to_prob(x) if pd.notna(x) else float("nan"))
        out["home_prob_raw"], out["away_prob_raw"] = ph_raw, pa_raw
        # fair (vig-removed)
        fair = [remove_vig(h, a) if pd.notna(h) and pd.notna(a) else (float("nan"), float("nan")) for h, a in zip(ph_raw, pa_raw)]
        out["home_prob"] = [f[0] for f in fair]
        out["away_prob"] = [f[1] for f in fair]
    return out
