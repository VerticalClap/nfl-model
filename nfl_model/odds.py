# nfl_model/odds.py
from __future__ import annotations
import pandas as pd
import math

# ---------- American odds helpers ----------
def american_to_prob(ml: float | int) -> float:
    ml = float(ml)
    return (ml / (ml + 100.0)) if ml >= 0 else (100.0 / (100.0 - ml))

def prob_to_american(p: float) -> float:
    if pd.isna(p) or p <= 0 or p >= 1:
        return float("nan")
    return 100.0 * p / (1 - p) if p < 0.5 else -100.0 * (1 - p) / p

def remove_vig(p_a: float, p_b: float) -> tuple[float,float]:
    """Return fair probs that sum to 1.0 given two raw (vigged) probs."""
    if pd.isna(p_a) or pd.isna(p_b):
        return (float("nan"), float("nan"))
    s = p_a + p_b
    if s <= 0:
        return (float("nan"), float("nan"))
    return (p_a / s, p_b / s)

def _median_or_none(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if v is not None and not pd.isna(v)]
    if not vals:
        return None
    return float(pd.Series(vals).median())

def _iter_events(raw):
    if isinstance(raw, list):
        return raw
    return raw.get("data", [])

# ---------- MONEYLINES ----------
def extract_moneylines(raw: list | dict, bookmakers: list[str] | None = None) -> pd.DataFrame:
    """
    Returns columns: home_team, away_team, home_ml, away_ml,
                     home_prob_raw, away_prob_raw, home_prob, away_prob
    If 'bookmakers' is provided (e.g., ['draftkings']), only those books are used.
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
                    if team == ht:
                        home_prices.append(price)
                    elif team == at:
                        away_prices.append(price)
        hml = _median_or_none(home_prices)
        aml = _median_or_none(away_prices)

        # implied raw probabilities from American odds
        ph_raw = american_to_prob(hml) if hml is not None else float("nan")
        pa_raw = american_to_prob(aml) if aml is not None else float("nan")
        ph, pa = remove_vig(ph_raw, pa_raw)

        rows.append({
            "home_team": ht, "away_team": at,
            "home_ml": hml, "away_ml": aml,
            "home_prob_raw": ph_raw, "away_prob_raw": pa_raw,
            "home_prob": ph, "away_prob": pa,
        })
    return pd.DataFrame(rows)

# ---------- SPREADS ----------
def extract_spreads(raw: list | dict, bookmakers: list[str] | None = None) -> pd.DataFrame:
    """
    Returns columns: home_team, away_team,
                     home_spread, away_spread,
                     home_spread_price, away_spread_price,
                     home_cover_prob, away_cover_prob  (vig-removed from prices)
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

        # convert spread prices -> implied cover probs (vig-removed)
        ph_raw = american_to_prob(hp) if hp is not None else float("nan")
        pa_raw = american_to_prob(ap) if ap is not None else float("nan")
        ph, pa = remove_vig(ph_raw, pa_raw)

        rows.append({
            "home_team": ht, "away_team": at,
            "home_spread": hs, "away_spread": as_,
            "home_spread_price": hp, "away_spread_price": ap,
            "home_cover_prob": ph, "away_cover_prob": pa,
        })
    return pd.DataFrame(rows)

# ---------- TOTALS (over/under) ----------
def extract_totals(raw: list | dict, bookmakers: list[str] | None = None) -> pd.DataFrame:
    """
    Returns columns: home_team, away_team,
                     total_points,
                     over_price, under_price,
                     over_prob, under_prob  (vig-removed)
    """
    rows = []
    for ev in _iter_events(raw):
        ht, at = ev.get("home_team"), ev.get("away_team")
        if not ht or not at:
            continue
        totals, over_prices, under_prices = [], [], []
        for bk in ev.get("bookmakers", []):
            if bookmakers and (bk.get("key") not in bookmakers):
                continue
            for m in bk.get("markets", []):
                if m.get("key") != "totals":
                    continue
                for o in m.get("outcomes", []):
                    name, point, price = o.get("name"), o.get("point"), o.get("price")
                    if point is not None:
                        totals.append(point)
                    if name and "over" in name.lower():
                        over_prices.append(price)
                    elif name and "under" in name.lower():
                        under_prices.append(price)

        tline = _median_or_none(totals)
        op = _median_or_none(over_prices)
        up = _median_or_none(under_prices)

        po_raw = american_to_prob(op) if op is not None else float("nan")
        pu_raw = american_to_prob(up) if up is not None else float("nan")
        po, pu = remove_vig(po_raw, pu_raw)

        rows.append({
            "home_team": ht, "away_team": at,
            "total_points": tline,
            "over_price": op, "under_price": up,
            "over_prob": po, "under_prob": pu,
        })
    return pd.DataFrame(rows)
