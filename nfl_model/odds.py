# nfl_model/odds.py
from __future__ import annotations
import math
from typing import Iterable, Optional, Dict, Any, List
import pandas as pd
import numpy as np

# --- Utilities ---------------------------------------------------------------

_TEAM_FIX = {
    "LA": "LAR",   # old Rams code
    "STL": "LAR",
    "SD": "LAC",
    "OAK": "LV",
    "WSH": "WAS",
}
def normalize_team(t: Optional[str]) -> Optional[str]:
    if t is None:
        return None
    t = t.strip().upper()
    return _TEAM_FIX.get(t, t)

def american_to_prob(odds: float) -> Optional[float]:
    """Convert American odds to implied probability (0..1)."""
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    elif o < 0:
        return (-o) / ((-o) + 100.0)
    else:
        return None

def remove_vig(ph: Optional[float], pa: Optional[float]) -> (Optional[float], Optional[float]):
    """Rescale two implied probs to sum to 1 (simple no-vig)."""
    if ph is None or pa is None:
        return None, None
    s = ph + pa
    if s <= 0:
        return None, None
    return ph / s, pa / s

def _median_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not xs:
        return None
    return float(np.median(xs))

# --- Extraction helpers (Odds API v4 JSON) ----------------------------------

def _iter_markets(event: Dict[str, Any], want: str, books: Optional[Iterable[str]]) -> Iterable[Dict[str, Any]]:
    for bk in event.get("bookmakers", []):
        bkey = (bk.get("key") or "").lower()
        if books and bkey not in {x.lower() for x in books}:
            continue
        for m in bk.get("markets", []):
            if (m.get("key") or "").lower() == want.lower():
                yield m

def _event_teams(event: Dict[str, Any]) -> (Optional[str], Optional[str]):
    # v4 events: "home_team" and "away_team"
    return normalize_team(event.get("home_team")), normalize_team(event.get("away_team"))

# --- Moneylines --------------------------------------------------------------

def extract_consensus_moneylines(events: List[Dict[str, Any]], books: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Return median moneylines and (vig-removed) probs per game.
    Columns: home_team, away_team, home_ml, away_ml, home_prob_raw, away_prob_raw, home_prob, away_prob
    """
    rows = []
    for ev in events:
        ht, at = _event_teams(ev)
        if not ht or not at:
            continue

        home_prices, away_prices = [], []

        for m in _iter_markets(ev, "h2h", books):
            # outcomes: [{'name':'HOME','price':-120} , {'name':'AWAY','price':+100}] or team names
            for o in m.get("outcomes", []):
                name = (o.get("name") or "").upper()
                price = o.get("price")
                if price is None:
                    continue
                # Some feeds use literal team names instead of HOME/AWAY
                if name in ("HOME", ht):
                    home_prices.append(float(price))
                elif name in ("AWAY", at):
                    away_prices.append(float(price))

        home_ml = _median_or_none(home_prices)
        away_ml = _median_or_none(away_prices)

        home_prob_raw = american_to_prob(home_ml) if home_ml is not None else None
        away_prob_raw = american_to_prob(away_ml) if away_ml is not None else None
        home_prob, away_prob = remove_vig(home_prob_raw, away_prob_raw)

        rows.append({
            "home_team": ht, "away_team": at,
            "home_ml": home_ml, "away_ml": away_ml,
            "home_prob_raw": home_prob_raw, "away_prob_raw": away_prob_raw,
            "home_prob": home_prob, "away_prob": away_prob,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["home_team", "away_team"]).reset_index(drop=True)
    return df

# --- Spreads -----------------------------------------------------------------

def extract_spreads(events: List[Dict[str, Any]], books: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Return median point spread (home_line, points relative to home team; negative = favorite) and odds.
    Columns: home_team, away_team, home_line, home_spread_odds, away_spread_odds
    """
    rows = []
    for ev in events:
        ht, at = _event_teams(ev)
        if not ht or not at:
            continue

        home_lines, home_odds, away_odds = [], [], []

        for m in _iter_markets(ev, "spreads", books):
            # outcomes include 'point' and 'price'
            for o in m.get("outcomes", []):
                name = (o.get("name") or "").upper()
                point = o.get("point")
                price = o.get("price")
                if point is None or price is None:
                    continue

                # Normalize: home_line is relative to HOME (negative = home favored)
                if name in ("HOME", ht):
                    home_lines.append(float(point))
                    home_odds.append(float(price))
                elif name in ("AWAY", at):
                    # Away spread odds only (line is the same magnitude but opposite sign usually)
                    away_odds.append(float(price))

        row = {
            "home_team": ht, "away_team": at,
            "home_line": _median_or_none(home_lines),
            "home_spread_odds": _median_or_none(home_odds),
            "away_spread_odds": _median_or_none(away_odds),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["home_team", "away_team"]).reset_index(drop=True)
    return df
