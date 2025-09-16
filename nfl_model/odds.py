# nfl_model/odds.py
from __future__ import annotations
import math
from typing import Iterable, List, Dict, Any
import pandas as pd
import numpy as np

TEAM_FIX = {
    "LA": "LAR", "STL": "LAR",
    "SD": "LAC",
    "OAK": "LV",
}

def _norm_team(t: str) -> str:
    if not isinstance(t, str):
        return t
    t = t.strip().upper()
    return TEAM_FIX.get(t, t)

def american_to_prob(odds: float | int | None) -> float | None:
    if odds is None or (isinstance(odds, float) and (math.isnan(odds))):
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return -o / (-o + 100.0)

def _choose_consensus(values: List[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not vals:
        return None
    return float(np.median(vals))

def _extract_event_teams(ev: Dict[str, Any]) -> tuple[str | None, str | None]:
    # Odds API puts team names in 'home_team' & 'away_team' (strings)
    ht = ev.get("home_team")
    at = ev.get("away_team")
    if isinstance(ht, str) and isinstance(at, str):
        return _norm_team(ht), _norm_team(at)
    # Fallback: infer from first bookmaker outcomes (rarely needed)
    for bk in ev.get("bookmakers", []):
        for m in bk.get("markets", []):
            for out in m.get("outcomes", []):
                name = out.get("name")
                if name:
                    # often full team names (e.g. "New York Jets") — keep upper token code if present
                    pass
    return None, None

def _book_priority(order_first: Iterable[str], book_key: str) -> int:
    # prefer DraftKings if present; otherwise use median across all
    order = list(order_first)
    if book_key in order:
        return order.index(book_key)
    # tie-breaker: keep others after
    return len(order) + 1

def extract_consensus_moneylines(raw: List[Dict[str, Any]], prefer_books: Iterable[str] = ("draftkings",)) -> pd.DataFrame:
    rows = []
    for ev in raw:
        home, away = _extract_event_teams(ev)
        if not home or not away:
            continue

        home_prices, away_prices = [], []
        first_pick = None

        # Prefer a specific book (e.g., DraftKings) if available; else use median across books
        # We’ll scan all books; remember any exact match from preferred list.
        preferred_seen = False
        pref_home = pref_away = None

        for bk in sorted(ev.get("bookmakers", []), key=lambda b: _book_priority(prefer_books, b.get("key",""))):
            for m in bk.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                # Odds API outcomes can be names matching team strings
                outs = m.get("outcomes", [])
                if not outs:
                    continue
                h = a = None
                for o in outs:
                    nm = _norm_team(o.get("name",""))
                    price = o.get("price")
                    if nm == home:
                        h = price
                    elif nm == away:
                        a = price
                if h is not None and a is not None:
                    if not preferred_seen and bk.get("key") in prefer_books:
                        pref_home, pref_away = h, a
                        preferred_seen = True
                    home_prices.append(h)
                    away_prices.append(a)

        if preferred_seen:
            home_ml, away_ml = float(pref_home), float(pref_away)
        else:
            home_ml, away_ml = _choose_consensus(home_prices), _choose_consensus(away_prices)

        rows.append({
            "home_team": home,
            "away_team": away,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_prob": american_to_prob(home_ml),
            "away_prob": american_to_prob(away_ml),
        })

    df = pd.DataFrame(rows)
    return df

def extract_consensus_spreads(raw: List[Dict[str, Any]], prefer_books: Iterable[str] = ("draftkings",)) -> pd.DataFrame:
    """
    Return one row per matchup:
      spread (home_line): negative means home favored by X points
      home_spread_odds / away_spread_odds: American prices for that line
    We prefer the line from a preferred book if available; otherwise median across books
    at their closest-to-zero home line (ties -> median).
    """
    rows = []
    for ev in raw:
        home, away = _extract_event_teams(ev)
        if not home or not away:
            continue

        # Collect lines per book
        per_book = []  # (book_key, home_line, home_price, away_price)
        for bk in sorted(ev.get("bookmakers", []), key=lambda b: _book_priority(prefer_books, b.get("key",""))):
            best = None
            for m in bk.get("markets", []):
                if m.get("key") != "spreads":
                    continue
                for out in m.get("outcomes", []):
                    # Each outcome has: name (team), point (float), price (int)
                    nm = _norm_team(out.get("name", ""))
                    pt = out.get("point", None)
                    price = out.get("price", None)
                    if pt is None or price is None:
                        continue
                    # We’ll assemble a pair (home_line, home_price, away_price)
                    if nm == home:
                        # start a tuple with home side
                        if best is None:
                            best = {"home_line": float(pt), "home_price": float(price), "away_price": None}
                        else:
                            best["home_line"] = float(pt)
                            best["home_price"] = float(price)
                    elif nm == away:
                        if best is None:
                            best = {"home_line": None, "home_price": None, "away_price": float(price)}
                        else:
                            best["away_price"] = float(price)
            if best and best.get("home_line") is not None and best.get("home_price") is not None and best.get("away_price") is not None:
                per_book.append((bk.get("key"), best["home_line"], best["home_price"], best["away_price"]))

        if not per_book:
            continue

        # Prefer first item if preferred book present (sorted earlier puts it first)
        # Otherwise compute medians across books using lines closest to zero (most comparable)
        # First, pick the cluster of lines closest to zero:
        lines = [abs(x[1]) for x in per_book]
        min_abs = min(lines)
        cluster = [x for x in per_book if abs(x[1]) == min_abs]

        # If the very first item was from a preferred book, use that; else median across cluster
        if per_book and per_book[0][0] in prefer_books:
            _, line, hp, ap = per_book[0]
            home_line = float(line)
            home_spread_odds = float(hp)
            away_spread_odds = float(ap)
        else:
            home_line = _choose_consensus([x[1] for x in cluster])
            home_spread_odds = _choose_consensus([x[2] for x in cluster])
            away_spread_odds = _choose_consensus([x[3] for x in cluster])

        rows.append({
            "home_team": home,
            "away_team": away,
            "home_line": home_line,                    # e.g. -3.5 means home favored
            "home_spread_odds": home_spread_odds,
            "away_spread_odds": away_spread_odds,
        })

    return pd.DataFrame(rows)
