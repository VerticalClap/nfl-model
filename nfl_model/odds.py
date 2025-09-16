# nfl_model/odds.py
from __future__ import annotations
import math
import statistics as stats
import pandas as pd

# Full-name -> abbrev (covers Odds API team strings)
TEAM_ABBR = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
    "Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS",
}

def _american_to_prob(ml: float) -> float:
    """Convert American odds to implied probability (no vig removal)."""
    if ml is None or pd.isna(ml): 
        return float("nan")
    ml = float(ml)
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return abs(ml) / (abs(ml) + 100.0)

def _vig_free_pair(p_home: float, p_away: float) -> tuple[float, float]:
    """Remove vig by renormalizing both sides to sum to 1."""
    if any(pd.isna(x) for x in (p_home, p_away)):
        return float("nan"), float("nan")
    s = p_home + p_away
    if s == 0:
        return float("nan"), float("nan")
    return p_home / s, p_away / s

def _pick_market(book: dict, want_keys=("h2h","moneyline")) -> dict | None:
    """Pick the first moneyline market in this bookmaker object."""
    for m in book.get("markets", []):
        k = (m.get("key") or "").lower()
        if k in want_keys:
            return m
    return None

def _row_from_event(ev: dict, want_books: list[str] | None) -> dict | None:
    """Extract median (or DK) moneylines and vig-removed fair probs for one event."""
    home_name = ev.get("home_team")
    away_name = ev.get("away_team")
    # Odds API sometimes only gives full names; convert to abbr
    home_abbr = TEAM_ABBR.get(home_name, home_name)
    away_abbr = TEAM_ABBR.get(away_name, away_name)

    home_lines, away_lines = [], []

    for bk in ev.get("bookmakers", []):
        key = (bk.get("key") or "").lower()
        if want_books:  # if user asked for a subset (e.g. ['draftkings'])
            if key not in [b.lower() for b in want_books]:
                continue
        m = _pick_market(bk)
        if not m:
            continue
        # Moneyline outcomes should identify the team by name
        for out in m.get("outcomes", []):
            name = out.get("name")
            price = out.get("price")  # American odds (int/float)
            if name == home_name:
                home_lines.append(price)
            elif name == away_name:
                away_lines.append(price)

    # If we asked for 1 book (e.g. DK) and didn’t get both sides, bail for this game
    if want_books and (not home_lines or not away_lines):
        return None

    if not home_lines or not away_lines:
        # No lines available at all
        return None

    # Median across the chosen books (or the single DK entry)
    h_ml = stats.median([float(x) for x in home_lines])
    a_ml = stats.median([float(x) for x in away_lines])

    # Raw implied (pre-vig)
    p_home_raw = _american_to_prob(h_ml)
    p_away_raw = _american_to_prob(a_ml)

    # Vig-removed
    p_home, p_away = _vig_free_pair(p_home_raw, p_away_raw)

    return {
        "home_team": home_abbr, "away_team": away_abbr,
        "home_ml": h_ml, "away_ml": a_ml,
        "home_prob": p_home, "away_prob": p_away,
        "home_prob_raw": p_home_raw, "away_prob_raw": p_away_raw,
    }

def extract_consensus_moneylines(raw: list[dict], books: list[str] | None = None) -> pd.DataFrame:
    """
    From TheOddsAPI v4 payload, return a DF with:
    home_team, away_team, home_ml, away_ml, home_prob, away_prob, home_prob_raw, away_prob_raw

    - If `books=[]` or None → use all books (median).
    - If `books=['draftkings']` → force DraftKings only.
    """
    rows = []
    for ev in raw or []:
        r = _row_from_event(ev, books)
        if r:
            rows.append(r)
    if not rows:
        return pd.DataFrame(columns=[
            "home_team","away_team","home_ml","away_ml",
            "home_prob","away_prob","home_prob_raw","away_prob_raw"
        ])
    df = pd.DataFrame(rows)
    # Make sure abbreviations like LA/SD/OAK are normalized to LAR/LAC/LV just in case
    df["home_team"] = df["home_team"].replace({"LA":"LAR","SD":"LAC","OAK":"LV"})
    df["away_team"] = df["away_team"].replace({"LA":"LAR","SD":"LAC","OAK":"LV"})
    return df
