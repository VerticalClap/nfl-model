# scripts/fetch_and_build.py
from __future__ import annotations
import os, json, math, statistics, requests
import pandas as pd
import nfl_data_py as nfl

ODDS_BASE = "https://api.the-odds-api.com/v4"

# ------------------------- helpers -------------------------

def ensure_cache() -> str:
    cache = os.environ.get("DATA_CACHE_DIR", "./cache")
    os.makedirs(cache, exist_ok=True)
    return cache

def american_to_prob(ml: float | int | None) -> float | None:
    """
    Convert American moneyline to implied probability (without vig removal).
    """
    if ml is None or (isinstance(ml, float) and math.isnan(ml)):
        return None
    ml = float(ml)
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return -ml / (-ml + 100.0)

def remove_vig_pair(p_home_raw: float | None, p_away_raw: float | None) -> tuple[float|None,float|None]:
    """
    Given raw implied probs for home and away, scale to sum to 1 (vig removal).
    """
    if p_home_raw is None or p_away_raw is None:
        return None, None
    s = p_home_raw + p_away_raw
    if s <= 0:
        return None, None
    return p_home_raw / s, p_away_raw / s

def norm_codes(s: pd.Series) -> pd.Series:
    """
    Normalize legacy team codes to current ones to improve joins.
    """
    return s.replace({"LA":"LAR", "STL":"LAR", "SD":"LAC", "OAK":"LV", "WSH":"WAS"})

# ------------------------- schedule -------------------------

def build_schedule_current_season(cache: str) -> pd.DataFrame:
    season = pd.Timestamp.today().year
    print(f"[schedule] building schedule for {season}")
    df = nfl.import_schedules([season])

    # pick a consistent gameday
    if "gameday" not in df.columns:
        for alt in ["game_date", "start_time"]:
            if alt in df.columns:
                df["gameday"] = pd.to_datetime(df[alt], errors="coerce")
                break
    else:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")

    # keep upcoming (today or later)
    today = pd.Timestamp.today().normalize()
    df = df[df["gameday"] >= today]

    keep = [c for c in ["season","week","gameday","home_team","away_team","game_id"] if c in df.columns]
    df = df[keep].sort_values(["week","gameday","home_team","away_team"]).reset_index(drop=True)
    out = os.path.join(cache, "schedule.csv")
    df.to_csv(out, index=False)
    print(f"[schedule] wrote {out} ({len(df)} rows)")
    return df

# ------------------------- odds fetching -------------------------

def fetch_odds_raw(api_key: str, markets: list[str]) -> list[dict]:
    """
    Fetch odds JSON for the given markets. We do NOT include time filters here
    (the Odds API will return upcoming events only by default).
    """
    params = {
        "apiKey": api_key,
        "regions": "us,us2",
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    url = f"{ODDS_BASE}/sports/americanfootball_nfl/odds"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# ------------------------- extraction -------------------------

def extract_moneylines(raw: list[dict], books: list[str] | None = None) -> pd.DataFrame:
    """
    Return one row per game with consensus (or specified books) moneylines
    plus vig-removed probabilities.
    """
    rows = []
    use_books = set(b.lower() for b in (books or []))
    for ev in raw:
        home, away = ev.get("home_team"), ev.get("away_team")
        home_quotes, away_quotes = [], []

        for bk in ev.get("bookmakers", []):
            bk_key = str(bk.get("key","")).lower()
            if use_books and bk_key not in use_books:
                continue
            for m in bk.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for o in m.get("outcomes", []):
                    name = o.get("name")
                    price = o.get("price")
                    if name == home:
                        home_quotes.append(price)
                    elif name == away:
                        away_quotes.append(price)

        if not home_quotes or not away_quotes:
            # no prices found for this event
            rows.append({
                "home_team": home, "away_team": away,
                "home_ml": None, "away_ml": None,
                "home_prob_raw": None, "away_prob_raw": None,
                "home_prob": None, "away_prob": None,
            })
            continue

        # use median across books
        home_ml = statistics.median(home_quotes)
        away_ml = statistics.median(away_quotes)

        p_home_raw = american_to_prob(home_ml)
        p_away_raw = american_to_prob(away_ml)
        p_home, p_away = remove_vig_pair(p_home_raw, p_away_raw)

        rows.append({
            "home_team": home, "away_team": away,
            "home_ml": home_ml, "away_ml": away_ml,
            "home_prob_raw": p_home_raw, "away_prob_raw": p_away_raw,
            "home_prob": p_home, "away_prob": p_away,
        })

    df = pd.DataFrame(rows)
    # normalize team codes
    df["home_team"] = norm_codes(df["home_team"].astype(str))
    df["away_team"] = norm_codes(df["away_team"].astype(str))
    return df

def extract_spreads(raw: list[dict], books: list[str] | None = None) -> pd.DataFrame:
    """
    Return one row per game with median home_line, and spread prices.
    Convention: home_line is the line for the home team (negative if favored).
    """
    rows = []
    use_books = set(b.lower() for b in (books or []))
    for ev in raw:
        home, away = ev.get("home_team"), ev.get("away_team")
        home_lines, home_prices, away_prices = [], [], []

        for bk in ev.get("bookmakers", []):
            bk_key = str(bk.get("key","")).lower()
            if use_books and bk_key not in use_books:
                continue
            for m in bk.get("markets", []):
                if m.get("key") != "spreads":
                    continue
                # outcomes usually contain two teams with 'point' and 'price'
                h_line, h_price, a_price = None, None, None
                for o in m.get("outcomes", []):
                    if o.get("name") == home:
                        h_line = o.get("point")
                        h_price = o.get("price")
                    elif o.get("name") == away:
                        a_price = o.get("price")
                if h_line is not None:
                    home_lines.append(h_line)
                if h_price is not None:
                    home_prices.append(h_price)
                if a_price is not None:
                    away_prices.append(a_price)

        if home_lines:
            row = {
                "home_team": home, "away_team": away,
                "home_line": statistics.median(home_lines),
                "home_spread_odds": statistics.median(home_prices) if home_prices else None,
                "away_spread_odds": statistics.median(away_prices) if away_prices else None,
            }
        else:
            row = {
                "home_team": home, "away_team": away,
                "home_line": None, "home_spread_odds": None, "away_spread_odds": None,
            }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["home_team"] = norm_codes(df["home_team"].astype(str))
    df["away_team"] = norm_codes(df["away_team"].astype(str))
    return df

# ------------------------- builder -------------------------

def build_pick_sheet(cache: str, books: list[str] | None = None) -> pd.DataFrame:
    """
    Build the pick sheet for the upcoming schedule:
      - schedule (cache/schedule.csv)
      - spreads + moneylines from The Odds API
      - merge and write cache/pick_sheet.csv
    """
    cache = ensure_cache()

    # 1) schedule (build or reuse)
    sched_path = os.path.join(cache, "schedule.csv")
    if os.path.exists(sched_path):
        schedule = pd.read_csv(sched_path, low_memory=False)
        # ensure types / normalization
        if "gameday" in schedule.columns:
            schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
    else:
        schedule = build_schedule_current_season(cache)

    schedule["home_team"] = norm_codes(schedule["home_team"].astype(str))
    schedule["away_team"] = norm_codes(schedule["away_team"].astype(str))

    # 2) odds
    api_key = os.environ.get("THE_ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("THE_ODDS_API_KEY is not set")

    raw = fetch_odds_raw(api_key, markets=["spreads", "h2h"])
    # quick debug
    markets_seen = sorted(
        {m.get("key") for ev in raw for bk in ev.get("bookmakers", []) for m in bk.get("markets", []) if m.get("key")}
    )
    print("[DEBUG] markets seen:", markets_seen)

    spreads = extract_spreads(raw, books=books)
    money = extract_moneylines(raw, books=books)

    # 3) merge
    keep_sched = [c for c in ["season","week","gameday","home_team","away_team","game_id"] if c in schedule.columns]
    base = schedule[keep_sched].copy()

    out = (base
           .merge(spreads, on=["home_team","away_team"], how="left")
           .merge(money,   on=["home_team","away_team"], how="left"))

    # 4) write
    out_path = os.path.join(cache, "pick_sheet.csv")
    out.to_csv(out_path, index=False)
    print(f"[pick_sheet] wrote {out_path} ({len(out)} rows)")
    return out

# ------------------------- entry point -------------------------

if __name__ == "__main__":
    cache = ensure_cache()
    # always (re)build schedule to keep it fresh
    build_schedule_current_season(cache)
    # then build the pick sheet
    build_pick_sheet(cache)
