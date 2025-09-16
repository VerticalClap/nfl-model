# scripts/fetch_and_build.py
from __future__ import annotations
import os
import json
import math
import requests
import pandas as pd
from datetime import timedelta

ODDS_BASE = "https://api.the-odds-api.com/v4"
BOOKS_PREFERRED = ["draftkings"]  # prefer DK if present

# --- helpers -----------------------------------------------------------------

def _ensure_cache() -> str:
    cache = os.environ.get("DATA_CACHE_DIR", "./cache")
    os.makedirs(cache, exist_ok=True)
    return cache

def _norm_team_codes(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    return s.replace({"LA": "LAR", "SD": "LAC", "OAK": "LV"})

def _american_to_prob(price: float) -> float:
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return float("nan")
    try:
        p = float(price)
    except Exception:
        return float("nan")
    if p > 0:
        return 100.0 / (p + 100.0)
    else:
        return (-p) / ((-p) + 100.0)

def _de_vig(p_home: float, p_away: float) -> tuple[float, float]:
    """Renormalize two implied probs to remove vig."""
    if any(map(lambda x: x is None or (isinstance(x, float) and math.isnan(x)), (p_home, p_away))):
        return float("nan"), float("nan")
    s = p_home + p_away
    if s <= 0 or not math.isfinite(s):
        return float("nan"), float("nan")
    return p_home / s, p_away / s

# --- schedule ----------------------------------------------------------------

def build_schedule_current(cache: str) -> pd.DataFrame:
    season = pd.Timestamp.today().year
    print(f"[schedule] building schedule for {season}")
    try:
        import nfl_data_py as nfl
        df = nfl.import_schedules([season])
    except Exception as e:
        raise RuntimeError(f"Could not import schedule via nfl_data_py: {e}")

    # pick a common set
    if "gameday" not in df.columns:
        for alt in ("game_date", "start_time", "start_time_utc"):
            if alt in df.columns:
                df = df.rename(columns={alt: "gameday"})
                break

    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    keep = [c for c in ["season","week","gameday","home_team","away_team","game_id"] if c in df.columns]
    df = df[keep].dropna(subset=["gameday","home_team","away_team"])
    df["home_team"] = _norm_team_codes(df["home_team"])
    df["away_team"] = _norm_team_codes(df["away_team"])

    # future games only is fine; doesn’t hurt if includes past ones
    df = df.sort_values(["gameday","home_team","away_team"]).reset_index(drop=True)

    out = os.path.join(cache, "schedule.csv")
    df.to_csv(out, index=False)
    print(f"[schedule] wrote {out} ({len(df)} rows)")
    return df

# --- odds fetch ---------------------------------------------------------------

def _fetch_odds(api_key: str, sport_key: str, markets: list[str], **params) -> list:
    if not api_key:
        raise RuntimeError("THE_ODDS_API_KEY not set")
    url = f"{ODDS_BASE}/sports/{sport_key}/odds"
    q = {
        "apiKey": api_key,
        "regions": "us,us2",
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    q.update(params)
    r = requests.get(url, params=q, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_spreads_and_h2h(api_key: str, sched: pd.DataFrame, cache: str) -> list:
    if sched.empty:
        return []
    # time window around schedule (to reduce noise)
    t0 = (sched["gameday"].min() - timedelta(days=2)).to_pydatetime().isoformat() + "Z"
    t1 = (sched["gameday"].max() + timedelta(days=9)).to_pydatetime().isoformat() + "Z"

    print("[odds] fetching spreads…")
    data_spreads = _fetch_odds(api_key, "americanfootball_nfl", ["spreads"],
                               commenceTimeFrom=t0, commenceTimeTo=t1)
    print(f"[odds] spreads events: {len(data_spreads)}")

    print("[odds] fetching moneylines…")
    data_h2h = _fetch_odds(api_key, "americanfootball_nfl", ["h2h"],
                           commenceTimeFrom=t0, commenceTimeTo=t1)
    print(f"[odds] h2h events: {len(data_h2h)}")

    # merge event lists by id (bookmakers vary per call)
    by_id = {}
    for ev in data_spreads + data_h2h:
        by_id.setdefault(ev["id"], {"id": ev["id"], "home_team": ev.get("home_team"), "away_team": ev.get("away_team"), "commence_time": ev.get("commence_time"), "bookmakers": {}})
        # merge bookmakers by key
        for bk in ev.get("bookmakers", []):
            by_id[ev["id"]]["bookmakers"].setdefault(bk["key"], {"key": bk["key"], "markets": {}})
            for m in bk.get("markets", []):
                by_id[ev["id"]]["bookmakers"][bk["key"]]["markets"][m["key"]] = m

    # dump raw for debugging
    with open(os.path.join(cache, "odds_raw.json"), "w", encoding="utf-8") as f:
        json.dump(list(by_id.values()), f, ensure_ascii=False, indent=2)
    print("[DEBUG] markets seen:", sorted({m
        for ev in by_id.values()
        for bk in ev["bookmakers"].values()
        for m in bk["markets"].keys()}))
    print(f"[odds] wrote {os.path.join(cache, 'odds_raw.json')}")
    return list(by_id.values())

# --- parsers ------------------------------------------------------------------

def _pick_bookmaker(bks: dict) -> dict | None:
    if not bks:
        return None
    # prefer DK, else any
    for name in BOOKS_PREFERRED:
        if name in bks:
            return bks[name]
    # otherwise first
    return next(iter(bks.values()))

def extract_spreads(events: list) -> pd.DataFrame:
    rows = []
    for ev in events:
        bk = _pick_bookmaker(ev.get("bookmakers", {}))
        if not bk: 
            continue
        m = bk["markets"].get("spreads")
        if not m:
            continue
        # outcomes contain team + point + price
        pts = {}
        for o in m.get("outcomes", []):
            team = o.get("name")
            if team == ev.get("home_team"):
                pts["home_line"] = o.get("point")
                pts["home_spread_odds"] = o.get("price")
            elif team == ev.get("away_team"):
                pts["away_line"] = o.get("point")
                pts["away_spread_odds"] = o.get("price")
        if "home_line" in pts:
            rows.append({
                "home_team": ev.get("home_team"),
                "away_team": ev.get("away_team"),
                "home_line": pts.get("home_line"),
                "home_spread_odds": pts.get("home_spread_odds"),
                "away_spread_odds": pts.get("away_spread_odds"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["home_team"] = _norm_team_codes(df["home_team"])
        df["away_team"] = _norm_team_codes(df["away_team"])
    return df

def extract_moneylines(events: list) -> pd.DataFrame:
    rows = []
    for ev in events:
        bk = _pick_bookmaker(ev.get("bookmakers", {}))
        if not bk:
            continue
        m = bk["markets"].get("h2h")
        if not m:
            continue
        prices = {}
        for o in m.get("outcomes", []):
            team = o.get("name")
            if team == ev.get("home_team"):
                prices["home_ml"] = o.get("price")
            elif team == ev.get("away_team"):
                prices["away_ml"] = o.get("price")
        if "home_ml" in prices and "away_ml" in prices:
            ph = _american_to_prob(prices["home_ml"])
            pa = _american_to_prob(prices["away_ml"])
            fh, fa = _de_vig(ph, pa)
            rows.append({
                "home_team": ev.get("home_team"),
                "away_team": ev.get("away_team"),
                "home_ml": prices["home_ml"],
                "away_ml": prices["away_ml"],
                "home_prob_raw": ph,
                "away_prob_raw": pa,
                "home_prob": fh,
                "away_prob": fa,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["home_team"] = _norm_team_codes(df["home_team"])
        df["away_team"] = _norm_team_codes(df["away_team"])
    return df

# --- build pick sheet ---------------------------------------------------------

def build_pick_sheet(cache: str) -> pd.DataFrame:
    api_key = os.environ.get("THE_ODDS_API_KEY", "").strip()
    cache = _ensure_cache()
    sched = build_schedule_current(cache)

    events = fetch_spreads_and_h2h(api_key, sched, cache)
    df_spreads = extract_spreads(events)
    df_ml = extract_moneylines(events)

    # left-join spreads and moneylines onto schedule
    merged = sched.merge(df_spreads, on=["home_team","away_team"], how="left") \
                  .merge(df_ml,      on=["home_team","away_team"], how="left")

    out = os.path.join(cache, "pick_sheet.csv")
    merged.to_csv(out, index=False)
    print(f"[pick_sheet] wrote {out} ({len(merged)} rows)")
    return merged

# --- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    build_pick_sheet(_ensure_cache())
