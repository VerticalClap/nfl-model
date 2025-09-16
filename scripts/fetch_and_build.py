# scripts/fetch_and_build.py
from __future__ import annotations
import os, json, statistics, requests
import pandas as pd
import nfl_data_py as nfl

ODDS_BASE = "https://api.the-odds-api.com/v4"

def ensure_cache() -> str:
    cache = os.environ.get("DATA_CACHE_DIR", "./cache")
    os.makedirs(cache, exist_ok=True)
    return cache

def american_to_prob(ml):
    if ml is None or pd.isna(ml): return None
    ml = float(ml)
    return 100.0/(ml+100.0) if ml>0 else -ml/(-ml+100.0)

def remove_vig_pair(p_home_raw, p_away_raw):
    if p_home_raw is None or p_away_raw is None: return None, None
    s = p_home_raw + p_away_raw
    if not s or s <= 0: return None, None
    return p_home_raw/s, p_away_raw/s

def norm_codes(s: pd.Series) -> pd.Series:
    return s.astype(str).replace({"LA":"LAR","STL":"LAR","SD":"LAC","OAK":"LV","WSH":"WAS"})

# ---------- schedule ----------
def build_schedule_current_season(cache: str) -> pd.DataFrame:
    season = pd.Timestamp.today().year
    print(f"[schedule] building schedule for {season}")
    df = nfl.import_schedules([season])
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    else:
        for alt in ["game_date","start_time"]:
            if alt in df.columns:
                df["gameday"] = pd.to_datetime(df[alt], errors="coerce")
                break
    today = pd.Timestamp.today().normalize()
    df = df[df["gameday"] >= today]
    keep = [c for c in ["season","week","gameday","home_team","away_team","game_id"] if c in df.columns]
    df = df[keep].sort_values(["week","gameday","home_team","away_team"]).reset_index(drop=True)
    out = os.path.join(cache, "schedule.csv")
    df.to_csv(out, index=False)
    print(f"[schedule] wrote {out} ({len(df)} rows)")
    return df

# ---------- odds fetch ----------
def fetch_odds_raw(api_key: str) -> list[dict]:
    url = f"{ODDS_BASE}/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": api_key,
        "regions": "us,us2",
        "markets": "spreads,h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Debug: what markets did we actually get?
    markets = sorted({m.get("key")
                      for ev in data
                      for bk in ev.get("bookmakers",[])
                      for m in bk.get("markets",[])
                      if m.get("key")})
    print("[DEBUG] markets seen:", markets)
    return data

# ---------- extractors ----------
def extract_moneylines(raw: list[dict], books: list[str] | None = None) -> pd.DataFrame:
    use = set(b.lower() for b in (books or []))
    rows = []
    for ev in raw:
        home, away = ev.get("home_team"), ev.get("away_team")
        hq, aq = [], []
        for bk in ev.get("bookmakers", []):
            if use and bk.get("key","").lower() not in use: continue
            for m in bk.get("markets", []):
                if m.get("key") != "h2h": continue
                for o in m.get("outcomes", []):
                    nm, price = o.get("name"), o.get("price")
                    if nm == home: hq.append(price)
                    elif nm == away: aq.append(price)
        if not hq or not aq:
            rows.append({"home_team":home,"away_team":away,
                         "home_ml":None,"away_ml":None,
                         "home_prob_raw":None,"away_prob_raw":None,
                         "home_prob":None,"away_prob":None})
            continue
        hml = statistics.median(hq); aml = statistics.median(aq)
        ph_raw = american_to_prob(hml); pa_raw = american_to_prob(aml)
        ph, pa = remove_vig_pair(ph_raw, pa_raw)
        rows.append({"home_team":home,"away_team":away,
                     "home_ml":hml,"away_ml":aml,
                     "home_prob_raw":ph_raw,"away_prob_raw":pa_raw,
                     "home_prob":ph,"away_prob":pa})
    df = pd.DataFrame(rows)
    df["home_team"] = norm_codes(df["home_team"])
    df["away_team"] = norm_codes(df["away_team"])
    return df

def extract_spreads(raw: list[dict], books: list[str] | None = None) -> pd.DataFrame:
    use = set(b.lower() for b in (books or []))
    rows = []
    for ev in raw:
        home, away = ev.get("home_team"), ev.get("away_team")
        home_lines, home_prices, away_prices = [], [], []
        for bk in ev.get("bookmakers", []):
            if use and bk.get("key","").lower() not in use: continue
            for m in bk.get("markets", []):
                if m.get("key") != "spreads": continue
                h_line = h_price = a_price = None
                for o in m.get("outcomes", []):
                    nm = o.get("name")
                    if nm == home:
                        h_line  = o.get("point")
                        h_price = o.get("price")
                    elif nm == away:
                        a_price = o.get("price")
                if h_line is not None: home_lines.append(h_line)
                if h_price is not None: home_prices.append(h_price)
                if a_price is not None: away_prices.append(a_price)
        if home_lines:
            rows.append({"home_team":home,"away_team":away,
                         "home_line":statistics.median(home_lines),
                         "home_spread_odds":statistics.median(home_prices) if home_prices else None,
                         "away_spread_odds":statistics.median(away_prices) if away_prices else None})
        else:
            rows.append({"home_team":home,"away_team":away,
                         "home_line":None,"home_spread_odds":None,"away_spread_odds":None})
    df = pd.DataFrame(rows)
    df["home_team"] = norm_codes(df["home_team"])
    df["away_team"] = norm_codes(df["away_team"])
    return df

# ---------- builder ----------
def build_pick_sheet(cache: str, books: list[str] | None = None) -> pd.DataFrame:
    cache = ensure_cache()
    # schedule
    sched_path = os.path.join(cache, "schedule.csv")
    if os.path.exists(sched_path):
        schedule = pd.read_csv(sched_path, low_memory=False)
        if "gameday" in schedule.columns:
            schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
    else:
        schedule = build_schedule_current_season(cache)
    schedule["home_team"] = norm_codes(schedule["home_team"])
    schedule["away_team"] = norm_codes(schedule["away_team"])

    # odds
    key = os.environ.get("THE_ODDS_API_KEY","").strip()
    if not key:
        raise RuntimeError("THE_ODDS_API_KEY is not set")
    raw = fetch_odds_raw(key)

    spreads = extract_spreads(raw, books=books)
    money   = extract_moneylines(raw, books=books)

    base_cols = [c for c in ["season","week","gameday","home_team","away_team","game_id"] if c in schedule.columns]
    base = schedule[base_cols].copy()

    out = (base
           .merge(spreads, on=["home_team","away_team"], how="left")
           .merge(money,   on=["home_team","away_team"], how="left"))

    out_path = os.path.join(cache, "pick_sheet.csv")
    out.to_csv(out_path, index=False)
    print(f"[pick_sheet] wrote {out_path} ({len(out)} rows)")
    return out

if __name__ == "__main__":
    cache = ensure_cache()
    build_schedule_current_season(cache)
    build_pick_sheet(cache)
