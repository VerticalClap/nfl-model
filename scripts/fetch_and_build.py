# scripts/fetch_and_build.py
import os, json, requests, pandas as pd
import nfl_data_py as nfl

ODDS_BASE = "https://api.the-odds-api.com/v4"
BOOKS = []   # [] => consensus (median across books). Or e.g. ["draftkings"].

def ensure_cache() -> str:
    cache = os.environ.get("DATA_CACHE_DIR", "./cache")
    os.makedirs(cache, exist_ok=True)
    return cache

# ---------- moneyline helpers ----------
def ml_to_prob(ml: float) -> float | None:
    if ml is None or pd.isna(ml): return None
    try: ml = float(ml)
    except Exception: return None
    if ml > 0:  return 100.0 / (ml + 100.0)
    else:       return (-ml) / ((-ml) + 100.0)

def fair_probs_from_mls(home_ml, away_ml):
    hp, ap = ml_to_prob(home_ml), ml_to_prob(away_ml)
    if hp is None or ap is None: return None, None, None, None
    s = hp + ap
    if s <= 0: return None, None, None, None
    return hp/s, ap/s, hp, ap  # (vig-removed), (raw)

# ---------- schedule ----------
def fetch_schedule_upcoming(cache: str) -> pd.DataFrame:
    season = pd.Timestamp.today().year
    print(f"[schedule] building schedule for {season}")
    df = nfl.import_schedules([season])
    if "gameday" not in df.columns:
        for alt in ["game_date", "start_time", "gamedate"]:
            if alt in df.columns:
                df["gameday"] = pd.to_datetime(df[alt], errors="coerce")
                break
    else:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    df = df[df["gameday"] >= today]
    keep = [c for c in ["season","week","gameday","home_team","away_team","game_id"] if c in df.columns]
    df = df[keep].sort_values(["week","gameday","home_team","away_team"]).reset_index(drop=True)
    out = os.path.join(cache, "schedule.csv")
    df.to_csv(out, index=False)
    print(f"[schedule] wrote {out} ({len(df)} rows)")
    return df

# ---------- odds fetch ----------
def fetch_odds(api_key: str, markets: list[str]) -> list:
    url = (
        f"{ODDS_BASE}/sports/americanfootball_nfl/odds"
        f"?apiKey={api_key}&regions=us,us2&markets={','.join(markets)}"
        f"&oddsFormat=american&dateFormat=iso"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def _consensus_markets(bookmakers: list[dict]) -> list[dict]:
    mkmap: dict[str, list[dict]] = {}
    for bk in bookmakers:
        for m in bk.get("markets", []):
            mkmap.setdefault(m.get("key"), []).append(m)
    out = []
    for mkey, mlist in mkmap.items():
        agg: dict[str, dict[str, list[float]]] = {}
        for m in mlist:
            for o in m.get("outcomes", []):
                name = o.get("name")
                if not name: continue
                a = agg.setdefault(name, {"price": [], "point": []})
                if o.get("price") is not None: a["price"].append(float(o["price"]))
                if o.get("point") is not None: a["point"].append(float(o["point"]))
        outcomes = []
        for name, vals in agg.items():
            oo = {"name": name}
            if vals["price"]: oo["price"] = float(pd.Series(vals["price"]).median())
            if vals["point"]: oo["point"] = float(pd.Series(vals["point"]).median())
            outcomes.append(oo)
        out.append({"key": mkey, "outcomes": outcomes})
    return out

def _pick_book(bookmakers, prefer):
    if not bookmakers: return None
    if not prefer:
        return {"key":"consensus", "markets": _consensus_markets(bookmakers)}
    keys = [bk.get("key") for bk in bookmakers]
    for want in prefer:
        if want in keys:
            return bookmakers[keys.index(want)]
    return bookmakers[0]

def extract_moneylines(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for ev in raw:
        home, away = ev.get("home_team"), ev.get("away_team")
        bk = _pick_book(ev.get("bookmakers", []), BOOKS)
        if not bk: continue
        m = next((x for x in bk.get("markets", []) if x.get("key")=="h2h"), None)
        if not m: continue
        hm = am = None
        for o in m.get("outcomes", []):
            if o.get("name") in (home, "Home"): hm = o.get("price")
            if o.get("name") in (away, "Away"): am = o.get("price")
        rows.append({"home_team":home, "away_team":away, "home_ml":hm, "away_ml":am})
    return pd.DataFrame(rows)

def extract_spreads(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for ev in raw:
        home, away = ev.get("home_team"), ev.get("away_team")
        bk = _pick_book(ev.get("bookmakers", []), BOOKS)
        if not bk: continue
        m = next((x for x in bk.get("markets", []) if x.get("key")=="spreads"), None)
        if not m: continue
        home_line = home_odds = away_odds = None
        for o in m.get("outcomes", []):
            if o.get("name") in (home, "Home"):
                home_line = o.get("point")
                home_odds = o.get("price")
            if o.get("name") in (away, "Away"):
                away_odds = o.get("price")
        rows.append({"home_team":home, "away_team":away,
                     "home_line":home_line, "home_spread_odds":home_odds, "away_spread_odds":away_odds})
    return pd.DataFrame(rows)

# ---------- pipeline ----------
def build_pick_sheet(cache: str):
    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        raise SystemExit("THE_ODDS_API_KEY is not set.")

    cache = ensure_cache()
    sched = fetch_schedule_upcoming(cache)

    raw_spreads = fetch_odds(api_key, ["spreads"])
    print(f"[odds] spreads events: {len(raw_spreads)}")
    raw_h2h = fetch_odds(api_key, ["h2h"])
    print(f"[odds] h2h events: {len(raw_h2h)}")

    markets_seen = sorted({m.get("key")
                           for ev in (raw_spreads + raw_h2h)
                           for bk in ev.get("bookmakers", [])
                           for m in bk.get("markets", []) if m.get("key")})
    print("[DEBUG] markets seen:", markets_seen)

    o_ml  = extract_moneylines(raw_h2h)
    o_sp  = extract_spreads(raw_spreads)

    rep = {"LA":"LAR","SD":"LAC","OAK":"LV"}
    for col in ["home_team","away_team"]:
        if col in sched.columns: sched[col] = sched[col].replace(rep)
        if col in o_ml.columns:  o_ml[col]  = o_ml[col].replace(rep)
        if col in o_sp.columns:  o_sp[col]  = o_sp[col].replace(rep)

    out = sched.merge(o_ml, on=["home_team","away_team"], how="left") \
               .merge(o_sp, on=["home_team","away_team"], how="left")

    for i, r in out.iterrows():
        hp, ap, hraw, araw = fair_probs_from_mls(r.get("home_ml"), r.get("away_ml"))
        out.at[i, "home_prob"] = hp
        out.at[i, "away_prob"] = ap
        out.at[i, "home_prob_raw"] = hraw
        out.at[i, "away_prob_raw"] = araw

    pick = os.path.join(cache, "pick_sheet.csv")
    out.to_csv(pick, index=False)
    print(f"[pick_sheet] wrote {pick} ({len(out)} rows)")

if __name__ == "__main__":
    build_pick_sheet(ensure_cache())
