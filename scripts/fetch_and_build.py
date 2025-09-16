# scripts/fetch_and_build.py
import os, json, requests, pandas as pd
import nfl_data_py as nfl

ODDS_BASE = "https://api.the-odds-api.com/v4"

def ensure_cache() -> str:
    cache = os.environ.get("DATA_CACHE_DIR", "./cache")
    os.makedirs(cache, exist_ok=True)
    return cache

def fetch_schedule_current_season(cache: str) -> pd.DataFrame:
    season = pd.Timestamp.today().year
    print(f"[schedule] building schedule for {season}")
    df = nfl.import_schedules([season])

    # normalize columns we need
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")

    # keep current & future only
    today = pd.Timestamp.today().normalize()
    if "gameday" in df.columns:
        df = df[df["gameday"] >= today]

    keep = [c for c in ["season","week","gameday","home_team","away_team","game_id"] if c in df.columns]
    df = df[keep].sort_values(["week","gameday","home_team","away_team"]).reset_index(drop=True)

    out = os.path.join(cache, "schedule.csv")
    df.to_csv(out, index=False)
    print(f"[schedule] wrote {out} ({len(df)} rows)")
    return df

def _fetch_odds(api_key: str, markets: str, regions: str = "us", bookmakers: str | None = None) -> list:
    """Minimal, API-friendly call: no time filters, only 'us', selectable markets."""
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,          # e.g. "spreads" or "h2h"
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers

    url = f"{ODDS_BASE}/sports/americanfootball_nfl/odds"
    r = requests.get(url, params=params, timeout=60)
    if r.status_code >= 400:
        # bubble up error text so we can see what's wrong
        raise requests.HTTPError(f"{r.status_code} {r.reason}: {r.url}\n{r.text}")
    return r.json()

def merge_markets(base: list, extra: list) -> list:
    """Merge 'extra' markets into 'base' by event id."""
    # Build index by (id or event id); fall back to index
    def _eid(ev, i):
        return ev.get("id") or ev.get("event_id") or f"idx:{i}"

    merged = { _eid(ev, i): ev for i, ev in enumerate(base) }
    for j, ev in enumerate(extra):
        key = _eid(ev, j)
        if key in merged:
            dst = merged[key]
            dst_bks = dst.setdefault("bookmakers", [])
            for bk in ev.get("bookmakers", []):
                # attach/extend markets per bookmaker
                found = False
                for db in dst_bks:
                    if db.get("key") == bk.get("key"):
                        db.setdefault("markets", [])
                        db["markets"].extend(bk.get("markets", []))
                        found = True
                        break
                if not found:
                    dst_bks.append(bk)
        else:
            merged[key] = ev
    return list(merged.values())

if __name__ == "__main__":
    cache = ensure_cache()
    fetch_schedule_current_season(cache)

    key = os.environ.get("THE_ODDS_API_KEY")
    if not key:
        print("[odds] skipped: THE_ODDS_API_KEY not set")
    else:
        try:
            # two simple passes, then merge
            print("[odds] fetching spreads…")
            spreads = _fetch_odds(key, markets="spreads", regions="us", bookmakers=None)
            print(f"[odds] spreads events: {len(spreads)}")

            print("[odds] fetching moneylines…")
            h2h = _fetch_odds(key, markets="h2h", regions="us", bookmakers=None)
            print(f"[odds] h2h events: {len(h2h)}")

            data = merge_markets(h2h, spreads)

            # quick debug: list markets seen
            seen = set()
            for ev in data:
                for bk in ev.get("bookmakers", []):
                    for m in bk.get("markets", []):
                        if m.get("key"):
                            seen.add(m["key"])
            print("[DEBUG] markets seen:", sorted(seen))

            out = os.path.join(cache, "odds_raw.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[odds] wrote {out}")
        except Exception as e:
            print("[odds] fetch failed:", e)
