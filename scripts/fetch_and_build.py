# scripts/fetch_and_build.py
import os, json, requests, pandas as pd

NFL_SCHED = "https://github.com/nflverse/nflfastR-data/raw/master/schedules/schedules.csv.gz"
ODDS_BASE = "https://api.the-odds-api.com/v4"

def ensure_cache():
    cache = os.environ.get("DATA_CACHE_DIR","./cache")
    os.makedirs(cache, exist_ok=True)
    return cache

def fetch_schedule(cache):
    p = os.path.join(cache, "schedules.csv.gz")
    if not os.path.exists(p):
        r = requests.get(NFL_SCHED, timeout=60)
        r.raise_for_status()
        with open(p, "wb") as f:
            f.write(r.content)
    df = pd.read_csv(p, compression="gzip", low_memory=False)
    # keep current + future seasons only (optional)
    df = df[df["season"] >= 2025].copy()
    out = os.path.join(cache, "schedule.csv")
    df.to_csv(out, index=False)
    print(f"[schedule] wrote {out} ({len(df)} rows)")
    return df

def fetch_odds(api_key, markets="h2h,spreads", regions="us"):
    url = f"{ODDS_BASE}/sports/americanfootball_nfl/odds"
    r = requests.get(
        url,
        params={
            "apiKey": api_key,
            "regions": regions,           # us books
            "markets": markets,           # <-- include spreads
            "oddsFormat": "american",
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    cache = ensure_cache()
    fetch_schedule(cache)

    key = os.environ.get("THE_ODDS_API_KEY")
    if key:
        try:
            data = fetch_odds(key, markets="h2h,spreads")  # ensure spreads requested
            with open(os.path.join(cache, "odds_raw.json"), "w", encoding="utf-8") as f:
                json.dump(data, f)
            print("[odds] wrote cache/odds_raw.json")
        except Exception as e:
            print("[odds] fetch failed:", e)
    else:
        print("[odds] skipped: THE_ODDS_API_KEY not set")
