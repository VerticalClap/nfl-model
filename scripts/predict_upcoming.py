import os, pandas as pd, requests

NFL_SCHED = "https://github.com/nflverse/nflfastR-data/raw/master/schedules/schedules.csv.gz"
ODDS_BASE = "https://api.the-odds-api.com/v4"

def ensure_cache():
    cache = os.environ.get("DATA_CACHE_DIR","./cache")
    os.makedirs(cache, exist_ok=True)
    return cache

def fetch_schedule():
    cache = ensure_cache()
    p = os.path.join(cache, "schedules.csv.gz")
    if not os.path.exists(p):
        r = requests.get(NFL_SCHED, timeout=60)
        r.raise_for_status()
        open(p, "wb").write(r.content)
    df = pd.read_csv(p, compression="gzip", low_memory=False)
    df.to_csv(os.path.join(cache,"schedule.csv"), index=False)
    print("Wrote cache/schedule.csv")
    return df

def fetch_odds(api_key: str):
    url = f"{ODDS_BASE}/sports/americanfootball_nfl/odds"
    r = requests.get(url, params={"apiKey": api_key, "regions":"us", "markets":"h2h,spreads,totals", "oddsFormat":"american"}, timeout=60)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    cache = ensure_cache()
    df = fetch_schedule()

    key = os.environ.get("THE_ODDS_API_KEY")
    if key:
        try:
            data = fetch_odds(key)
            import json
            with open(os.path.join(cache,"odds_raw.json"), "w", encoding="utf-8") as f:
                f.write(json.dumps(data))
            print("Wrote cache/odds_raw.json")
        except Exception as e:
            print("Odds fetch failed:", e)
    else:
        print("N

@'
import os
cache = os.environ.get("DATA_CACHE_DIR","./cache")
print("Predict placeholder. When historical features are added, this will write pick_sheet.csv to", cache)
