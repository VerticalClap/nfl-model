import os, json, time, requests, pandas as pd

# Try these in order (first is the most reliable raw link)
SCHEDULE_URLS = [
    "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/schedules/schedules.csv.gz",
    "https://github.com/nflverse/nflfastR-data/raw/master/schedules/schedules.csv.gz",
]

ODDS_BASE = "https://api.the-odds-api.com/v4"

def ensure_cache():
    cache = os.environ.get("DATA_CACHE_DIR","./cache")
    os.makedirs(cache, exist_ok=True)
    return cache

def _download(url, timeout=60):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def fetch_schedule(cache, max_retries=2, pause=2.0):
    gz_path = os.path.join(cache, "schedules.csv.gz")
    last_err = None
    for url in SCHEDULE_URLS:
        for attempt in range(1, max_retries+1):
            try:
                print(f"Fetching schedule (attempt {attempt}) from {url}")
                blob = _download(url, timeout=60)
                with open(gz_path, "wb") as f:
                    f.write(blob)
                df = pd.read_csv(gz_path, compression="gzip", low_memory=False)
                out = os.path.join(cache, "schedule.csv")
                df.to_csv(out, index=False)
                print(f"Wrote {out} ({len(df)} rows)")
                return df
            except Exception as e:
                last_err = e
                print(f"  failed: {e}")
                time.sleep(pause)
        print("  trying next URL…")
    raise RuntimeError(f"All schedule URLs failed. Last error: {last_err}")

def fetch_odds(api_key):
    url = f"{ODDS_BASE}/sports/americanfootball_nfl/odds"
    r = requests.get(url, params={
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american"
    }, timeout=60)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    cache = ensure_cache()
    fetch_schedule(cache)

    key = os.environ.get("THE_ODDS_API_KEY")
    if key:
        try:
            data = fetch_odds(key)
            with open(os.path.join(cache, "odds_raw.json"), "w", encoding="utf-8") as f:
                json.dump(data, f)
            print("Wrote cache/odds_raw.json")
        except Exception as e:
            print("Odds fetch failed:", e)
    else:
        print("No THE_ODDS_API_KEY set; skipping odds.")

