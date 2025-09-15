import os, json, time, requests, pandas as pd

# New primary source: nflverse-data releases
SCHEDULE_URLS = [
    # Official release (CSV)
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/schedules.csv",
    # Same release, gzipped variant (sometimes present)
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/schedules.csv.gz",
    # Legacy mirrors as last resort
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
    return r

def fetch_schedule(cache, max_retries=2, pause=2.0):
    out_csv = os.path.join(cache, "schedule.csv")
    last_err = None
    for url in SCHEDULE_URLS:
        for attempt in range(1, max_retries+1):
            try:
                print(f"Fetching schedule (attempt {attempt}) from {url}")
                r = _download(url, timeout=60)
                # Decide how to read: csv vs csv.gz
                if url.endswith(".csv.gz"):
                    import io, gzip
                    df = pd.read_csv(io.BytesIO(gzip.decompress(r.content)))
                else:
                    # Some GitHub release assets need .content, not .text
                    import io
                    df = pd.read_csv(io.BytesIO(r.content))
                df.to_csv(out_csv, index=False)
                print(f"Wrote {out_csv} ({len(df)} rows)")
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
