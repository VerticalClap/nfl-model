import os, json, pandas as pd
import nfl_data_py as nfl

ODDS_BASE = "https://api.the-odds-api.com/v4"

def ensure_cache():
    cache = os.environ.get("DATA_CACHE_DIR", "./cache")
    os.makedirs(cache, exist_ok=True)
    return cache

def fetch_schedule(cache):
    # Fetch full schedule (all seasons available)
    print("Fetching schedule via nfl_data_py…")
    df = nfl.import_schedules([2020, 2021, 2022, 2023, 2024])  # add seasons as needed
    out = os.path.join(cache, "schedule.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")
    return df

def fetch_odds(api_key):
    import requests
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
