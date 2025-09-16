# scripts/fetch_and_build.py
import os, json, requests, pandas as pd
import nfl_data_py as nfl

ODDS_BASE = "https://api.the-odds-api.com/v4"

def fetch_schedule(season=2025):
    sched = nfl.import_schedules([season])
    return sched

def fetch_odds_spreads_only(api_key, regions="us", bookmaker="draftkings"):
    url = f"{ODDS_BASE}/sports/americanfootball_nfl/odds"
    r = requests.get(
        url,
        params={
            "apiKey": api_key,
            "regions": regions,
            "markets": "spreads",      # <-- force spreads only
            "bookmakers": bookmaker,   # try DK first; we can widen later
            "oddsFormat": "american",
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()

def main():
    season = 2025
    cache = "./cache"
    os.makedirs(cache, exist_ok=True)

    # schedule
    s = fetch_schedule(season)
    p_sched = os.path.join(cache, "schedule.csv")
    s.to_csv(p_sched, index=False)
    print(f"[schedule] wrote {p_sched} ({len(s)} rows)")

    # spreads-only probe
    key = os.environ.get("THE_ODDS_API_KEY")
    if not key:
        print("[odds] ERROR: THE_ODDS_API_KEY not set")
        return

    try:
        data = fetch_odds_spreads_only(key)
        p_odds = os.path.join(cache, "odds_raw.json")
        with open(p_odds, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[odds] wrote {p_odds} (spreads-only probe)")
    except Exception as e:
        print("[odds] fetch failed:", e)

if __name__ == "__main__":
    main()
