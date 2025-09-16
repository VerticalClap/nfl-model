import os
import nfl_data_py as nfl
import requests
import json
import pandas as pd
from nfl_model.pipeline import build_pick_sheet

ODDS_BASE = "https://api.the-odds-api.com/v4"


def fetch_schedule(season=2025):
    """Fetch NFL schedule for a given season."""
    sched = nfl.import_schedules([season])
    return sched


def fetch_odds(api_key, markets="h2h,spreads", regions="us"):
    """Fetch odds data from The Odds API for NFL games."""
    url = f"{ODDS_BASE}/sports/americanfootball_nfl/odds"
    r = requests.get(
        url,
        params={
            "apiKey": api_key,
            "regions": regions,
            "markets": markets,       # moneylines and spreads
            "bookmakers": "draftkings",  # only DraftKings for consistency
            "oddsFormat": "american",
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def main():
    season = 2025
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Fetch schedule
    print(f"[schedule] fetching NFL schedule for {season}…")
    sched = fetch_schedule(season)
    sched_path = os.path.join(cache_dir, "schedule.csv")
    sched.to_csv(sched_path, index=False)
    print(f"[schedule] wrote {sched_path} ({len(sched)} rows)")

    # 2. Fetch odds
    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        print("[odds] ERROR: THE_ODDS_API_KEY not set")
        return

    try:
        data = fetch_odds(api_key)
        odds_path = os.path.join(cache_dir, "odds_raw.json")
        with open(odds_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[odds] wrote {odds_path}")
    except Exception as e:
        print("[odds] fetch failed:", e)

    # 3. Build pick sheet
    try:
        build_pick_sheet(cache_dir)
    except Exception as e:
        print("[pick_sheet] build failed:", e)


if __name__ == "__main__":
    main()
