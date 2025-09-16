# scripts/fetch_and_build.py
import os, json, requests, pandas as pd
import nfl_data_py as nfl
from datetime import datetime, timedelta, timezone

ODDS_BASE = "https://api.the-odds-api.com/v4"

def fetch_schedule(season=2025):
    return nfl.import_schedules([season])

def _iso(ts: datetime) -> str:
    return ts.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

def fetch_odds(api_key: str, days_ahead: int = 9):
    """
    Ask for BOTH moneylines (h2h) and spreads, across US regions,
    and restrict to a near-term window so books actually return spreads.
    """
    now = datetime.utcnow()
    frm = _iso(now - timedelta(days=1))
    to  = _iso(now + timedelta(days=days_ahead))

    params = {
        "apiKey": api_key,
        "regions": "us,us2",          # broaden US coverage
        "markets": "h2h,spreads",     # <-- ask for spreads
        "oddsFormat": "american",
        "dateFormat": "iso",
        "commenceTimeFrom": frm,      # time window increases chance spreads are returned
        "commenceTimeTo": to,
        # no bookmaker filter -> allow any book that has spreads
    }

    url = f"{ODDS_BASE}/sports/americanfootball_nfl/odds"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    # small debug summary to confirm markets present
    markets = set()
    for ev in data:
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                k = m.get("key")
                if k:
                    markets.add(k)
    print("[odds] markets seen:", sorted(markets))
    return data

def main():
    season = 2025
    cache = "./cache"
    os.makedirs(cache, exist_ok=True)

    # 1) schedule
    sched = fetch_schedule(season)
    sched_path = os.path.join(cache, "schedule.csv")
    sched.to_csv(sched_path, index=False)
    print(f"[schedule] wrote {sched_path} ({len(sched)} rows)")

    # 2) odds
    key = os.environ.get("THE_ODDS_API_KEY")
    if not key:
        print("[odds] ERROR: THE_ODDS_API_KEY not set")
        return

    data = fetch_odds(key, days_ahead=9)
    odds_path = os.path.join(cache, "odds_raw.json")
    with open(odds_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[odds] wrote {odds_path}")

if __name__ == "__main__":
    main()
