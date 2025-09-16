# scripts/fetch_and_build.py
import os, json, requests, pandas as pd
import nfl_data_py as nfl
from datetime import datetime, timedelta, timezone

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
    else:
        for alt in ["game_date", "start_time"]:
            if alt in df.columns:
                df["gameday"] = pd.to_datetime(df[alt], errors="coerce")
                break

    # keep current & future only
    today = pd.Timestamp.today().normalize()
    df = df[df["gameday"] >= today]

    keep = [c for c in ["season","week","gameday","home_team","away_team","game_id"] if c in df.columns]
    df = df[keep].sort_values(["week","gameday","home_team","away_team"]).reset_index(drop=True)

    out = os.path.join(cache, "schedule.csv")
    df.to_csv(out, index=False)
    print(f"[schedule] wrote {out} ({len(df)} rows)")
    return df

def _iso(ts: datetime) -> str:
    return ts.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

def fetch_odds_raw(api_key: str, days_ahead: int = 9) -> list:
    """
    Ask for BOTH moneylines (h2h) and spreads, broaden regions,
    and limit to a near-term window to increase the chance spreads are returned.
    """
    now = datetime.utcnow()
    params = {
        "apiKey": api_key,
        "regions": "us,us2",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "commenceTimeFrom": _iso(now - timedelta(days=1)),
        "commenceTimeTo":   _iso(now + timedelta(days=days_ahead)),
    }
    url = f"{ODDS_BASE}/sports/americanfootball_nfl/odds"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Debug: list markets we actually got back
    seen = set()
    for ev in data:
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                k = m.get("key")
                if k:
                    seen.add(k)
    print("[DEBUG] markets seen:", sorted(seen))
    return data

if __name__ == "__main__":
    cache = ensure_cache()
    fetch_schedule_current_season(cache)

    key = os.environ.get("THE_ODDS_API_KEY")
    if not key:
        print("[odds] skipped: THE_ODDS_API_KEY not set")
    else:
        try:
            data = fetch_odds_raw(key, days_ahead=9)
            out = os.path.join(cache, "odds_raw.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[odds] wrote {out}")
        except Exception as e:
            print("[odds] fetch failed:", e)
