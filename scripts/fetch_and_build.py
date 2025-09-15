import os, io, json, requests, pandas as pd
import nfl_data_py as nfl

ODDS_BASE = "https://api.the-odds-api.com/v4"

def ensure_cache() -> str:
    cache = os.environ.get("DATA_CACHE_DIR", "./cache")
    os.makedirs(cache, exist_ok=True)
    return cache

def fetch_schedule_current_season(cache: str) -> pd.DataFrame:
    season = pd.Timestamp.today().year
    print(f"[schedule] fetching NFL schedule for {season}…")
    df = nfl.import_schedules([season])

    # normalize columns we need
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    else:
        # fallback if nfl_data_py changes
        for alt in ["game_date", "start_time"]:
            if alt in df.columns:
                df["gameday"] = pd.to_datetime(df[alt], errors="coerce")
                break

    # keep upcoming only
    today = pd.Timestamp.today().normalize()
    df = df[df["gameday"] >= today]

    # small, consistent subset
    keep = [c for c in ["season","week","gameday","home_team","away_team","game_id"] if c in df.columns]
    df = df[keep].sort_values(["week","gameday","home_team","away_team"]).reset_index(drop=True)

    out = os.path.join(cache, "schedule.csv")
    df.to_csv(out, index=False)
    print(f"[schedule] wrote {out} ({len(df)} rows)")
    return df

def fetch_odds_raw(api_key: str) -> list:
    url = f"{ODDS_BASE}/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american"
    }
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    cache = ensure_cache()
    fetch_schedule_current_season(cache)

    key = os.environ.get("THE_ODDS_API_KEY", "").strip()
    if key:
        try:
            raw = fetch_odds_raw(key)
            with open(os.path.join(cache, "odds_raw.json"), "w", encoding="utf-8") as f:
                json.dump(raw, f)
            print("[odds] wrote cache/odds_raw.json")
        except Exception as e:
            print(f"[odds] fetch failed: {e}")
    else:
        print("[odds] THE_ODDS_API_KEY not set; skipping odds.")
