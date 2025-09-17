# scripts/fetch_and_build.py
import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta

ODDS_BASE = "https://api.the-odds-api.com/v4"

# ------------------------------
# Utilities / cache
# ------------------------------
def ensure_cache() -> str:
    cache = os.environ.get("DATA_CACHE_DIR", "./cache")
    os.makedirs(cache, exist_ok=True)
    return cache

TEAM_FIX = {"LA": "LAR", "SD": "LAC", "OAK": "LV"}  # historical codes -> modern

def _normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].replace(TEAM_FIX)
    if "away_team" in df.columns:
        df["away_team"] = df["away_team"].replace(TEAM_FIX)
    return df

# ------------------------------
# Schedule (upcoming only)
# ------------------------------
def build_schedule(cache: str) -> pd.DataFrame:
    try:
        import nfl_data_py as nfl
    except Exception as e:
        raise RuntimeError(
            "nfl_data_py is required. pip install nfl-data-py"
        ) from e

    season = pd.Timestamp.today().year
    print(f"[schedule] building schedule for {season}")
    sched = nfl.import_schedules([season])

    # pick a consistent 'gameday' column
    if "gameday" not in sched.columns:
        for alt in ["game_date", "start_time"]:
            if alt in sched.columns:
                sched["gameday"] = pd.to_datetime(sched[alt], errors="coerce")
                break
    else:
        sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")

    # upcoming only (today or later)
    today = pd.Timestamp.today().normalize()
    sched = sched[sched["gameday"] >= today]

    keep = [c for c in ["season", "week", "gameday", "home_team", "away_team", "game_id"] if c in sched.columns]
    sched = sched[keep].sort_values(["week", "gameday", "home_team", "away_team"]).reset_index(drop=True)
    sched = _normalize_teams(sched)

    out = os.path.join(cache, "schedule.csv")
    sched.to_csv(out, index=False)
    print(f"[schedule] wrote {out} ({len(sched)} rows)")
    return sched

# ------------------------------
# Odds fetching & parsing
# ------------------------------
def fetch_odds_raw(api_key: str, cache: str) -> list:
    # Get both markets (spreads + h2h) in a single call
    params = {
        "apiKey": api_key,
        "regions": "us,us2",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    url = f"{ODDS_BASE}/sports/americanfootball_nfl/odds"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    raw_path = os.path.join(cache, "odds_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[odds] wrote {raw_path}")
    return data

def _american_to_prob(ml: float) -> float:
    """Convert American moneyline to implied probability (no vig removed)."""
    if ml is None or pd.isna(ml):
        return None
    ml = float(ml)
    if ml < 0:
        return (-ml) / ((-ml) + 100.0)
    else:
        return 100.0 / (ml + 100.0)

def _vig_strip(ph: float, pa: float):
    """Remove vig via proportional rescale; returns (fair_ph, fair_pa)."""
    if ph is None or pa is None:
        return None, None
    total = ph + pa
    if total == 0:
        return None, None
    return ph / total, pa / total

def extract_moneylines(raw: list, books=None) -> pd.DataFrame:
    """Return consensus moneylines + probs for each matchup."""
    if books is None:
        books = []  # empty = use all

    rows = []
    for ev in raw:
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not home or not away:
            continue

        hmls, amls = [], []
        for bk in ev.get("bookmakers", []):
            if books and bk.get("key") not in books:
                continue
            for m in bk.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for out in m.get("outcomes", []):
                    if out.get("name") == home:
                        hmls.append(out.get("price"))
                    elif out.get("name") == away:
                        amls.append(out.get("price"))

        # consensus (median) if available
        home_ml = float(pd.Series(hmls).median()) if len(hmls) else None
        away_ml = float(pd.Series(amls).median()) if len(amls) else None

        ph_raw = _american_to_prob(home_ml) if home_ml is not None else None
        pa_raw = _american_to_prob(away_ml) if away_ml is not None else None
        ph, pa = _vig_strip(ph_raw, pa_raw) if (ph_raw is not None and pa_raw is not None) else (None, None)

        rows.append({
            "home_team": home,
            "away_team": away,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_prob_raw": ph_raw,
            "away_prob_raw": pa_raw,
            "home_prob": ph,
            "away_prob": pa,
        })

    df = pd.DataFrame(rows)
    return _normalize_teams(df)

def extract_spreads(raw: list, books=None) -> pd.DataFrame:
    """Return consensus spreads (home_line) + spread odds."""
    if books is None:
        books = []  # empty = use all

    rows = []
    for ev in raw:
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not home or not away:
            continue

        home_lines, home_odds, away_odds = [], [], []
        for bk in ev.get("bookmakers", []):
            if books and bk.get("key") not in books:
                continue
            for m in bk.get("markets", []):
                if m.get("key") != "spreads":
                    continue
                # outcomes have 'point' (spread) and 'price' (odds)
                h_point, h_price, a_price = None, None, None
                for out in m.get("outcomes", []):
                    nm = out.get("name")
                    if nm == home:
                        h_point = out.get("point")
                        h_price = out.get("price")
                    elif nm == away:
                        a_price = out.get("price")
                if h_point is not None:
                    home_lines.append(h_point)
                if h_price is not None:
                    home_odds.append(h_price)
                if a_price is not None:
                    away_odds.append(a_price)

        home_line = float(pd.Series(home_lines).median()) if len(home_lines) else None
        h_odds = float(pd.Series(home_odds).median()) if len(home_odds) else None
        a_odds = float(pd.Series(away_odds).median()) if len(away_odds) else None

        rows.append({
            "home_team": home,
            "away_team": away,
            "home_line": home_line,
            "home_spread_odds": h_odds,
            "away_spread_odds": a_odds,
        })

    df = pd.DataFrame(rows)
    return _normalize_teams(df)

# ------------------------------
# Build pick sheet
# ------------------------------
def build_pick_sheet(cache: str) -> pd.DataFrame:
    sched_path = os.path.join(cache, "schedule.csv")
    odds_path = os.path.join(cache, "odds_raw.json")
    if not os.path.exists(sched_path) or not os.path.exists(odds_path):
        raise RuntimeError("Missing schedule.csv or odds_raw.json in cache. Run this script normally.")

    sched = pd.read_csv(sched_path, parse_dates=["gameday"])
    raw = json.load(open(odds_path, "r", encoding="utf-8"))

    # Extract both markets
    ml = extract_moneylines(raw)
    sp = extract_spreads(raw)

    # Merge into schedule
    out = sched.merge(ml, on=["home_team", "away_team"], how="left")
    out = out.merge(sp, on=["home_team", "away_team"], how="left")

    # ---- TEMP MODEL COLUMNS so dashboard renders (will be replaced by real model later)
    # model_home_prob/model_spread = mirror market; edges = 0
    if "home_prob" in out.columns and "home_line" in out.columns:
        out["model_home_prob"] = out["home_prob"].fillna(0.5)
        out["model_spread"] = out["home_line"].fillna(0.0)
        out["edge_prob"] = out["model_home_prob"] - out["home_prob"]
        out["edge_spread_pts"] = out["model_spread"] - out["home_line"]

    out = out.sort_values(["week", "gameday", "home_team", "away_team"]).reset_index(drop=True)
    out_path = os.path.join(cache, "pick_sheet.csv")
    out.to_csv(out_path, index=False)
    print(f"[pick_sheet] wrote {out_path} ({len(out)} rows)")
    return out

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    cache = ensure_cache()
    sched = build_schedule(cache)

    api_key = os.environ.get("THE_ODDS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing THE_ODDS_API_KEY. Set it and re-run.")

    raw = fetch_odds_raw(api_key, cache)
    build_pick_sheet(cache)
