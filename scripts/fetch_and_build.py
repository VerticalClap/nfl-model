# scripts/fetch_and_build.py
import os, json, requests, pandas as pd
import nfl_data_py as nfl

ODDS_BASE = "https://api.the-odds-api.com/v4"


def fetch_schedule(season=2025):
    return nfl.import_schedules([season])


def fetch_odds(api_key, markets="h2h,spreads", regions="us,us2", bookmakers=None):
    """Broaden regions and (by default) do NOT filter to a single book."""
    params = {
        "apiKey": api_key,
        "regions": regions,          # broaden to us + us2
        "markets": markets,          # ask for spreads + moneyline
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers

    url = f"{ODDS_BASE}/sports/americanfootball_nfl/odds"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    season = 2025
    cache = "./cache"
    os.makedirs(cache, exist_ok=True)

    # 1) Schedule
    s = fetch_schedule(season)
    p_sched = os.path.join(cache, "schedule.csv")
    s.to_csv(p_sched, index=False)
    print(f"[schedule] wrote {p_sched} ({len(s)} rows)")

    # 2) Odds: try (h2h + spreads) across many US books
    key = os.environ.get("THE_ODDS_API_KEY")
    if not key:
        print("[odds] ERROR: THE_ODDS_API_KEY not set")
        return

    data = fetch_odds(key, markets="h2h,spreads", regions="us,us2", bookmakers=None)

    # If spreads still missing, try a spreads-only second pass and merge
    markets_seen = set()
    for ev in data:
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m.get("key"):
                    markets_seen.add(m["key"])

    if "spreads" not in markets_seen:
        print("[odds] spreads not present; trying spreads-only fallback…")
        data_spreads = fetch_odds(key, markets="spreads", regions="us,us2", bookmakers=None)
        # Merge: prefer spreads from fallback if event ids match
        by_id = {ev.get("id") or ev.get("event_id") or i: ev for i, ev in enumerate(data)}
        for ev in data_spreads:
            ev_id = ev.get("id") or ev.get("event_id")
            if ev_id in by_id:
                # append any spreads markets found
                base = by_id[ev_id]
                base_bks = base.setdefault("bookmakers", [])
                for bk in ev.get("bookmakers", []):
                    # attach spreads market into existing bookmaker or append new bk
                    found = False
                    for bb in base_bks:
                        if bb.get("key") == bk.get("key"):
                            bb.setdefault("markets", [])
                            bb["markets"].extend(bk.get("markets", []))
                            found = True
                            break
                    if not found:
                        base_bks.append(bk)
            else:
                by_id[ev_id] = ev
        data = list(by_id.values())

    p_odds = os.path.join(cache, "odds_raw.json")
    with open(p_odds, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[odds] wrote {p_odds}")


if __name__ == "__main__":
    main()
