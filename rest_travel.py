# nfl_model/rest_travel.py
from __future__ import annotations
import pandas as pd, numpy as np, math, os

# Uses your existing reference/nfl_stadiums.csv (already in the repo)
REF_PATH = os.path.join("reference", "nfl_stadiums.csv")

def _haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between two lat/lon points."""
    R = 6371.0  # Earth radius (km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

def load_stadiums() -> pd.DataFrame:
    """Load stadium coordinates; expect columns: team, stadium, lat, lon."""
    df = pd.read_csv(REF_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"team": "team_code"})
    return df[["team_code", "stadium", "lat", "lon"]]

def add_rest_and_travel(upcoming: pd.DataFrame, past_sched: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute rest days and away travel distance for each upcoming game.

    Returns:
      merged_df, feature_cols
        - merged_df: original upcoming with new columns
        - feature_cols: names of the new feature columns
    """
    st = load_stadiums()
    st = st.rename(columns={"team_code": "home_team"})  # to merge by home team later

    g = past_sched.copy()
    if "gameday" in g.columns:
        g["gameday"] = pd.to_datetime(g["gameday"], errors="coerce")
    else:
        g["gameday"] = pd.NaT

    # Build (team, last_played_date)
    home_hist = g[["home_team", "gameday"]].rename(columns={"home_team": "team"})
    away_hist = g[["away_team", "gameday"]].rename(columns={"away_team": "team"})
    hist = pd.concat([home_hist, away_hist], axis=0).dropna(subset=["gameday"])
    last_played = (hist.sort_values("gameday")
                        .groupby("team").gameday.max().reset_index()
                        .rename(columns={"team": "home_team", "gameday": "last_played"}))

    out = upcoming.copy()
    if "gameday" in out.columns:
        out["gameday"] = pd.to_datetime(out["gameday"], errors="coerce")

    # Rest days: home
    home = out.merge(last_played, on="home_team", how="left")
    home["home_rest_days"] = (home["gameday"] - home["last_played"]).dt.days
    home = home.drop(columns=["last_played"])

    # Rest days: away
    last_played_away = last_played.rename(columns={"home_team": "away_team"})
    away = home.merge(last_played_away, on="away_team", how="left")
    away["away_rest_days"] = (away["gameday"] - away["last_played"]).dt.days
    away = away.drop(columns=["last_played"])

    # Travel distance (away stadium -> home stadium)
    st_home = st.rename(columns={"stadium": "home_stadium", "lat": "home_lat", "lon": "home_lon"})
    st_away = st.rename(columns={"home_team": "away_team", "team_code": "away_team",
                                 "stadium": "away_stadium", "lat": "away_lat", "lon": "away_lon"})

    merged = (away.merge(st_home, on="home_team", how="left")
                  .merge(st_away, on="away_team", how="left"))

    merged["travel_km"] = merged.apply(
        lambda r: _haversine(r["away_lat"], r["away_lon"], r["home_lat"], r["home_lon"])
        if pd.notna(r.get("away_lat")) and pd.notna(r.get("home_lat")) else np.nan,
        axis=1
    )

    # Deltas / simple proxies
    merged["rest_delta"] = merged["home_rest_days"].fillna(0) - merged["away_rest_days"].fillna(0)
    merged["travel_dir_km"] = merged["travel_km"].fillna(0)

    feature_cols = ["home_rest_days", "away_rest_days", "rest_delta", "travel_km", "travel_dir_km"]
    return merged, feature_cols
