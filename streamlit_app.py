# streamlit_app.py
from __future__ import annotations
import os
import time
import pandas as pd
import streamlit as st

st.set_page_config(page_title="🏈 NFL Picks", layout="wide")
st.title("🏈 NFL Picks — Live Sheet")

# Debug banner so we know the correct file is running
p = os.path.abspath(__file__)
st.caption(f"✅ Loaded: {p} — mtime: {time.ctime(os.path.getmtime(p))}")

DATA_DIR = "data"    # repo-tracked copy (from Actions or manual commit)
CACHE_DIR = "cache"  # local build output (fallback)

def load_csv(name: str) -> pd.DataFrame | None:
    p1 = os.path.join(DATA_DIR, name)
    if os.path.exists(p1):
        return pd.read_csv(p1, low_memory=False)
    p2 = os.path.join(CACHE_DIR, name)
    if os.path.exists(p2):
        return pd.read_csv(p2, low_memory=False)
    return None

df = load_csv("pick_sheet.csv")
if df is None or df.empty:
    st.info("No pick_sheet.csv yet. Run `python scripts/fetch_and_build.py`, then copy into /data.")
    st.stop()

# ---- Filters ----
c1, c2 = st.columns(2)
with c1:
    seasons = sorted(df["season"].dropna().unique().tolist()) if "season" in df.columns else []
    season = st.selectbox("Season", seasons, index=max(0, len(seasons)-1)) if seasons else None
with c2:
    weeks = (
        sorted(df.loc[df["season"] == season, "week"].dropna().unique().tolist())
        if season is not None and "week" in df.columns
        else []
    )
    week = st.selectbox("Week", weeks, index=0) if weeks else None

# ✅ SAFE team list (operate on Series → dropna → unique)
home_series = df.get("home_team", pd.Series(dtype="object"))
away_series = df.get("away_team", pd.Series(dtype="object"))
teams_series = pd.concat([home_series, away_series], ignore_index=True)
all_teams = sorted(teams_series.dropna().unique().tolist())
team_filter = st.selectbox("Filter by team (optional)", ["(All)"] + all_teams, index=0)

# Apply filters
filt = df.copy()
if season is not None and "season" in filt.columns:
    filt = filt[filt["season"] == season]
if week is not None and "week" in filt.columns:
    filt = filt[filt["week"] == week]
if team_filter != "(All)" and {"home_team", "away_team"}.issubset(filt.columns):
    filt = filt[(filt["home_team"] == team_filter) | (filt["away_team"] == team_filter)]

# ---- Tabs ----
tab_ml, tab_spreads = st.tabs(["Moneyline", "Spreads"])

with tab_ml:
    st.subheader("Moneylines (DraftKings preferred → otherwise consensus)")
    ml_cols = [
        "gameday", "home_team", "away_team",
        "home_ml", "away_ml", "home_prob", "away_prob",
        "home_prob_raw", "away_prob_raw",
    ]
    ml_cols = [c for c in ml_cols if c in filt.columns]
    if ml_cols:
        st.dataframe(
            filt[ml_cols].sort_values(["gameday", "home_team", "away_team"], na_position="last"),
            use_container_width=True,
        )
    else:
        st.warning("Moneyline columns not found. Rebuild data and try again.")

with tab_spreads:
    st.subheader("Point Spreads (home_line < 0 → home favored)")
    sp_cols = [
        "gameday", "home_team", "away_team",
        "home_line", "home_spread_odds", "away_spread_odds",
        # Optional model outputs if present:
        "model_spread", "edge_spread"
    ]
    sp_cols = [c for c in sp_cols if c in filt.columns]
    if sp_cols:
        st.dataframe(
            filt[sp_cols].sort_values(["gameday", "home_team", "away_team"], na_position="last"),
            use_container_width=True,
        )
    else:
        st.warning(
            "Spreads not available. Ensure your odds fetch includes 'spreads' and your pipeline writes "
            "`home_line`, `home_spread_odds`, `away_spread_odds` into pick_sheet.csv."
        )
