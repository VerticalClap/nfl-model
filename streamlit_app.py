# streamlit_app.py
from __future__ import annotations
import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="🏈 NFL Picks", layout="wide")
st.title("🏈 NFL Picks — Live Sheet")

DATA_DIR = "data"   # repo-tracked copy (from Actions or your manual commit)
CACHE_DIR = "cache" # local build output (fallback)

def _load_csv(name: str) -> pd.DataFrame | None:
    p = os.path.join(DATA_DIR, name)
    if os.path.exists(p):
        return pd.read_csv(p, low_memory=False)
    p2 = os.path.join(CACHE_DIR, name)
    if os.path.exists(p2):
        return pd.read_csv(p2, low_memory=False)
    return None

df = _load_csv("pick_sheet.csv")
if df is None or df.empty:
    st.info("No pick_sheet.csv yet. Run `python scripts/fetch_and_build.py`, then copy to `data/`.")
    st.stop()

# ------- Filters -------
left, right = st.columns(2)
with left:
    seasons = sorted(df["season"].dropna().unique().tolist()) if "season" in df.columns else []
    season = st.selectbox("Season", seasons, index=max(0, len(seasons)-1)) if seasons else None
with right:
    weeks = (
        sorted(df.loc[df["season"] == season, "week"].dropna().unique().tolist())
        if season is not None and "week" in df.columns
        else []
    )
    week = st.selectbox("Week", weeks, index=0) if weeks else None

# Optional team filter (fix for dropna/unique)
teams_series = pd.concat(
    [df["home_team"], df["away_team"]],
    ignore_index=True,
) if {"home_team","away_team"}.issubset(df.columns) else pd.Series(dtype="object")

all_teams = sorted(teams_series.dropna().unique().tolist())
team_filter = st.selectbox("Filter by team (optional)", ["(All)"] + all_teams, index=0)

filt = df.copy()
if season is not None and "season" in df.columns:
    filt = filt[filt["season"] == season]
if week is not None and "week" in df.columns:
    filt = filt[filt["week"] == week]
if team_filter != "(All)" and {"home_team","away_team"}.issubset(filt.columns):
    filt = filt[(filt["home_team"] == team_filter) | (filt["away_team"] == team_filter)]

# ------- Tabs -------
tab_ml, tab_spreads = st.tabs(["Moneyline", "Spreads"])

with tab_ml:
    st.subheader("Moneylines (DraftKings preferred → otherwise median across books)")
    cols_ml = [
        "gameday","home_team","away_team",
        "home_ml","away_ml","home_prob","away_prob",
        "home_prob_raw","away_prob_raw",
    ]
    cols_ml = [c for c in cols_ml if c in filt.columns]
    if cols_ml:
        st.dataframe(
            filt[cols_ml].sort_values(["gameday","home_team","away_team"], na_position="last"),
            use_container_width=True,
        )
    else:
        st.warning("Moneyline columns not found yet. Rebuild data or check the pipeline.")

with tab_spreads:
    st.subheader("Point Spreads (home_line < 0 means home is favored)")
    cols_sp = ["gameday","home_team","away_team","home_line","home_spread_odds","away_spread_odds"]
    cols_sp = [c for c in cols_sp if c in filt.columns]
    if cols_sp:
        st.dataframe(
            filt[cols_sp].sort_values(["gameday","home_team","away_team"], na_position="last"),
            use_container_width=True,
        )
    else:
        st.warning(
            "Spreads not available. Make sure your odds fetch included 'spreads' and rebuild:\n"
            "`python scripts/fetch_and_build.py` → `python -c \"from nfl_model.pipeline import build_pick_sheet; build_pick_sheet('./cache')\"` "
            "→ copy cache CSVs into /data."
        )
