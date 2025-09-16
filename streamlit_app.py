# streamlit_app.py
from __future__ import annotations
import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="🏈 NFL Picks", layout="wide")
st.title("🏈 NFL Picks — Live Sheet")

DATA_DIR = "data"
CACHE_DIR = "cache"

def load_csv(name: str) -> pd.DataFrame | None:
    """Try data/ first, then cache/."""
    p = os.path.join(DATA_DIR, name)
    if os.path.exists(p):
        return pd.read_csv(p, low_memory=False)
    p2 = os.path.join(CACHE_DIR, name)
    if os.path.exists(p2):
        return pd.read_csv(p2, low_memory=False)
    return None

df = load_csv("pick_sheet.csv")
if df is None or df.empty:
    st.info("No pick_sheet.csv yet. Run `python scripts/fetch_and_build.py`, then copy results into /data.")
    st.stop()

# ---------------- Filters ----------------
c1, c2, c3 = st.columns(3)

with c1:
    seasons = sorted(df.get("season", pd.Series(dtype="int")).dropna().unique().tolist())
    season = st.selectbox("Season", seasons, index=max(0, len(seasons)-1)) if seasons else None

with c2:
    if season is not None and "week" in df.columns:
        weeks = sorted(df.loc[df["season"] == season, "week"].dropna().unique().tolist())
    else:
        weeks = sorted(df.get("week", pd.Series(dtype="int")).dropna().unique().tolist())
    week = st.selectbox("Week", weeks, index=0) if weeks else None

with c3:
    # ✅ Build team list as Series, THEN dropna, THEN unique
    teams_series = pd.concat(
        [
            df.get("home_team", pd.Series(dtype="object")),
            df.get("away_team", pd.Series(dtype="object")),
        ],
        ignore_index=True,
    )
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

# ---------------- Tabs ----------------
tab_ml, tab_spreads = st.tabs(["Moneyline (Model vs Market)", "Spread / ATS"])

with tab_ml:
    st.subheader("Moneyline — Market prices & vig-removed fair probabilities")
    cols = [
        "gameday", "home_team", "away_team",
        "home_ml", "away_ml",
        "home_prob", "away_prob",           # fair (vig-removed)
        "home_prob_raw", "away_prob_raw",   # raw (with vig)
        "home_prob_model", "away_prob_model"
    ]
    cols = [c for c in cols if c in filt.columns]
    if cols:
        st.dataframe(
            filt[cols].sort_values(["gameday", "home_team", "away_team"], na_position="last"),
            use_container_width=True,
        )
    else:
        st.warning("Moneyline columns not found yet. Rebuild data and try again.")

with tab_spreads:
    st.subheader("Spread / ATS — Market vs model edges")
    cols = [
        "gameday", "home_team", "away_team",
        "home_line", "home_spread_odds", "away_spread_odds",
        "model_spread", "edge_points", "edge_pct"
    ]
    cols = [c for c in cols if c in filt.columns]
    if cols:
        st.dataframe(
            filt[cols].sort_values(["gameday", "home_team", "away_team"], na_position="last"),
            use_container_width=True,
        )
    else:
        st.warning("Spread columns not found yet. Make sure your odds fetch includes `spreads` and rebuild.")
