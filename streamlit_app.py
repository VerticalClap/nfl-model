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
    p1 = os.path.join(DATA_DIR, name)
    if os.path.exists(p1):
        return pd.read_csv(p1, low_memory=False)
    p2 = os.path.join(CACHE_DIR, name)
    if os.path.exists(p2):
        return pd.read_csv(p2, low_memory=False)
    return None

# ---------- Load ----------
df = load_csv("pick_sheet.csv")
if df is None or df.empty:
    st.info("No pick_sheet.csv yet. Run: `python scripts/fetch_and_build.py`, then copy into /data.")
    st.stop()

# ---------- Filters ----------
c1, c2, c3 = st.columns(3)
with c1:
    seasons = sorted(df["season"].dropna().unique().tolist()) if "season" in df.columns else []
    season = st.selectbox("Season", seasons, index=max(0, len(seasons) - 1)) if seasons else None
with c2:
    weeks = (
        sorted(df.loc[df["season"] == season, "week"].dropna().unique().tolist())
        if season is not None and "week" in df.columns else []
    )
    week = st.selectbox("Week", weeks, index=0) if weeks else None
with c3:
    teams_series = pd.concat(
        [
            df.get("home_team", pd.Series(dtype="object")),
            df.get("away_team", pd.Series(dtype="object")),
        ],
        ignore_index=True,
    )
    all_teams = sorted(teams_series.dropna().unique().tolist())
    team_filter = st.selectbox("Filter by team (optional)", ["(All)"] + all_teams, index=0)

filt = df.copy()
if season is not None and "season" in filt.columns:
    filt = filt[filt["season"] == season]
if week is not None and "week" in filt.columns:
    filt = filt[filt["week"] == week]
if team_filter != "(All)" and {"home_team", "away_team"}.issubset(filt.columns):
    filt = filt[(filt["home_team"] == team_filter) | (filt["away_team"] == team_filter)]

# Recompute edges here if needed (in case the CSV doesn’t carry them yet)
if {"model_spread", "home_line"}.issubset(filt.columns) and "edge_spread_pts" not in filt.columns:
    filt = filt.assign(edge_spread_pts=(filt["model_spread"] - filt["home_line"]).round(2))
if {"model_home_prob", "home_prob"}.issubset(filt.columns) and "edge_prob" not in filt.columns:
    filt = filt.assign(edge_prob=(filt["model_home_prob"] - filt["home_prob"]).round(3))

tab_ml, tab_spreads, tab_model = st.tabs(["Moneyline", "Spreads", "Model vs Market"])

# ---------- Moneyline ----------
with tab_ml:
    st.subheader("Moneylines (Market)")
    cols = [c for c in [
        "gameday", "home_team", "away_team",
        "home_ml", "away_ml", "home_prob", "away_prob",
        "home_prob_raw", "away_prob_raw",
    ] if c in filt.columns]
    if cols:
        st.dataframe(
            filt[cols].sort_values(["gameday", "home_team", "away_team"], na_position="last"),
            width="stretch",
        )
    else:
        st.warning("Moneyline columns not found yet.")

# ---------- Spreads ----------
with tab_spreads:
    st.subheader("Point Spreads (Market)")
    cols = [c for c in [
        "gameday", "home_team", "away_team",
        "home_line", "home_spread_odds", "away_spread_odds",
    ] if c in filt.columns]
    if cols:
        st.dataframe(
            filt[cols].sort_values(["gameday", "home_team", "away_team"], na_position="last"),
            width="stretch",
        )
    else:
        st.warning("Spreads not available. Rebuild data with spreads and refresh.")

# ---------- Model vs Market ----------
with tab_model:
    st.subheader("Model vs Market")
    left, right = st.columns([3, 2])

    with left:
        cols = [c for c in [
            "gameday", "home_team", "away_team",
            "model_spread", "model_home_prob",
            "home_line", "home_ml", "home_prob",
            "edge_spread_pts", "edge_prob",
        ] if c in filt.columns]
        if cols:
            st.dataframe(
                filt[cols].sort_values(["gameday", "home_team", "away_team"], na_position="last"),
                width="stretch",
            )
        else:
            st.warning("Model columns not present. Rebuild to populate `model_spread` / `model_home_prob`.")

    with right:
        st.caption("Legend")
        st.markdown(
            """
            - **model_spread**: Home − Away points (positive = home favorite).
            - **model_home_prob**: Model win probability for home team.
            - **home_line / home_prob**: Market spread / win probability.
            - **edge_spread_pts**: model_spread − home_line.
            - **edge_prob**: model_home_prob − home_prob.
            """
        )
