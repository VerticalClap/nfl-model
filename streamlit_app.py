import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NFL Picks — Live", layout="wide")
st.title("🏈 NFL Picks — Live Sheet")

DATA_PATH = os.path.join("data", "pick_sheet.csv")  # pulled from GitHub by Actions

@st.cache_data(ttl=60)
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, low_memory=False)
    else:
        df = pd.DataFrame()
    # parse dates, if present
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    return df

df = load_data()
if df.empty:
    st.warning("No pick_sheet.csv yet. Run the GitHub Action or the local build step.")
    st.stop()

# Filters
col1, col2 = st.columns([2,2])
with col1:
    season = st.selectbox("Season", sorted(df["season"].dropna().unique()))
with col2:
    weeks = sorted(df.loc[df["season"]==season, "week"].dropna().unique())
    week = st.selectbox("Week", weeks)

show = df[(df["season"]==season) & (df["week"]==week)].copy()
if show.empty:
    st.info("No rows for the chosen filters.")
    st.stop()

# Tabs: Moneyline vs Spread
tab1, tab2 = st.tabs(["Moneyline (Model vs Market)", "Spread / ATS"])

with tab1:
    st.subheader("Model vs Market — Moneyline")
    cols = [
        "gameday","home_team","away_team",
        "home_ml","away_ml","home_prob","away_prob",          # market fair
        "home_prob_model","away_prob_model",                  # model
        "home_edge","away_edge",
        "home_kelly_5pct","away_kelly_5pct"
    ]
    present = [c for c in cols if c in show.columns]
    st.dataframe(show[present].sort_values(["gameday","home_team"]))

with tab2:
    st.subheader("Model vs Market — Spread (ATS)")
    cols = [
        "gameday","home_team","away_team",
        "model_spread_home",                                  # model spread (home minus away)
        "home_spread","away_spread","home_spread_price","away_spread_price",
        "home_cover_prob","away_cover_prob",                  # market fair cover probs (if prices)
        "spread_edge_pts"
    ]
    present = [c for c in cols if c in show.columns]
    st.dataframe(show[present].sort_values(["gameday","home_team"]))

st.caption("Data auto-loads from the repo’s /data folder on GitHub (auto-refreshed every 60s).")
