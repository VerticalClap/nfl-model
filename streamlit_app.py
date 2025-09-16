# streamlit_app.py
import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NFL Picks — Live", layout="wide")
st.title("🏈 NFL Picks — Live Sheet")

DATA_PATH = os.path.join("data", "pick_sheet.csv")

@st.cache_data(ttl=60)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    # Convert numeric-likes
    for c in ["home_ml","away_ml","home_prob","away_prob",
              "home_spread","away_spread","home_spread_price","away_spread_price",
              "home_cover_prob","away_cover_prob",
              "total_points","over_price","under_price","over_prob","under_prob",
              "home_rest_days","away_rest_days","rest_delta","travel_km","travel_dir_km"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data(DATA_PATH)
if df.empty:
    st.warning("No pick_sheet.csv yet. Run the GitHub Action or the local build step.")
    st.stop()

# ---- Filters ----
top = st.columns([1.2, 1.2, 2, 1.2])
with top[0]:
    season = st.selectbox("Season", sorted(df["season"].dropna().unique()))
w_df = df[df["season"] == season].copy()

with top[1]:
    week = st.selectbox("Week", sorted(w_df["week"].dropna().unique()))
show = w_df[w_df["week"] == week].copy()

with top[2]:
    # optional team filter
    teams = sorted(pd.unique(show[["home_team","away_team"]].values.ravel()))
    team = st.selectbox("Filter by team (optional)", ["(All)"] + teams)
    if team != "(All)":
        show = show[(show["home_team"] == team) | (show["away_team"] == team)]

with top[3]:
    sort_by_time = st.checkbox("Sort by kickoff time", value=True)

if "gameday" in show.columns and sort_by_time:
    show = show.sort_values(["gameday","home_team","away_team"])

# Small header
st.caption("Columns refresh automatically from the repo’s /data/pick_sheet.csv (auto-refresh ~60s).")

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["Moneyline (Model vs Market)", "Spread / ATS", "Totals (O/U)"])

# Helper to show only existing cols
def present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

with tab1:
    st.subheader("Moneyline — Market prices & vig-removed fair probabilities")
    cols = [
        "gameday","home_team","away_team",
        "home_ml","away_ml",
        "home_prob","away_prob",               # fair (vig-removed) from book
        "home_prob_raw","away_prob_raw",       # raw (vigged) from book
        # If/when you add model probs in Step 4+, include below:
        "home_prob_model","away_prob_model",
        "home_kelly_5pct","away_kelly_5pct",   # if your pipeline computes these
        "home_rest_days","away_rest_days","rest_delta","travel_km","travel_dir_km",
    ]
    st.dataframe(show[present(show, cols)], use_container_width=True)

with tab2:
    st.subheader("Spread / ATS — Lines, prices, and cover probabilities (vig-removed)")
    cols = [
        "gameday","home_team","away_team",
        "home_spread","away_spread",
        "home_spread_price","away_spread_price",
        "home_cover_prob","away_cover_prob",
        # If/when you add model spread/ATS later, show them here:
        "model_spread_home","spread_edge_pts","home_cover_model",
        "home_rest_days","away_rest_days","rest_delta","travel_km","travel_dir_km",
    ]
    st.dataframe(show[present(show, cols)], use_container_width=True)

with tab3:
    st.subheader("Totals (Over/Under) — Line, prices, and probabilities (vig-removed)")
    cols = [
        "gameday","home_team","away_team",
        "total_points","over_price","under_price",
        "over_prob","under_prob",
    ]
    st.dataframe(show[present(show, cols)], use_container_width=True)

# Helpful note
st.info(
    "Tip: If you want **DraftKings-only** numbers vs **consensus** (median across books), "
    "edit `BOOKS` in `nfl_model/pipeline.py` — set to `['draftkings']` or `[]`."
)
