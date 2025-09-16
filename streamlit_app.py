# streamlit_app.py
import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="🏈 NFL Picks", layout="wide")
st.title("🏈 NFL Picks — Live Sheet")

DATA_DIR = "data"  # repo-tracked copy
CACHE_DIR = "cache" # local fresh copy if present

def _load_csv(name: str) -> pd.DataFrame | None:
    # Prefer repo-tracked data/ so it works without local scripts
    p = os.path.join(DATA_DIR, name)
    if os.path.exists(p):
        return pd.read_csv(p, low_memory=False)
    # Fallback to cache
    p2 = os.path.join(CACHE_DIR, name)
    if os.path.exists(p2):
        return pd.read_csv(p2, low_memory=False)
    return None

df = _load_csv("pick_sheet.csv")
if df is None or df.empty:
    st.info("No pick_sheet.csv yet. Run the workflow or `python scripts/fetch_and_build.py` then copy to `data/`.")
    st.stop()

# Filters
col1, col2 = st.columns(2)
with col1:
    seasons = sorted(df["season"].dropna().unique().tolist()) if "season" in df.columns else []
    season = st.selectbox("Season", seasons, index=max(0, len(seasons)-1)) if seasons else None
with col2:
    weeks = sorted(df.loc[df["season"]==season, "week"].dropna().unique().tolist()) if season is not None else []
    week = st.selectbox("Week", weeks, index=0) if weeks else None

filt = df.copy()
if season is not None:
    filt = filt[filt["season"]==season]
if week is not None:
    filt = filt[filt["week"]==week]

tab_ml, tab_spreads = st.tabs(["Moneyline", "Spreads"])

with tab_ml:
    cols = [c for c in ["gameday","home_team","away_team","home_ml","away_ml","home_prob","away_prob",
                        "home_prob_raw","away_prob_raw"] if c in filt.columns]
    st.subheader("Moneylines (DraftKings preferred → otherwise median)")
    st.dataframe(filt[cols].sort_values(["gameday","home_team","away_team"]), use_container_width=True)

with tab_spreads:
    need = ["gameday","home_team","away_team","home_line","home_spread_odds","away_spread_odds"]
    have = [c for c in need if c in filt.columns]
    st.subheader("Point Spreads (home_line: negative = home favored)")
    if len(have) < 3:
        st.warning("No spreads available yet. Re-run `python scripts/fetch_and_build.py` after the spreads patch.")
    else:
        st.dataframe(
            filt[have].sort_values(["gameday","home_team","away_team"]),
            use_container_width=True
        )
