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
    for base in (DATA_DIR, CACHE_DIR):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return pd.read_csv(p, low_memory=False)
    return None

df = load_csv("pick_sheet.csv")
if df is None or df.empty:
    st.info("No pick_sheet.csv yet. Run `python scripts/fetch_and_build.py`, then copy into /data.")
    st.stop()

# ---------- Filters ----------
top = st.container()
with top:
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        seasons = sorted(df["season"].dropna().unique().tolist()) if "season" in df.columns else []
        season = st.selectbox("Season", seasons, index=max(0, len(seasons)-1)) if seasons else None
    with c2:
        if season is not None and "week" in df.columns:
            weeks = sorted(df.loc[df["season"] == season, "week"].dropna().unique().tolist())
        else:
            weeks = []
        week = st.selectbox("Week", weeks, index=0) if weeks else None
    with c3:
        # Build team list robustly (dropna on Series, not on numpy array)
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
if team_filter != "(All)" and {"home_team","away_team"}.issubset(filt.columns):
    filt = filt[(filt["home_team"] == team_filter) | (filt["away_team"] == team_filter)]

tab_ml, tab_spreads = st.tabs(["Moneyline", "Spreads"])

# ---------- Moneyline ----------
with tab_ml:
    st.subheader("Moneylines (book vs. model when available)")

    # We’ll only show columns that actually exist to avoid breaking the view
    preferred_order = [
        "gameday","home_team","away_team",
        "home_ml","away_ml",
        "home_prob","away_prob",            # book (vig-removed) probs if present
        "home_prob_raw","away_prob_raw",    # raw probs before vig removal (optional)
        "home_prob_model","away_prob_model",# your model probs (optional)
        "home_edge_model",                  # model edge vs book (optional)
        "home_kelly_5pct","away_kelly_5pct" # Kelly (optional)
    ]
    cols = [c for c in preferred_order if c in filt.columns]
    if cols:
        st.dataframe(
            filt[cols].sort_values(
                [c for c in ["gameday","home_team","away_team"] if c in filt.columns],
                na_position="last"
            ),
            use_container_width=True
        )
    else:
        st.warning("Moneyline columns aren’t present in pick_sheet.csv right now.")

# ---------- Spreads ----------
with tab_spreads:
    st.subheader("Point Spreads (book vs. model when available)")
    preferred_order = [
        "gameday","home_team","away_team",
        "home_line","home_spread_odds","away_spread_odds",   # book spread & prices
        "model_spread",                                      # your model spread (home negative = home favored)
        "spread_edge_model",                                 # model edge vs book spread
        "kelly_spread_home","kelly_spread_away"              # optional Kelly sizing
    ]
    cols = [c for c in preferred_order if c in filt.columns]
    if cols:
        st.dataframe(
            filt[cols].sort_values(
                [c for c in ["gameday","home_team","away_team"] if c in filt.columns],
                na_position="last"
            ),
            use_container_width=True
        )
    else:
        st.info("No spread columns found. Rebuild data with spreads and refresh.")
