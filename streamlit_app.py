# streamlit_app.py
from __future__ import annotations

import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="🏈 NFL Picks — Live Sheet", layout="wide")
st.title("🏈 NFL Picks — Live Sheet")

DATA_DIR = "data"
CACHE_DIR = "cache"

def load_csv(name: str) -> pd.DataFrame | None:
    """
    Prefer data/ then fall back to cache/.
    """
    p1 = os.path.join(DATA_DIR, name)
    if os.path.exists(p1):
        return pd.read_csv(p1, low_memory=False)

    p2 = os.path.join(CACHE_DIR, name)
    if os.path.exists(p2):
        return pd.read_csv(p2, low_memory=False)

    return None


# ---------------------------
# Load the pick sheet
# ---------------------------
df = load_csv("pick_sheet.csv")
if df is None or df.empty:
    st.info(
        "No pick_sheet.csv yet. Run `python scripts\\fetch_and_build.py`, "
        "then copy the output from `cache/` to `data/`, or wait for Actions."
    )
    st.stop()

# Normalize a few columns that might arrive as strings
for c in ("gameday",):
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# ---------------------------
# Filters
# ---------------------------
c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    seasons = (
        sorted(df["season"].dropna().astype(int).unique().tolist())
        if "season" in df.columns else []
    )
    season = st.selectbox("Season", seasons, index=max(0, len(seasons) - 1)) if seasons else None

with c2:
    if season is not None and "week" in df.columns and "season" in df.columns:
        weeks = sorted(df.loc[df["season"] == season, "week"].dropna().astype(int).unique().tolist())
    else:
        weeks = []
    week = st.selectbox("Week", weeks, index=0) if weeks else None

with c3:
    # ✅ FIX: build team list as Series -> dropna() -> unique()
    home = df.get("home_team", pd.Series(dtype="object"))
    away = df.get("away_team", pd.Series(dtype="object"))
    teams_series = pd.concat([home, away], ignore_index=True)
    all_teams = sorted(teams_series.dropna().astype(str).unique().tolist())
    team_filter = st.selectbox("Filter by team (optional)", ["(All)"] + all_teams, index=0)

# Apply filters
filt = df.copy()
if season is not None and "season" in filt.columns:
    filt = filt[filt["season"] == season]
if week is not None and "week" in filt.columns:
    filt = filt[filt["week"] == week]
if team_filter != "(All)" and {"home_team", "away_team"}.issubset(filt.columns):
    filt = filt[(filt["home_team"] == team_filter) | (filt["away_team"] == team_filter)]

st.caption("Columns refresh automatically from the repo’s `/data/pick_sheet.csv` (auto-refresh ~60s).")

# ---------------------------
# Tabs
# ---------------------------
tab_ml, tab_spreads = st.tabs(["Moneyline (Model vs Market)", "Spread / ATS"])

with tab_ml:
    st.subheader("Moneyline — Market prices & vig-removed fair probabilities")

    # Show whatever of these columns exist
    money_cols_pref = [
        "gameday", "home_team", "away_team",
        "home_ml", "away_ml",
        "home_prob", "away_prob",
        "home_prob_raw", "away_prob_raw",
        # optional model columns if/when you add them
        "home_prob_model", "away_prob_model",
        "home_kelly_5pct", "away_kelly_5pct",
    ]
    money_cols = [c for c in money_cols_pref if c in filt.columns]

    if money_cols:
        st.dataframe(
            filt[money_cols].sort_values(
                [c for c in ["gameday", "home_team", "away_team"] if c in filt.columns],
                na_position="last"
            ),
            use_container_width=True
        )
    else:
        st.warning("Moneyline columns not found yet. Rebuild data or check `pick_sheet.csv`.")

with tab_spreads:
    st.subheader("Spread / ATS — Market vs model edges")

    # Expecting these columns after spreads extraction
    spread_cols_pref = [
        "gameday", "home_team", "away_team",
        "home_line", "home_spread_odds", "away_spread_odds",
        # optional model columns if/when you add them
        "model_spread", "edge_points", "edge_pct",
    ]
    spread_cols = [c for c in spread_cols_pref if c in filt.columns]

    if spread_cols:
        st.dataframe(
            filt[spread_cols].sort_values(
                [c for c in ["gameday", "home_team", "away_team"] if c in filt.columns],
                na_position="last"
            ),
            use_container_width=True
        )
    else:
        st.warning(
            "Spreads not available. Ensure your data includes `home_line`, "
            "`home_spread_odds`, `away_spread_odds` (and optional model columns)."
        )

st.button("Reload data now", on_click=lambda: st.experimental_rerun())
