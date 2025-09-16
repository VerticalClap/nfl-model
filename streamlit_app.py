# streamlit_app.py
import os
import time
import pandas as pd
import streamlit as st

DATA_PATH = os.path.join("data", "pick_sheet.csv")

st.set_page_config(page_title="NFL Picks — Live Sheet", layout="wide")
st.title("🏈 NFL Picks — Live Sheet")

@st.cache_data(ttl=60)
def load_df():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, low_memory=False)
    # basic cleaning
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    return df

df = load_df()
if df.empty:
    st.warning("No pick_sheet.csv yet. Run the GitHub Action or `python scripts/fetch_and_build.py`.")
    st.stop()

# --- Filters ---
col1,col2,col3 = st.columns([1,1,2])
with col1:
    season = st.selectbox("Season", sorted(df["season"].dropna().unique()), index=len(sorted(df["season"].unique()))-1)
with col2:
    weeks = sorted(df.loc[df["season"]==season,"week"].dropna().unique())
    week = st.selectbox("Week", weeks, index=0)
with col3:
    teams = ["(All)"] + sorted(pd.unique(pd.concat([df["home_team"], df["away_team"]]).dropna()))
    team_filter = st.selectbox("Filter by team (optional)", teams, index=0)

base = df[(df["season"]==season) & (df["week"]==week)].copy()
if team_filter != "(All)":
    base = base[(base["home_team"]==team_filter) | (base["away_team"]==team_filter)]

st.caption("Columns refresh automatically from the repo’s /data/pick_sheet.csv (auto-refresh ~60s).")

tabs = st.tabs(["Moneyline (Model vs Market)", "Spread / ATS", "Totals (O/U)"])

# -------------------------------------------------------------------
# MONEYLINE TAB
# -------------------------------------------------------------------
with tabs[0]:
    st.subheader("Moneyline — Market prices & vig-removed fair probabilities")

    cols_to_show = [
        "gameday", "home_team", "away_team",
        "home_ml", "away_ml",
        "home_prob", "away_prob",            # vig-removed market fair prob
        "home_prob_raw", "away_prob_raw",    # pre-vig implied prob
        "home_prob_model", "away_prob_model",
        "home_kelly_5pct", "away_kelly_5pct"
    ]
    avail = [c for c in cols_to_show if c in base.columns]
    if not avail:
        st.info("No moneyline columns found yet in pick_sheet.csv.")
    else:
        st.dataframe(base[avail].sort_values("gameday"), use_container_width=True)

    st.caption(
        "Tip: if you want **DraftKings-only** numbers vs **consensus** (median across books), "
        "edit the `books` list inside the merge step in `nfl_model/pipeline.py` to `['draftkings']`."
    )

# -------------------------------------------------------------------
# SPREAD / ATS TAB (simple placeholder – will populate after you add spreads to pick_sheet)
# -------------------------------------------------------------------
with tabs[1]:
    st.subheader("Spread / ATS — Market vs model edges")
    cols_spread = [
        "gameday","home_team","away_team",
        "spread","home_prob_model","away_prob_model",
        "home_edge","away_edge","home_kelly_5pct","away_kelly_5pct"
    ]
    avail2 = [c for c in cols_spread if c in base.columns]
    if avail2:
        st.dataframe(base[avail2].sort_values("gameday"), use_container_width=True)
    else:
        st.info("Spread/ATS fields will appear once spreads are added to pick_sheet.")

# -------------------------------------------------------------------
# TOTALS TAB (placeholder for when you add totals data)
# -------------------------------------------------------------------
with tabs[2]:
    st.subheader("Totals (O/U) — Coming soon")
    st.info("Once totals are added to pick_sheet, this tab will populate.")

# Manual reload button
if st.button("Reload data now"):
    load_df.clear()
    st.experimental_rerun()
