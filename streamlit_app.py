import os, json, pandas as pd, numpy as np, streamlit as st

CACHE = os.environ.get("DATA_CACHE_DIR", "./cache")

@st.cache_data(show_spinner=False)
def load_pick_sheet():
    p = os.path.join(CACHE, "pick_sheet.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    # canonical columns if missing
    for c in ["home_prob","away_prob","home_ml","away_ml","gameday","season","week","home_team","away_team"]:
        if c not in df.columns:
            df[c] = np.nan
    return df

@st.cache_data(show_spinner=False)
def load_schedule():
    p = os.path.join(CACHE, "schedule.csv")
    return pd.read_csv(p, low_memory=False) if os.path.exists(p) else pd.DataFrame()

st.set_page_config(page_title="NFL Picks Dashboard", layout="wide")
st.title("🏈 NFL Picks — Live Sheet")

ps = load_pick_sheet()
sched = load_schedule()

if ps.empty:
    st.warning("No pick_sheet.csv yet. Run the GitHub Action or `python scripts/fetch_and_build.py`.")
    st.stop()

left, right = st.columns([3,2])

with left:
    st.subheader("Filters")
    seasons = sorted(ps["season"].dropna().unique().tolist())
    season = st.selectbox("Season", seasons[-1:] + seasons[:-1], index=0) if seasons else None
    week_opts = sorted(ps.loc[ps["season"]==season,"week"].dropna().unique().tolist()) if season else []
    week = st.selectbox("Week", week_opts, index=len(week_opts)-1 if week_opts else 0) if week_opts else None

    view = ps.copy()
    if season is not None: view = view[view["season"]==season]
    if week is not None: view = view[view["week"]==week]

    # Edge columns (vs implied, if ML present)
    def ml_to_prob(ml):
        if pd.isna(ml): return np.nan
        ml=float(ml); return 100/(ml+100) if ml>=0 else (-ml)/((-ml)+100)

    view["home_prob_implied"] = view["home_ml"].apply(ml_to_prob)
    view["away_prob_implied"] = view["away_ml"].apply(ml_to_prob)
    for side in ["home","away"]:
        if f"{side}_prob" in view.columns:
            view[f"{side}_edge"] = view[f"{side}_prob"] - view[f"{side}_prob_implied"]

    show_cols = [c for c in [
        "season","week","gameday","home_team","away_team",
        "home_ml","away_ml","home_prob","away_prob","home_edge","away_edge",
        "home_kelly_5pct","away_kelly_5pct",
        "temp_f","wind_mph","gust_mph","precip_prob","qb_out","ol_starters_out","db_starters_out"
    ] if c in view.columns]

    st.subheader("Pick Sheet")
    st.dataframe(view[show_cols].sort_values(["season","week","gameday","home_edge"], ascending=[True,True,True,False]), use_container_width=True)

with right:
    st.subheader("Calibration snapshot")
    # Simple reliability-style view if historical predictions exist
    if {"pred","actual"}.issubset(view.columns):
        bins = pd.cut(view["pred"], bins=np.linspace(0,1,11))
        calib = view.groupby(bins)["actual"].mean().rename("empirical").to_frame()
        calib["count"] = view.groupby(bins).size()
        st.dataframe(calib, use_container_width=True)
    else:
        st.info("Calibration will show once we train and log predictions with outcomes.")

st.caption("Tip: as lines shift during the week, watch the edge and Kelly columns move.")
