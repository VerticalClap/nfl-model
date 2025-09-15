# streamlit_app.py  —  NFL Picks Dashboard (public-repo mode)
import os, io, requests, pandas as pd, numpy as np, streamlit as st

# ===== repo settings (edit if you renamed) =====
REPO_OWNER = "VerticalClap"
REPO_NAME  = "nfl-model"
BRANCH     = "main"

REMOTE_BASE = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/data"
LOCAL_DATA  = "./data"         # fallback if remote not reachable
CACHE_TTL   = 60               # seconds; auto-refresh CSVs every minute

# ---------- helpers ----------
def moneyline_to_prob(ml):
    if pd.isna(ml): return np.nan
    ml = float(ml)
    return 100/(ml+100) if ml >= 0 else (-ml)/((-ml)+100)

@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def load_csv(name: str) -> pd.DataFrame:
    # try GitHub raw first (public), then local ./data
    url = f"{REMOTE_BASE}/{name}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content), low_memory=False)
    except Exception:
        p = os.path.join(LOCAL_DATA, name)
        return pd.read_csv(p, low_memory=False) if os.path.exists(p) else pd.DataFrame()

# ---------- UI ----------
st.set_page_config(page_title="NFL Picks — Live Sheet", layout="wide")
st.title("🏈 NFL Picks — Live Sheet")

# quick manual reload
if st.button("Reload data now"):
    st.cache_data.clear()

ps = load_csv("pick_sheet.csv")
sched = load_csv("schedule.csv")

if ps.empty:
    st.warning("No pick_sheet.csv in /data yet. Make sure the GitHub Action ran and pushed /data.")
    st.stop()

# normalize datatypes
if "gameday" in ps.columns:
    ps["gameday"] = pd.to_datetime(ps["gameday"], errors="coerce")

# derive implied probability columns if moneylines present
if "home_ml" in ps.columns and "home_prob" not in ps.columns:
    ps["home_prob_implied"] = ps["home_ml"].apply(moneyline_to_prob)
if "away_ml" in ps.columns and "away_prob" not in ps.columns:
    ps["away_prob_implied"] = ps["away_ml"].apply(moneyline_to_prob)

# compute edges if both model/fair probs and implied probs exist
for side in ["home","away"]:
    mp = f"{side}_prob"             # fair / de-vig prob from pipeline
    ip = f"{side}_prob_implied"     # implied from moneyline
    if mp in ps.columns and ip in ps.columns:
        ps[f"{side}_edge"] = ps[mp] - ps[ip]

# filters
left, right = st.columns([3,2])
with left:
    st.subheader("Filters")
    seasons = sorted([int(x) for x in ps["season"].dropna().unique().tolist()]) if "season" in ps else []
    default_season = max(seasons) if seasons else None
    season = st.selectbox("Season", seasons, index=seasons.index(default_season) if seasons else 0)

    week_opts = sorted(ps.loc[ps["season"]==season, "week"].dropna().unique().tolist()) if season is not None else []
    week = st.selectbox("Week", week_opts, index=len(week_opts)-1 if week_opts else 0)

    view = ps.copy()
    if season is not None: view = view[view["season"]==season]
    if week is not None:   view = view[view["week"]==week]

    # show main table
    show_cols = [c for c in [
        "season","week","gameday","home_team","away_team",
        "home_ml","away_ml",
        "home_prob","away_prob","home_prob_implied","away_prob_implied",
        "home_edge","away_edge",
        "home_kelly_5pct","away_kelly_5pct",
        "temp_f","wind_mph","gust_mph","precip_prob","qb_out","ol_starters_out","db_starters_out"
    ] if c in view.columns]

    st.subheader("Pick Sheet")
    st.dataframe(
        view[show_cols].sort_values(
            by=[c for c in ["season","week","gameday","home_edge"] if c in show_cols],
            ascending=[True, True, True, False] if "home_edge" in show_cols else True
        ),
        use_container_width=True
    )

with right:
    st.subheader("Calibration snapshot")
    # If you later log model preds vs actuals, this will populate
    if {"pred","actual"}.issubset(ps.columns):
        bins = pd.cut(ps["pred"], bins=np.linspace(0,1,11))
        calib = ps.groupby(bins)["actual"].mean().rename("empirical").to_frame()
        calib["count"] = ps.groupby(bins).size()
        st.dataframe(calib, use_container_width=True)
    else:
        st.info("Calibration will display once predictions vs outcomes are logged.")

st.caption("Data auto-loads from the repo’s /data folder on GitHub (auto-refreshed every 60s).")
