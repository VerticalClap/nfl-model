import os, io, requests, pandas as pd, numpy as np, streamlit as st

# ===== CONFIG =====
REPO_OWNER = "VerticalClap"
REPO_NAME  = "nfl-model"
BRANCH     = "main"

REMOTE_BASE = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/data"
LOCAL_CACHE = os.environ.get("DATA_CACHE_DIR", "./cache")

# ===== Data loaders =====
@st.cache_data(show_spinner=False, ttl=60)  # auto-refresh every 60s
def load_csv(path: str) -> pd.DataFrame:
    ...

    url = f"{REMOTE_BASE}/{path}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content), low_memory=False)
    except Exception:
        # fallback local
        p = os.path.join(LOCAL_CACHE, path)
        return pd.read_csv(p, low_memory=False) if os.path.exists(p) else pd.DataFrame()

def moneyline_to_prob(ml):
    if pd.isna(ml): return np.nan
    ml = float(ml)
    return 100/(ml+100) if ml >= 0 else (-ml)/((-ml)+100)

# ===== UI =====
st.set_page_config(page_title="NFL Picks Dashboard", layout="wide")
st.title("🏈 NFL Picks — Live Sheet")

ps = load_csv("pick_sheet.csv")
sched = load_csv("schedule.csv")

if ps.empty:
    st.warning("No pick_sheet.csv found yet. Wait for GitHub Actions to finish (data/ folder), or re-run workflow.")
    st.stop()

left, right = st.columns([3,2])

with left:
    st.subheader("Filters")
    seasons = sorted([int(x) for x in ps["season"].dropna().unique().tolist()]) if "season" in ps.columns else []
    season = st.selectbox("Season", seasons[-1:] + seasons[:-1], index=0) if seasons else None

    week_opts = sorted(ps.loc[ps["season"]==season,"week"].dropna().unique().tolist()) if season else []
    week = st.selectbox("Week", week_opts, index=len(week_opts)-1 if week_opts else 0) if week_opts else None

    view = ps.copy()
    if season is not None: view = view[view["season"]==season]
    if week is not None: view = view[view["week"]==week]

    # Edge vs implied
    if "home_ml" in view:
        view["home_prob_implied"] = view["home_ml"].apply(moneyline_to_prob)
    if "away_ml" in view:
        view["away_prob_implied"] = view["away_ml"].apply(moneyline_to_prob)
    for side in ["home","away"]:
        ip = f"{side}_prob_implied"
        mp = f"{side}_prob"
        if ip in view and mp in view:
            view[f"{side}_edge"] = view[mp] - view[ip]

    show_cols = [c for c in [
        "season","week","gameday","home_team","away_team",
        "home_ml","away_ml","home_prob","away_prob","home_edge","away_edge",
        "home_kelly_5pct","away_kelly_5pct",
        "temp_f","wind_mph","gust_mph","precip_prob","qb_out","ol_starters_out","db_starters_out"
    ] if c in view.columns]

    st.subheader("Pick Sheet")
    st.dataframe(
        view[show_cols].sort_values(["season","week","gameday","home_edge"], ascending=[True,True,True,False]),
        use_container_width=True
    )

with right:
    st.subheader("Calibration snapshot")
    if {"pred","actual"}.issubset(view.columns):
        bins = pd.cut(view["pred"], bins=np.linspace(0,1,11))
        calib = view.groupby(bins)["actual"].mean().rename("empirical").to_frame()
        calib["count"] = view.groupby(bins).size()
        st.dataframe(calib, use_container_width=True)
    else:
        st.info("Calibration will display once predictions vs outcomes are logged.")

st.caption("Data auto-loads from the repo’s /data folder on GitHub.")
