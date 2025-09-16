# streamlit_app.py
# NFL Picks — Live Sheet
# - Moneyline: market + model probabilities (vig-removed), Kelly
# - Spread / ATS: market spread (from market prob), model spread (from model prob),
#   model cover probability, edge, Kelly at -110
# - Totals: placeholder (we’ll wire real totals once we add them to the pipeline)

import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------
# Utilities: standard normal CDF / inverse CDF (no SciPy needed)
# -------------------------------------------------------------
_SQRT2 = math.sqrt(2.0)

def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))

# Probit approximation (Acklam's approximation)
# Source: https://web.archive.org/web/20150910044702/http://home.online.no/~pjacklam/notes/invnorm/
def _ppf(p: float) -> float:
    """Approximate inverse standard normal CDF (probit). p in (0, 1)."""
    if p <= 0.0:
        return -1e9
    if p >= 1.0:
        return 1e9

    # Coefficients
    a1 = -3.969683028665376e+01
    a2 =  2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 =  1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 =  2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 =  1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 =  6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 =  4.374664141464968e+00
    c6 =  2.938163982698783e+00

    d1 =  7.784695709041462e-03
    d2 =  3.224671290700398e-01
    d3 =  2.445134137142996e+00
    d4 =  3.754408661907416e+00

    # Break-points
    plow  = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    elif p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
               (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)

def _clip01(x: pd.Series, eps: float = 1e-6) -> pd.Series:
    return x.clip(eps, 1 - eps)

# -------------------------------------------------------------
# Conversions
# -------------------------------------------------------------
def american_to_prob(odds: pd.Series) -> pd.Series:
    """American odds -> implied probability (no vig removal)."""
    o = odds.astype(float)
    pos = o > 0
    neg = ~pos
    p = pd.Series(np.nan, index=o.index, dtype=float)
    p[pos] = 100.0 / (o[pos] + 100.0)
    p[neg] = (-o[neg]) / ((-o[neg]) + 100.0)
    return p

def remove_vig(p_home: pd.Series, p_away: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Proportional vig removal so p_home + p_away = 1."""
    s = p_home + p_away
    s = s.replace({0: np.nan})
    return (p_home / s, p_away / s)

# -------------------------------------------------------------
# Spread math
# -------------------------------------------------------------
SD_MARGIN = 13.5  # std dev of NFL scoring margin used for the probit mapping

def prob_to_home_line(prob_home: pd.Series) -> pd.Series:
    """
    Convert home win probability to a 'home line' (spread) in points.
    By convention: negative line means HOME is favored.
    Derivation: P(home_margin > 0) = Φ(-line / SD) => line = -SD * Φ^{-1}(P)
    """
    p = _clip01(prob_home)
    return -SD_MARGIN * p.apply(_ppf)

def home_cover_prob(mu: pd.Series, line_home: pd.Series) -> pd.Series:
    """
    Model probability that the HOME team covers the market line.
    μ (mu) is model mean margin for HOME (μ = SD * Φ^{-1}(P_home_model)).
    If line_home = -3, cover event is (margin > 3).
    P(cover) = 1 - Φ((T - μ)/SD) where T = -line_home (positive threshold).
    """
    T = -line_home  # threshold home must exceed
    z = (T - mu) / SD_MARGIN
    return 1.0 - z.apply(_phi)

def kelly_fraction(p: pd.Series, price: float = -110.0) -> pd.Series:
    """
    Kelly for ATS at given price (default -110).
    b is net odds in decimal: e.g., -110 -> b ≈ 0.9091 (risk 1 to win 0.9091).
    f = (b*p - (1-p)) / b, clipped at [0, 1].
    """
    if price < 0:
        b = 100.0 / (-price)
    else:
        b = price / 100.0
    f = (b * p - (1.0 - p)) / b
    return f.clip(lower=0.0)

# -------------------------------------------------------------
# Data loading
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_pick_sheet() -> pd.DataFrame:
    # Prefer committed /data (Actions output); fall back to local cache for dev
    try:
        df = pd.read_csv("data/pick_sheet.csv")
    except Exception:
        df = pd.read_csv("cache/pick_sheet.csv")
    # Clean types
    if "gameday" in df.columns:
        try:
            df["gameday"] = pd.to_datetime(df["gameday"])
        except Exception:
            pass
    # Normalize a few team codes just in case
    for col in ("home_team", "away_team"):
        if col in df.columns:
            df[col] = df[col].replace({"LA": "LAR", "SD": "LAC", "OAK": "LV"})
    return df

# -------------------------------------------------------------
# Layout
# -------------------------------------------------------------
st.set_page_config(page_title="NFL Picks — Live Sheet", layout="wide")
st.title("🏈 NFL Picks — Live Sheet")

df = load_pick_sheet()

# Top filters
left, mid, right = st.columns([1, 1, 2])

with left:
    season_list = sorted(df["season"].dropna().unique().tolist()) if "season" in df.columns else []
    season = st.selectbox("Season", season_list, index=len(season_list)-1 if season_list else 0)

with mid:
    # Only weeks for selected season
    wk_list = sorted(df.loc[df["season"] == season, "week"].dropna().unique().tolist()) if "week" in df.columns else []
    default_week = wk_list[0] if wk_list else None
    week = st.selectbox("Week", wk_list, index=wk_list.index(default_week) if default_week in wk_list else 0)

with right:
    all_teams = sorted(pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True)).dropna().tolist())
    team_filter = st.selectbox("Filter by team (optional)", ["(All)"] + all_teams, index=0)

# Filter dataframe
mask = (df["season"] == season) & (df["week"] == week)
if team_filter != "(All)":
    mask &= (df["home_team"].eq(team_filter) | df["away_team"].eq(team_filter))
d = df.loc[mask].copy()
d = d.sort_values("gameday", kind="stable")

st.caption("Columns refresh automatically from the repo’s `/data/pick_sheet.csv` (auto-refresh ~60s).")

tab_ml, tab_spread, tab_totals = st.tabs(["Moneyline (Model vs Market)", "Spread / ATS", "Totals (O/U)"])

# -------------------------------------------------------------
# Moneyline tab
# -------------------------------------------------------------
with tab_ml:
    cols_ml = [
        "gameday", "home_team", "away_team",
        "home_ml", "away_ml",
        "home_prob", "away_prob",
        "home_prob_raw", "away_prob_raw",
        "home_prob_model", "away_prob_model",
        "home_kelly_5pct", "away_kelly_5pct",
    ]
    avail = [c for c in cols_ml if c in d.columns]
    if not avail:
        st.info("No moneyline columns are present yet in pick_sheet.csv.")
    else:
        st.subheader("Moneyline — Market prices & vig-removed fair probabilities")
        st.dataframe(d[avail], use_container_width=True)

# -------------------------------------------------------------
# Spread / ATS tab
# -------------------------------------------------------------
with tab_spread:
    st.subheader("Spread / ATS — Market vs model edges")

    dd = d.copy()

    # We need market & model win probabilities:
    # - home_prob: market fair prob (vig-removed). If missing, derive from ML odds.
    # - home_prob_model: your model prob. If missing, fall back to market for display (edge=0).
    if "home_prob" not in dd.columns or dd["home_prob"].isna().all():
        if {"home_ml", "away_ml"}.issubset(dd.columns):
            ph = american_to_prob(dd["home_ml"])
            pa = american_to_prob(dd["away_ml"])
            ph_fair, _ = remove_vig(ph, pa)
            dd["home_prob"] = ph_fair
        else:
            dd["home_prob"] = np.nan

    if "home_prob_model" not in dd.columns:
        dd["home_prob_model"] = dd["home_prob"]  # neutral fallback

    # Compute spreads:
    # market_spread from market fair prob; model_spread from model prob
    dd["market_spread"] = prob_to_home_line(dd["home_prob"])
    dd["model_spread"]  = prob_to_home_line(dd["home_prob_model"])

    # Model mean margin μ (HOME). Used for cover probability vs market line
    mu = SD_MARGIN * _clip01(dd["home_prob_model"]).apply(_ppf)  # μ = SD * Φ^{-1}(P_home_model)

    # Model probability that HOME covers the market line:
    dd["home_cover_prob_model"] = home_cover_prob(mu, dd["market_spread"])

    # Edge in points (model − market). Negative => model favors HOME more than market does.
    dd["spread_edge"] = dd["model_spread"] - dd["market_spread"]

    # Kelly fraction at -110 (as percent)
    dd["kelly_pct_spread"] = 100.0 * kelly_fraction(dd["home_cover_prob_model"], price=-110.0)

    # Pretty rounding for display
    show = dd[[
        "gameday", "home_team", "away_team",
        "market_spread", "model_spread", "spread_edge",
        "home_cover_prob_model", "kelly_pct_spread"
    ]].copy()

    for c in ("market_spread", "model_spread", "spread_edge"):
        show[c] = show[c].round(1)
    show["home_cover_prob_model"] = (show["home_cover_prob_model"] * 100).round(1)
    show["kelly_pct_spread"] = show["kelly_pct_spread"].round(1)

    st.dataframe(show, use_container_width=True)
    st.caption("Notes: spreads are HOME lines (negative => home favored). Market spread is derived from market win probability if no explicit spread prices are present. Kelly is computed for -110 ATS.")

# -------------------------------------------------------------
# Totals tab (placeholder)
# -------------------------------------------------------------
with tab_totals:
    st.info("Totals (O/U) coming soon — as soon as we add totals to the pipeline.")
