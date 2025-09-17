# nfl_model/modeling.py
from __future__ import annotations
import numpy as np
import pandas as pd
from math import erf, sqrt
from typing import Tuple, List

# ---- helpers ----

def _safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:
        pass

def norm_cdf(x: np.ndarray) -> np.ndarray:
    # standard normal CDF via erf
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

# ---- model ----

FEATURES: List[str] = [
    # primary differentials we created
    "x_epa_off", "x_sr_off", "x_epa_def", "x_sr_def",
    # small bias terms to help early-season when rolls are sparse
    "home_off_epp8", "away_off_epp8"
]

def _design_matrix(df: pd.DataFrame, with_bias: bool = True) -> np.ndarray:
    cols = [c for c in FEATURES if c in df.columns]
    X = df[cols].fillna(0.0).to_numpy(dtype=float)
    if with_bias:
        X = np.column_stack([np.ones(len(X)), X])
    return X

def train_spread_model(train_df: pd.DataFrame) -> Tuple[np.ndarray, float, List[str]]:
    """
    Fit y = Xb on home_margin using least squares. Return (beta, sigma, used_features).
    sigma is the residual std dev -> used to turn spread into win probability.
    """
    if train_df.empty or "home_margin" not in train_df.columns:
        _safe_print("[model] No training data; using zero coefficients.")
        beta = np.zeros(1 + len(FEATURES))
        return beta, 10.5, FEATURES  # fallback sigma ~ typical NFL score spread std

    y = train_df["home_margin"].to_numpy(dtype=float)
    X = _design_matrix(train_df, with_bias=True)

    # Least squares
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    # residual sigma
    resid = y - X @ beta
    dof = max(1, len(y) - len(beta))
    sigma = float(np.sqrt(np.sum(resid**2) / dof))
    sigma = max(sigma, 6.0)  # sanity floor

    used = [c for c in FEATURES if c in train_df.columns]
    _safe_print(f"[model] trained with {len(y)} games | sigma≈{sigma:.2f}")
    return beta, sigma, used

def predict_spread_and_prob(beta: np.ndarray, sigma: float, feat_up: pd.DataFrame) -> pd.DataFrame:
    """
    Produce model_spread (home margin) and model_home_prob for upcoming.
    """
    if feat_up.empty:
        return pd.DataFrame(columns=["game_id", "model_spread", "model_home_prob"])

    Xup = _design_matrix(feat_up, with_bias=True)
    pred_margin = Xup @ beta
    # win prob = P(margin > 0) under Normal(pred_margin, sigma)
    z = pred_margin / max(sigma, 1e-6)
    win_prob = norm_cdf(z)

    out = feat_up[["game_id"]].copy()
    out["model_spread"] = pred_margin
    out["model_home_prob"] = win_prob
    return out

def attach_model_outputs(
    train_df: pd.DataFrame,
    up_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Train on train_df and predict for up_df. Returns DataFrame with
    ['game_id','model_spread','model_home_prob'].
    """
    beta, sigma, used = train_spread_model(train_df)
    _safe_print(f"[model] features used: {used}")
    preds = predict_spread_and_prob(beta, sigma, up_df)
    return preds
