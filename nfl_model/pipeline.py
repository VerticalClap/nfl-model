from .features import build_feature_matrix
from .modeling import train_with_timeseries_cv, calibrate, predict_ensemble

def train_calibrated_model(df, target="win"):
    X = build_feature_matrix(df)
    y = df[target].astype(int)
    models, _ = train_with_timeseries_cv(X, y, n_splits=5, random_state=42)
    cut = int(len(X)*0.8)
    iso = calibrate(models, X.iloc[cut:], y.iloc[cut:])
    return {"models": models, "iso": iso}

def infer_probs(bundle, X):
    raw = predict_ensemble(bundle["models"], X)
    return bundle["iso"].transform(raw)
