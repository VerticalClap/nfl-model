import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression

def train_with_timeseries_cv(X, y, n_splits=5, random_state=42):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds = np.zeros(len(y), dtype=float)
    models = []
    for i, (tr, va) in enumerate(tscv.split(X)):
        m = LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state+i
        )
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:,1]
        preds[va] = p
        models.append(m)
    return models, preds

def calibrate(models, X_valid, y_valid):
    pred = np.mean([m.predict_proba(X_valid)[:,1] for m in models], axis=0)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pred, y_valid)
    return iso

def predict_ensemble(models, X):
    return np.mean([m.predict_proba(X)[:,1] for m in models], axis=0)
