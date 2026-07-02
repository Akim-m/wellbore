"""Fit the two-stage residual model and predict full TVT vectors.

Stage 1 (pointwise): 3-seed HistGradientBoostingRegressor ensemble on per-point
features, residual path smoothed within the well.
Stage 2 (well-level): Ridge on well-summary features predicting the well's
n-weighted mean residual — 2/3 of pooled MSE is per-well bias, so this is the
big lever.
Prediction = anchor + W_PT * smooth(gbm) + W_WELL * ridge_const. Weights are
the OLS stack fit on 5-fold OOF (see HANDOFF); honest OOF pooled RMSE 15.130
vs 15.910 hold-TVT.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import model
from eda import ps_index
from features import well_features, well_summary
from model import load_well
from progress import every, log

MIN_PS = 50    # too little pre-PS context -> hold-TVT fallback
SUB = 5        # keep 1-in-SUB post-PS points when pooling training data
RSMOOTH = 301  # smooth the predicted residual path within each well (points)
SEEDS = (0, 1, 2)
W_PT, W_WELL = 0.467, 0.560   # stack weights (full-OOF OLS)
RIDGE_ALPHA = 10.0

# heavier regularization beats the default under pooled RMSE
HP = dict(max_iter=200, learning_rate=0.05, l2_regularization=5.0,
          min_samples_leaf=1000)


def fit_model(wells, feat_kw=None, hp=None):
    feat_kw = feat_kw or {}
    total = len(wells)
    log(f"fit: building features from {total} train wells {feat_kw}")
    Xs, ys, S, sy, sn = [], [], [], [], []
    for n, w in enumerate(wells, 1):
        hw, tw = load_well("train", w)
        if not (MIN_PS <= ps_index(hw) < len(hw)):
            continue
        X, names, label, ps, anchor = well_features(hw, tw, with_label=True, **feat_kw)
        Xs.append(X[::SUB])
        ys.append(label[::SUB])
        S.append(well_summary(hw, tw, X, names, ps, anchor))
        sy.append(float(label.mean()))
        sn.append(len(label))
        if every(n, total):
            log(f"fit: {n}/{total} wells, {sum(len(a) for a in Xs)} rows")
    X, y = np.vstack(Xs), np.concatenate(ys)

    log(f"fit: training {len(SEEDS)}-seed GBM ensemble on {X.shape[0]} rows x {X.shape[1]} feats")
    gbms = [HistGradientBoostingRegressor(**{**HP, **(hp or {}), "random_state": s}).fit(X, y)
            for s in SEEDS]

    # well-bias ridge: impute with train medians, standardize, weight by points
    S = np.vstack(S)
    med = np.nanmedian(S, axis=0)
    S = np.where(np.isfinite(S), S, med)
    mu, sd = S.mean(0), S.std(0) + 1e-9
    ridge = Ridge(alpha=RIDGE_ALPHA).fit((S - mu) / sd, np.array(sy),
                                         sample_weight=np.array(sn, dtype=float))
    log("fit: done (GBM ensemble + well-bias ridge)")
    return dict(gbms=gbms, ridge=ridge, med=med, mu=mu, sd=sd)


def predict_well(bundle, hw, tw, feat_kw=None):
    out = hw["TVT_input"].to_numpy().copy().astype(float)
    ps = ps_index(hw)
    if ps >= len(hw):
        return out
    anchor = out[ps - 1]
    if ps < MIN_PS:
        out[ps:] = anchor
        return out
    X, names, _, _, _ = well_features(hw, tw, with_label=False, **(feat_kw or {}))
    raw = np.mean([g.predict(X) for g in bundle["gbms"]], axis=0)
    r = model._smooth(np.clip(raw, -model.BAND, model.BAND), RSMOOTH)

    s = well_summary(hw, tw, X, names, ps, anchor)
    s = np.where(np.isfinite(s), s, bundle["med"])
    c = float(bundle["ridge"].predict(((s - bundle["mu"]) / bundle["sd"])[None, :])[0])

    out[ps:] = anchor + W_PT * r + W_WELL * c
    return out
