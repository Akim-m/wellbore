"""OOF experiments over the feature cache. Usage: exp.py VARIANT [VARIANT...]

Each variant fits 5-fold GBMs on cached features and saves pooled OOF raw
residual predictions to oof_<variant>.npz for later post-processing/stacking.
Reports pooled RMSE (smooth 301, shrink swept) like cv.py for comparability.
"""
import glob
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
import model as dp
from eda import DATA
from progress import log

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = 5
HP = dict(max_iter=200, learning_rate=0.05, l2_regularization=5.0,
          min_samples_leaf=1000, random_state=0)

BASE = ["dmd", "dz", "lat_dist", "incl", "gr", "gr_res_anchor", "gr_offset",
        "pre_tvt_std", "pre_tvt_slope", "gr_grad", "gr_lag100", "gr_lag250",
        "gr_roll_std"]

VARIANTS = {
    "v3":        BASE,
    "align":     BASE + ["align_path", "align_delta", "align_slope", "align_ev"],
    "refslope":  BASE + ["ref_slope", "implied_off", "mean_gr_res"],
    "inter":     BASE + ["inter_slope_dmd", "dmd_frac"],
    "leads":     BASE + ["gr_lead100", "gr_lead250"],
    "all":       BASE + ["align_path", "align_delta", "align_slope", "align_ev",
                         "ref_slope", "implied_off", "mean_gr_res",
                         "inter_slope_dmd", "dmd_frac", "gr_lead100", "gr_lead250"],
}


def load_cache():
    # pickle is safe here: cache.pkl is generated locally by build_cache.py in this scratchpad
    with open(os.path.join(HERE, "cache.pkl"), "rb") as f:
        c = pickle.load(f)
    return c["names"], c["wells"]


def folds(cache):
    allw = sorted(os.path.basename(p).replace("__horizontal_well.csv", "")
                  for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))
    return [[w for w in allw[k::5] if w in cache] for k in range(5)]


def run(name, cols, names, cache, fold_lists, hp=None, seeds=(0,)):
    idx = [names.index(c) for c in cols]
    parts = {}   # well -> raw prediction
    for k, hold in enumerate(fold_lists):
        hset = set(hold)
        tr = [w for w in cache if w not in hset]
        Xtr = np.vstack([cache[w]["X"][::SUB][:, idx] for w in tr])
        ytr = np.concatenate([cache[w]["y"][::SUB] for w in tr])
        preds = []
        for s in seeds:
            reg = HistGradientBoostingRegressor(**{**HP, **(hp or {}), "random_state": s})
            reg.fit(Xtr, ytr)
            preds.append({w: reg.predict(cache[w]["X"][:, idx]) for w in hold})
        for w in hold:
            parts[w] = np.clip(np.mean([p[w] for p in preds], axis=0), -dp.BAND, dp.BAND)
        log(f"{name}: fold {k} done ({len(Xtr)} rows, {len(idx)} feats, {len(seeds)} seeds)")
    return parts


def pool(cache, parts):
    ws = sorted(parts)
    T = np.concatenate([cache[w]["y"] for w in ws])          # residual truth (TVT-anchor)
    R = np.concatenate([parts[w] for w in ws])
    W = np.concatenate([np.full(len(parts[w]), i) for i, w in enumerate(ws)])
    D = np.concatenate([cache[w]["dmd"] for w in ws])
    return T, R, W, D


def smooth_within(R, W, win=301):
    out = R.copy()
    for w in np.unique(W):
        m = W == w
        out[m] = dp._smooth(R[m], win)
    return out


def report(name, T, R, W):
    Rs = smooth_within(R, W)
    base = float(np.sqrt(np.mean(T ** 2)))
    best = (base, 0.0)
    for s in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        pr = float(np.sqrt(np.mean((T - s * Rs) ** 2)))
        if pr < best[0]:
            best = (pr, s)
    # decomposition of the best prediction's remaining error
    E = T - best[1] * Rs
    bias = dev = 0.0
    for w in np.unique(W):
        m = W == w
        b = E[m].mean()
        bias += b * b * m.sum()
        dev += ((E[m] - b) ** 2).sum()
    log(f"RESULT {name}: pooled={best[0]:.4f} ({best[0]-base:+.4f} vs hold {base:.4f}) "
        f"shrink={best[1]}  bias_share={bias/(bias+dev):.3f}")
    return best


def main():
    names, cache = load_cache()
    fold_lists = folds(cache)
    for v in sys.argv[1:]:
        base, _, mod = v.partition(":")           # e.g. "all:s3" = 3-seed ensemble
        cols = VARIANTS[base]
        seeds = tuple(range(int(mod[1:]))) if mod.startswith("s") else (0,)
        v = v.replace(":", "_")                   # windows-safe filename
        parts = run(v, cols, names, cache, fold_lists, seeds=seeds)
        T, R, W, D = pool(cache, parts)
        np.savez_compressed(os.path.join(HERE, f"oof_{v}.npz"), R=R.astype(np.float32))
        if not os.path.exists(os.path.join(HERE, "oof_common.npz")):
            np.savez_compressed(os.path.join(HERE, "oof_common.npz"),
                                T=T.astype(np.float32), W=W.astype(np.int32),
                                D=D.astype(np.float32))
        report(v, T, R, W)


if __name__ == "__main__":
    main()
