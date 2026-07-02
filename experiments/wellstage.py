"""Well-level bias model: predict each well's (n-weighted) mean future residual.

67.7% of pooled MSE is per-well bias, so a good constant per well is the big
lever. 5-fold OOF, sample_weight = points per well (pooled metric). Saves
oof_wellbias.npz (constant broadcast per point) for stacking.
"""
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
from progress import log

from exp import folds, load_cache, pool, report

HERE = os.path.dirname(os.path.abspath(__file__))

# constant-per-well columns + aggregates of pointwise ones
CONST = ["pre_tvt_std", "pre_tvt_slope", "ref_slope", "align_delta",
         "align_slope", "align_ev", "mean_gr_res"]
AGG = ["implied_off", "gr_offset", "gr_res_anchor"]   # -> nanmedian over well


def well_matrix(cache, names):
    ci = [names.index(c) for c in CONST]
    ai = [names.index(c) for c in AGG]
    ws = sorted(cache)
    F, Y, N = [], [], []
    for w in ws:
        X, y, dmd = cache[w]["X"], cache[w]["y"], cache[w]["dmd"]
        row = list(X[0, ci])
        for j in ai:
            col = X[:, j]
            row.append(np.nanmedian(col) if np.isfinite(col).any() else np.nan)
        row += [np.log1p(len(y)), np.log1p(float(dmd[-1]))]
        F.append(row)
        Y.append(float(y.mean()))
        N.append(len(y))
    F = np.array(F, dtype=float)
    F[~np.isfinite(F)] = np.nan
    return ws, F, np.array(Y), np.array(N, dtype=float)


def main():
    names, cache = load_cache()
    fold_lists = folds(cache)
    ws, F, Y, N = well_matrix(cache, names)
    wi = {w: i for i, w in enumerate(ws)}

    for mname, make in (("gbm", lambda: HistGradientBoostingRegressor(
                             max_iter=150, learning_rate=0.05, min_samples_leaf=40,
                             l2_regularization=5.0, random_state=0)),
                        ("ridge", lambda: Ridge(alpha=10.0))):
        parts = {}
        for k, hold in enumerate(fold_lists):
            hset = set(hold)
            tr = np.array([i for w, i in wi.items() if w not in hset])
            Ftr, Ytr, Ntr = F[tr], Y[tr], N[tr]
            if mname == "ridge":   # ridge can't take NaN: impute with train medians
                med = np.nanmedian(Ftr, axis=0)
                Ftr = np.where(np.isfinite(Ftr), Ftr, med)
                mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
                Ftr = (Ftr - mu) / sd
            reg = make()
            reg.fit(Ftr, Ytr, sample_weight=Ntr)
            for w in hold:
                f = F[wi[w]:wi[w] + 1]
                if mname == "ridge":
                    f = (np.where(np.isfinite(f), f, med) - mu) / sd
                c = float(reg.predict(f)[0])
                parts[w] = np.full(len(cache[w]["y"]), c)
            log(f"wellbias-{mname}: fold {k} done")
        T, R, W, D = pool(cache, parts)
        np.savez_compressed(os.path.join(HERE, f"oof_wellbias_{mname}.npz"),
                            R=R.astype(np.float32))
        report(f"wellbias-{mname}", T, R, W)


if __name__ == "__main__":
    main()
