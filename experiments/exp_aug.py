"""OOF test of PS augmentation: fit GBM + ridge on augmented cuts, evaluate on
true-PS cuts (base cache). Saves oof_aug_gbm.npz / oof_aug_ridge.npz for stacking.
"""
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
import model as dp
from progress import log

from exp import BASE, folds, load_cache, pool, report

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = 5
HP = dict(max_iter=200, learning_rate=0.05, l2_regularization=5.0,
          min_samples_leaf=1000, random_state=0)
COLS = BASE + ["inter_slope_dmd", "dmd_frac"]   # = shipped pointwise feature set


def main():
    names, cache = load_cache()                  # base cache: true-PS cuts (eval)
    with open(os.path.join(HERE, "aug_cache.pkl"), "rb") as f:   # local, self-generated
        aug = pickle.load(f)
    anames, awells = aug["names"], aug["wells"]
    idx = [names.index(c) for c in COLS]
    aidx = [anames.index(c) for c in COLS]
    fold_lists = folds(cache)

    parts_g, parts_r = {}, {}
    for k, hold in enumerate(fold_lists):
        hset = set(hold)
        tr = [w for w in awells if w not in hset and w in cache]
        Xs, ys, S, sy, sn = [], [], [], [], []
        for w in tr:
            for cut in awells[w]:
                Xs.append(cut["X"][::SUB][:, aidx])
                ys.append(cut["y"][::SUB])
                S.append(cut["s"])
                sy.append(float(cut["y"].mean()))
                sn.append(len(cut["y"]))
        Xtr, ytr = np.vstack(Xs), np.concatenate(ys)
        log(f"fold {k}: {len(tr)} wells, {len(S)} cuts, {len(ytr)} rows")
        gbm = HistGradientBoostingRegressor(**HP).fit(Xtr, ytr)

        S = np.vstack(S).astype(float)
        med = np.nanmedian(S, axis=0)
        S = np.where(np.isfinite(S), S, med)
        mu, sd = S.mean(0), S.std(0) + 1e-9
        ridge = Ridge(alpha=10.0).fit((S - mu) / sd, np.array(sy),
                                      sample_weight=np.array(sn, dtype=float))

        for w in hold:
            X = cache[w]["X"][:, idx]
            parts_g[w] = np.clip(gbm.predict(X), -dp.BAND, dp.BAND)
            # well summary for the TRUE cut = last aug cut (frac=1.0) if present
            cuts = awells.get(w)
            s = next((c["s"] for c in cuts if c["frac"] == 1.0), None) if cuts else None
            if s is None:
                parts_r[w] = np.zeros(len(cache[w]["y"]))
            else:
                s = np.where(np.isfinite(s), s, med)
                c = float(ridge.predict(((s - mu) / sd)[None, :])[0])
                parts_r[w] = np.full(len(cache[w]["y"]), c)
        log(f"fold {k} done")

    T, Rg, W, D = pool(cache, parts_g)
    _, Rr, _, _ = pool(cache, parts_r)
    np.savez_compressed(os.path.join(HERE, "oof_aug_gbm.npz"), R=Rg.astype(np.float32))
    np.savez_compressed(os.path.join(HERE, "oof_aug_ridge.npz"), R=Rr.astype(np.float32))
    report("aug_gbm", T, Rg, W)
    report("aug_ridge", T, Rr, W)


if __name__ == "__main__":
    main()
