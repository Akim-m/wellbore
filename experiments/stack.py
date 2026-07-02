"""Fold-honest stacking of saved OOF raws. Usage: stack.py name1 name2 ...

Weights = OLS of true residual on the (smoothed) model predictions, fit on 4
folds, applied to the 5th. Optionally a dmd-binned rescale on top. All numbers
reported are honest (nothing tuned on the evaluated fold).
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
import model as dp
from progress import log

from exp import folds, load_cache, smooth_within

HERE = os.path.dirname(os.path.abspath(__file__))
DBINS = np.array([0, 500, 1000, 2000, 4000, 1e18])


def build_common():
    names, cache = load_cache()
    ws = sorted(cache)
    T = np.concatenate([cache[w]["y"] for w in ws])
    W = np.concatenate([np.full(len(cache[w]["y"]), i) for i, w in enumerate(ws)])
    D = np.concatenate([cache[w]["dmd"] for w in ws])
    fold_of = {}
    for k, lst in enumerate(folds(cache)):
        for w in lst:
            fold_of[w] = k
    F = np.concatenate([np.full(len(cache[w]["y"]), fold_of[w]) for w in ws])
    return T, W, D, F


def ols(T, P):
    """weights minimizing ||T - P w||^2 (no intercept)."""
    return np.linalg.lstsq(P, T, rcond=None)[0]


def main():
    names_in = sys.argv[1:]
    T, W, D, F = build_common()
    P = []
    for nm in names_in:
        R = np.load(os.path.join(HERE, f"oof_{nm}.npz"))["R"].astype(float)
        P.append(smooth_within(R, W))
        log(f"loaded+smoothed {nm}")
    P = np.column_stack(P)

    base = float(np.sqrt(np.mean(T ** 2)))
    log(f"hold baseline {base:.4f} | stacking {names_in}")

    pred = np.zeros_like(T)
    for k in range(5):
        tr, te = F != k, F == k
        w = ols(T[tr], P[tr])
        pred[te] = P[te] @ w
        log(f"fold {k}: weights {np.round(w, 3)}")
    honest = float(np.sqrt(np.mean((T - pred) ** 2)))
    log(f"STACK honest pooled = {honest:.4f} ({honest-base:+.4f})")

    # dmd-binned rescale of the stacked prediction (fit off-fold)
    pred2 = pred.copy()
    bi = np.digitize(D, DBINS) - 1
    for k in range(5):
        tr, te = F != k, F == k
        for b in range(len(DBINS) - 1):
            mtr = tr & (bi == b)
            mte = te & (bi == b)
            if mtr.sum() < 500 or not mte.any():
                continue
            num = float(np.dot(T[mtr], pred[mtr]))
            den = float(np.dot(pred[mtr], pred[mtr])) + 1e-9
            pred2[mte] = pred[mte] * (num / den)
    honest2 = float(np.sqrt(np.mean((T - pred2) ** 2)))
    log(f"STACK+dmdcal honest pooled = {honest2:.4f} ({honest2-base:+.4f})")


if __name__ == "__main__":
    main()
