"""5-fold out-of-fold pooled RMSE of the SHIPPED pipeline — the trustworthy gate.

Every well is held out exactly once; predictions come from the exact
train_residual.fit_model/predict_well path (GBM ensemble + well ridge + stack
weights), so this measures what the notebook will do. Note the stack weights
were themselves fit on this OOF (one 2-param fit — mild optimism, documented).
"""
import glob
import os

import numpy as np

from eda import DATA, ps_index
from model import load_well
from progress import log
from train_residual import MIN_PS, fit_model, predict_well

FK = {"seq": True}

wells = sorted(os.path.basename(p).replace("__horizontal_well.csv", "")
               for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))

T, A, P = [], [], []   # true, hold-anchor, shipped prediction
for k in range(5):
    holdout = set(wells[k::5])
    train = [w for w in wells if w not in holdout]
    log(f"fold {k}: fit on {len(train)} wells")
    bundle = fit_model(train, FK)
    for w in sorted(holdout):
        hw, tw = load_well("train", w)
        ps = ps_index(hw)
        if not (MIN_PS <= ps < len(hw)):
            continue
        T.append(hw["TVT"].to_numpy()[ps:])
        A.append(np.full(len(hw) - ps, hw["TVT_input"].to_numpy()[ps - 1]))
        P.append(predict_well(bundle, hw, tw, FK)[ps:])
    log(f"fold {k} done, {sum(len(t) for t in T)} points so far")

T, A, P = map(np.concatenate, (T, A, P))
base = float(np.sqrt(np.mean((T - A) ** 2)))
ours = float(np.sqrt(np.mean((T - P) ** 2)))
log(f"OOF pooled hold-TVT baseline = {base:.4f}  ({len(T)} points, all {len(wells)} wells)")
log(f"OOF pooled shipped pipeline = {ours:.4f}  ({ours - base:+.4f})")
