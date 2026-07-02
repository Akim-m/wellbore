"""Sweep DP params against the hold-TVT baseline on train wells."""
import glob
import os

import numpy as np

import model
from eda import DATA, ps_index
from model import load_well, predict_tvt

wells = sorted(
    os.path.basename(p).replace("__horizontal_well.csv", "")
    for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv"))
)
sample = wells[::len(wells) // 60][:60]

data = []
for w in sample:
    hw, tw = load_well("train", w)
    ps = ps_index(hw)
    if 5 <= ps < len(hw):
        data.append((hw, tw, ps, hw["TVT"].to_numpy()))

base = np.array([np.sqrt(np.mean((t[ps:] - t[ps - 1]) ** 2)) for _, _, ps, t in data])
print(f"wells={len(data)}  hold-TVT median={np.median(base):.2f} mean={base.mean():.2f}\n")

model.BAND, model.STEP = 40.0, 0.5
for smooth in (1, 25, 51):
    for lam in (2.0, 8.0):
        for alpha in (0.0, 0.02, 0.1):
            model.SMOOTH, model.LAM, model.ALPHA = smooth, lam, alpha
            rmse = np.array([
                np.sqrt(np.mean((true[ps:] - predict_tvt(hw, tw)[ps:]) ** 2))
                for hw, tw, ps, true in data])
            print(f"smooth={smooth:2d} lam={lam:4.1f} alpha={alpha:4.2f}  "
                  f"median={np.median(rmse):6.2f} mean={rmse.mean():6.2f} "
                  f"wins={int((rmse<base).sum())}/{len(rmse)}")
