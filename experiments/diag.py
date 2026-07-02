"""Diagnostics: anchor exactness, error decomposition, residual drift vs dmd."""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
from eda import DATA, ps_index

wells = sorted(os.path.basename(p).replace("__horizontal_well.csv", "")
               for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))

anchor_err = []          # |TVT_input - TVT| at ps-1
pre_eq = []              # fraction of pre-PS rows where TVT_input == TVT exactly
bias_sq, dev_sq, npts = [], [], []   # per-well decomposition of hold-TVT error
drift_dmd, drift_r = [], []          # pooled (dmd, TVT-anchor) samples

for w in wells:
    hw = pd.read_csv(os.path.join(DATA, "train", f"{w}__horizontal_well.csv"))
    ps = ps_index(hw)
    n = len(hw)
    if ps < 2 or ps >= n:
        continue
    tvt = hw["TVT"].to_numpy()
    tin = hw["TVT_input"].to_numpy()
    md = hw["MD"].to_numpy()

    anchor_err.append(abs(tin[ps - 1] - tvt[ps - 1]))
    pre = np.isfinite(tin[:ps]) & np.isfinite(tvt[:ps])
    pre_eq.append(np.mean(np.abs(tin[:ps][pre] - tvt[:ps][pre]) < 1e-6) if pre.any() else np.nan)

    r = tvt[ps:] - tin[ps - 1]          # residual of hold-TVT
    m = np.isfinite(r)
    r = r[m]
    if len(r) == 0:
        continue
    b = r.mean()
    bias_sq.append(b * b * len(r))
    dev_sq.append(((r - b) ** 2).sum())
    npts.append(len(r))

    dd = (md[ps:] - md[ps - 1])[m]
    step = max(1, len(r) // 50)
    drift_dmd.append(dd[::step])
    drift_r.append(r[::step])

anchor_err = np.array(anchor_err)
print(f"wells used: {len(npts)}")
print(f"anchor |TVT_input-TVT| at ps-1: max={anchor_err.max():.4f}, mean={anchor_err.mean():.6f}")
print(f"pre-PS exact-equality fraction: min={np.nanmin(pre_eq):.4f}, mean={np.nanmean(pre_eq):.4f}")

tot = sum(bias_sq) + sum(dev_sq)
N = sum(npts)
print(f"\nhold-TVT pooled RMSE (train, all points) = {np.sqrt(tot/N):.4f}")
print(f"  per-well-bias share of MSE   = {sum(bias_sq)/tot:.3f}  (RMSE if only bias removed: {np.sqrt(sum(dev_sq)/N):.4f})")
print(f"  within-well-dev share of MSE = {sum(dev_sq)/tot:.3f}  (RMSE if only dev removed:  {np.sqrt(sum(bias_sq)/N):.4f})")

dd = np.concatenate(drift_dmd)
rr = np.concatenate(drift_r)
print("\nresidual (TVT - anchor) vs distance-beyond-PS (ft):")
bins = [0, 500, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 1e9]
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (dd >= lo) & (dd < hi)
    if m.sum() < 50:
        continue
    print(f"  dmd {int(lo):>5}-{int(hi) if hi < 1e9 else 'inf':>5}: n={m.sum():>6}  mean={rr[m].mean():+7.2f}  rms={np.sqrt((rr[m]**2).mean()):7.2f}")
