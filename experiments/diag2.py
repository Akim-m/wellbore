"""Is window-level GR matching discriminative enough to track TVT?

For sampled post-PS windows across train wells:
  A. beta: how much do within-window TVT wiggles follow Z wiggles?
  B. E(t) = window MSE of gr vs ref(t + beta*(Z-Zmid)) over a TVT grid.
     Does argmin_t E locate true TVT better than holding the anchor?
Reports per-window |argmin-true| vs |anchor-true| by distance-from-PS.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
import model
from eda import DATA, ps_index
from model import load_well
from progress import log

WIN = 128          # window length (points ~ ft)
STRIDE = 64
BAND2 = 60.0       # search band around anchor for window centers
GRIDS = np.arange(-BAND2, BAND2 + 0.5, 0.5)   # candidate TVT offsets vs anchor

wells = sorted(os.path.basename(p).replace("__horizontal_well.csv", "")
               for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))
rng = np.random.default_rng(0)
sample = list(rng.choice(wells, 120, replace=False))

rows = []   # (dmd_mid, err_grmatch_b0, err_grmatch_b1, err_hold, corr_tz, beta)
for wn, w in enumerate(sample, 1):
    hw, tw = load_well("train", w)
    ps = ps_index(hw)
    n = len(hw)
    if not (50 <= ps < n) or n - ps < WIN * 2:
        continue
    anchor = hw["TVT_input"].to_numpy()[ps - 1]
    md = hw["MD"].to_numpy()
    z = hw["Z"].to_numpy()
    tvt = hw["TVT"].to_numpy()

    grid = np.arange(anchor - model.BAND - BAND2, anchor + model.BAND + BAND2 + 0.5, 0.5)
    gr_h, ref = model._reference(grid, hw, ps, tw)

    for st in range(ps, n - WIN, STRIDE):
        sl = slice(st, st + WIN)
        g = gr_h[sl]
        if not np.isfinite(g).all():
            continue
        t_true = tvt[sl].mean()
        if not np.isfinite(t_true):
            continue
        zd = z[sl] - z[sl].mean()
        td = tvt[sl] - t_true
        # A: coupling of TVT wiggles to Z wiggles
        vz = float(np.dot(zd, zd))
        beta = float(np.dot(td, zd) / vz) if vz > 1e-6 else 0.0
        corr = float(np.corrcoef(td, zd)[0, 1]) if (td.std() > 1e-6 and zd.std() > 1e-6) else 0.0

        # B: window matching for beta in {0, 1}
        errs = []
        for b in (0.0, 1.0):
            paths = (anchor + GRIDS[:, None]) + b * zd[None, :]     # (T, WIN)
            r = np.interp(paths.ravel(), grid, ref).reshape(paths.shape)
            e = np.nanmean((g[None, :] - r) ** 2, axis=1)
            e[~np.isfinite(e)] = np.inf
            t_hat = anchor + GRIDS[int(np.argmin(e))]
            errs.append(abs(t_hat - t_true))
        rows.append((md[st + WIN // 2] - md[ps - 1], errs[0], errs[1],
                     abs(anchor - t_true), corr, beta))
    if wn % 20 == 0:
        log(f"{wn}/{len(sample)} wells, {len(rows)} windows")

R = np.array(rows)
log(f"total windows: {len(R)}")
print(f"\nTVT-Z coupling within {WIN}-pt windows: corr median={np.median(R[:,4]):.3f}, "
      f"beta median={np.median(R[:,5]):.3f}, beta p25/p75={np.percentile(R[:,5],25):.2f}/{np.percentile(R[:,5],75):.2f}")
print("\nwindow |t_hat - t_true| (ft) by distance from PS   [b0 = flat window, b1 = Z-warp]")
print(f"{'dmd bin':>12} {'n':>6} {'hold':>8} {'match_b0':>9} {'match_b1':>9}  (medians)")
bins = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 5000), (5000, 12000)]
for lo, hi in bins:
    m = (R[:, 0] >= lo) & (R[:, 0] < hi)
    if m.sum() < 30:
        continue
    print(f"{lo:>5}-{hi:<6} {m.sum():>6} {np.median(R[m,3]):>8.2f} {np.median(R[m,1]):>9.2f} {np.median(R[m,2]):>9.2f}")
print("\nmean squared (pooled-metric view):")
for lo, hi in bins:
    m = (R[:, 0] >= lo) & (R[:, 0] < hi)
    if m.sum() < 30:
        continue
    print(f"{lo:>5}-{hi:<6} {m.sum():>6} {np.sqrt(np.mean(R[m,3]**2)):>8.2f} "
          f"{np.sqrt(np.mean(R[m,1]**2)):>9.2f} {np.sqrt(np.mean(R[m,2]**2)):>9.2f}")
