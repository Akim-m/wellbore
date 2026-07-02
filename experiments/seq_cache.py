"""Full-length sequence cache for the pre-PS-context TCN (v3).

Per well: cut-independent channels (GR shape). Per cut: PS-relative channels
including the known TVT residual (the steering history), which zeroes out at PS.
Loss mask = post-PS points with finite label.
"""
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
import model
from eda import DATA, ps_index
from model import load_well
from progress import every, log

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seq_cache.pkl")
MIN_PS = 50
FRACS = (0.5, 0.65, 0.8, 1.0)

# channel names (docs): well-level [gr_h, gr_grad, gr_roll, incl]
# cut-level [dmd_k, dz, lat, known_res, known_mask, gr_res_anchor, gr_offset]


def well_channels(hw):
    md = hw["MD"].to_numpy()
    z = hw["Z"].to_numpy()
    gr_h = model._smooth(model._zscore(hw["GR"].to_numpy()), model.SMOOTH)
    grad = np.gradient(np.where(np.isfinite(gr_h), gr_h, 0.0), md)
    roll = np.sqrt(np.clip(model._smooth(gr_h ** 2, 101) - model._smooth(gr_h, 101) ** 2, 0, None))
    incl = np.gradient(z, md)
    return np.stack([gr_h, grad, roll, incl]).astype(np.float32)


def cut_channels(hw, tw, gr_h, ps):
    n = len(hw)
    md, x, y, z = (hw[c].to_numpy() for c in ("MD", "X", "Y", "Z"))
    tin = hw["TVT_input"].to_numpy().astype(float)
    anchor = tin[ps - 1]

    grid = np.arange(anchor - model.BAND, anchor + model.BAND + model.GRID, model.GRID)
    _, ref = model._reference(grid, hw, ps, tw)
    a_idx = int(round((anchor - grid[0]) / model.GRID))
    ref_anchor = ref[a_idx] if np.isfinite(ref[a_idx]) else np.nanmean(ref)

    gr_res = gr_h - ref_anchor
    cost = (gr_h[:, None] - ref[None, :]) ** 2
    cost[:, ~np.isfinite(ref)] = np.inf
    cost[~np.isfinite(gr_h), :] = 0.0
    gr_off = np.argmin(cost, axis=1) * model.GRID + grid[0] - anchor

    known = np.isfinite(tin)
    kres = np.where(known, tin - anchor, 0.0)
    C = np.stack([
        (md - md[ps - 1]) / 1000.0,
        z - z[ps - 1],
        np.hypot(x - x[ps - 1], y - y[ps - 1]) / 1000.0,
        kres,
        known.astype(float),
        gr_res,
        gr_off,
    ]).astype(np.float32)
    return C, np.float32(anchor)


def main():
    wells = sorted(os.path.basename(p).replace("__horizontal_well.csv", "")
                   for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))
    cache = {}
    for i, w in enumerate(wells, 1):
        hw, tw = load_well("train", w)
        ps = ps_index(hw)
        if not (MIN_PS <= ps < len(hw)):
            continue
        WC = well_channels(hw)
        tvt = hw["TVT"].to_numpy().astype(np.float32)
        cuts = []
        for f in FRACS:
            p = int(round(f * ps))
            if p < MIN_PS or p >= len(hw):
                continue
            h = hw
            if p != ps:
                h = hw.copy()
                t2 = tvt.astype(float).copy()
                t2[p:] = np.nan
                h["TVT_input"] = t2
            C, anchor = cut_channels(h, tw, WC[0], p)
            cuts.append(dict(C=C, ps=p, anchor=anchor))
        if cuts:
            cache[w] = dict(WC=WC, tvt=tvt, cuts=cuts)
        if every(i, len(wells)):
            log(f"seqcache: {i}/{len(wells)} wells")
    with open(OUT, "wb") as fo:
        pickle.dump(cache, fo, protocol=4)
    log(f"cached {len(cache)} wells -> {OUT}")


if __name__ == "__main__":
    main()
