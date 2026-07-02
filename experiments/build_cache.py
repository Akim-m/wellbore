"""Build a per-well feature cache: v3 features + candidate extras.

Loads each train well once, computes the exact v3 feature matrix (asserted equal
to features.well_features) plus new candidate columns, and pickles everything so
OOF experiments only fit GBMs.
"""
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
import model
from eda import DATA, ps_index
from features import well_features
from model import load_well
from progress import every, log

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache.pkl")
MIN_PS = 50

# parametric alignment grids
DELTAS = np.arange(-20.0, 20.5, 1.0)          # constant TVT offset (ft)
SLOPES = np.arange(-0.004, 0.0042, 0.0004)    # TVT drift per ft of MD

EXTRA = ["inter_slope_dmd", "ref_slope", "implied_off", "align_path",
         "align_delta", "align_slope", "align_ev", "gr_lead100", "gr_lead250",
         "dmd_frac", "mean_gr_res"]


def extras(hw, tw, ps, anchor, X, names):
    n = len(hw)
    md = hw["MD"].to_numpy()
    post = slice(ps, n)
    dmd = X[:, names.index("dmd")]
    pre_slope = X[:, names.index("pre_tvt_slope")]
    gr = X[:, names.index("gr")]
    gr_res = X[:, names.index("gr_res_anchor")]

    grid = np.arange(anchor - model.BAND, anchor + model.BAND + model.GRID, model.GRID)
    gr_h, ref = model._reference(grid, hw, ps, tw)
    a_idx = int(round((anchor - grid[0]) / model.GRID))

    # local slope of the reference GR at the anchor (z-units per ft of TVT)
    lo, hi = max(0, a_idx - 4), min(len(grid), a_idx + 5)
    seg, gseg = ref[lo:hi], grid[lo:hi]
    ok = np.isfinite(seg)
    ref_slope = np.polyfit(gseg[ok], seg[ok], 1)[0] if ok.sum() > 3 else np.nan

    # implied TVT offset from GR mismatch under local linearization
    denom = np.sign(ref_slope) * max(abs(ref_slope), 0.02) if np.isfinite(ref_slope) else np.nan
    implied = np.clip(gr_res / denom, -model.BAND, model.BAND) if np.isfinite(denom) else np.full(len(gr_res), np.nan)

    # parametric alignment: TVT = anchor + delta + slope*dmd, matched to ref
    paths = anchor + DELTAS[:, None, None] + SLOPES[None, :, None] * dmd[None, None, :]
    ref_at = np.interp(paths.ravel(), grid, ref).reshape(paths.shape)
    d2 = (gr[None, None, :] - ref_at) ** 2
    fin = np.isfinite(d2)
    cnt = fin.sum(axis=2)
    cost = np.where(cnt > 0, np.nansum(np.where(fin, d2, 0.0), axis=2) / np.maximum(cnt, 1), np.inf)
    if np.isfinite(cost).any() and cnt.max() >= 0.3 * len(dmd):
        bi = np.unravel_index(np.argmin(cost), cost.shape)
        a_delta, a_slope = float(DELTAS[bi[0]]), float(SLOPES[bi[1]])
        c00 = cost[len(DELTAS) // 2, len(SLOPES) // 2]
        ev = float(c00 - cost[bi]) if np.isfinite(c00) else np.nan
        align_path = a_delta + a_slope * dmd
    else:
        a_delta = a_slope = ev = np.nan
        align_path = np.full(len(dmd), np.nan)

    gr_lead100 = np.interp(md[post] + 100, md, gr_h) - gr
    gr_lead250 = np.interp(md[post] + 250, md, gr_h) - gr
    dmd_frac = dmd / max(dmd[-1], 1.0)
    mean_gr_res = np.nanmean(gr_res) if np.isfinite(gr_res).any() else np.nan

    m = len(dmd)
    cols = [pre_slope * dmd,
            np.full(m, ref_slope), implied, align_path,
            np.full(m, a_delta), np.full(m, a_slope), np.full(m, ev),
            gr_lead100, gr_lead250, dmd_frac, np.full(m, mean_gr_res)]
    E = np.column_stack(cols)
    E[~np.isfinite(E)] = np.nan
    return E


def main():
    wells = sorted(os.path.basename(p).replace("__horizontal_well.csv", "")
                   for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))
    cache = {}
    names_ref = None
    for i, w in enumerate(wells, 1):
        hw, tw = load_well("train", w)
        ps = ps_index(hw)
        if not (MIN_PS <= ps < len(hw)):
            continue
        X, names, label, ps2, anchor = well_features(hw, tw, with_label=True, seq=True)
        assert ps2 == ps
        E = extras(hw, tw, ps, anchor, X, names)
        names_ref = names + EXTRA
        cache[w] = dict(X=np.hstack([X, E]).astype(np.float32),
                        y=label.astype(np.float32),
                        anchor=np.float32(anchor),
                        dmd=(hw["MD"].to_numpy()[ps:] - hw["MD"].to_numpy()[ps - 1]).astype(np.float32))
        if every(i, len(wells)):
            log(f"cache: {i}/{len(wells)} wells")
    with open(OUT, "wb") as f:
        pickle.dump(dict(names=names_ref, wells=cache), f, protocol=4)
    tot = sum(len(v["y"]) for v in cache.values())
    log(f"cached {len(cache)} wells, {tot} points, {len(names_ref)} features -> {OUT}")


if __name__ == "__main__":
    main()
