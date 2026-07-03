"""Build TCN-v5 caches: v4's 27 channels with the single pf_delta replaced by
multiscale PF (s3/s5/s8 residuals + cross-scale spread) plus 3 beam-search paths.

Channels: 15 base + pf_s3/pf_s5/pf_s8/pf_spread + 11 costvol + beam_a/b/c = 33.
Inputs: pf_cuts_*.npz ((4,n) residuals, fracs<1), pf_ms_*.npz ((4,n) raw + __a
anchor, frac=1.0), beam_cuts_*.npz ((3,n) residuals, all fracs).
Writes aug_cache_v5.pkl + cache_v5.pkl. Missing arrays -> NaN.
"""
import glob
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
import model
from eda import DATA, ps_index
from features import well_features
from model import load_well
from progress import every, log

MIN_PS = 50
FRACS = (0.5, 0.65, 0.8, 1.0)
OFFS = (-80.0, -40.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 40.0, 80.0)
EXT = ["pf_s3", "pf_s5", "pf_s8", "pf_spread"] + [f"cv_{int(o)}" for o in OFFS] + \
      ["beam_a", "beam_b", "beam_c"]


def load_pf_ms():
    """key -> (4,n) residual-vs-anchor, all scales."""
    pf = {}
    for p in glob.glob(os.path.join(HERE, "pf_cuts_*.npz")):
        d = np.load(p)
        for k in d.files:
            if d[k].ndim == 2:          # skip any stale single-scale leftovers
                pf[k] = d[k]
    for p in glob.glob(os.path.join(HERE, "pf_ms_*.npz")):
        d = np.load(p)
        for k in d.files:
            if not k.endswith("__a"):
                pf[f"{k}@1.0"] = (d[k] - float(d[f"{k}__a"][0])).astype(np.float32)
    return pf


def load_beam():
    bm = {}
    for p in glob.glob(os.path.join(HERE, "beam_cuts_*.npz")):
        d = np.load(p)
        for k in d.files:
            bm[k] = d[k]
    return bm


def costvol(hw, tw, ps, anchor, npost):
    grid = np.arange(anchor - 120.0, anchor + 120.0 + model.GRID, model.GRID)
    gr_h, ref = model._reference(grid, hw, ps, tw)
    gr = gr_h[ps:]
    cols = []
    for o in OFFS:
        r = np.interp(np.full(npost, anchor + o), grid, ref)
        cols.append(gr - r)
    return np.stack(cols, axis=1).astype(np.float32)


def recut(hw, p):
    h = hw.copy()
    t = h["TVT"].to_numpy(float).copy()
    t[p:] = np.nan
    h["TVT_input"] = t
    return h


def main():
    pf, bm = load_pf_ms(), load_beam()
    log(f"loaded: {len(pf)} pf cuts, {len(bm)} beam cuts")
    wells = sorted(os.path.basename(p).replace("__horizontal_well.csv", "")
                   for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))
    aug, base = {}, {}
    names_ref = None
    nopf = nobm = 0
    for i, w in enumerate(wells, 1):
        hw, tw = load_well("train", w)
        ps = ps_index(hw)
        if not (MIN_PS <= ps < len(hw)):
            continue
        cuts = []
        for f in FRACS:
            p = int(round(f * ps))
            if p < MIN_PS or p >= len(hw):
                continue
            h = recut(hw, p) if p != ps else hw
            X, names, label, p2, anchor = well_features(h, tw, with_label=True, seq=True)
            npost = len(label)
            key = f"{w}@{f}"
            pfv = pf.get(key)
            if pfv is None or pfv.shape[1] != npost:
                pfc = np.full((npost, 4), np.nan, np.float32)
                nopf += 1
            else:
                pfc = np.column_stack([pfv[0], pfv[1], pfv[2], pfv.std(0)]).astype(np.float32)
            bmv = bm.get(key)
            if bmv is None or bmv.shape[1] != npost:
                bmc = np.full((npost, 3), np.nan, np.float32)
                nobm += 1
            else:
                bmc = bmv.T.astype(np.float32)
            cv = costvol(h, tw, p2, anchor, npost)
            Xv = np.hstack([X, pfc, cv, bmc]).astype(np.float32)
            names_ref = names + EXT
            cuts.append(dict(X=Xv, y=label.astype(np.float32), frac=f))
            if f == 1.0:
                base[w] = dict(X=Xv, y=label.astype(np.float32),
                               anchor=np.float32(anchor),
                               dmd=(hw["MD"].to_numpy()[p2:] - hw["MD"].to_numpy()[p2 - 1]).astype(np.float32))
        if cuts:
            aug[w] = cuts
        if every(i, len(wells)):
            log(f"ext5: {i}/{len(wells)} wells (pf-miss {nopf}, beam-miss {nobm})")
    with open(os.path.join(HERE, "aug_cache_v5.pkl"), "wb") as fo:
        pickle.dump(dict(names=names_ref, wells=aug), fo, protocol=4)
    with open(os.path.join(HERE, "cache_v5.pkl"), "wb") as fo:
        pickle.dump(dict(names=names_ref, wells=base), fo, protocol=4)
    log(f"done: {len(aug)} wells, pf-miss {nopf}, beam-miss {nobm}, {len(names_ref)} channels")


if __name__ == "__main__":
    main()
