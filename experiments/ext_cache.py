"""Build TCN-v4 caches: 15 base channels + pf_delta + 11-offset GR cost-volume.

Writes aug_cache_v4.pkl (training cuts, fracs .5/.65/.8/1.0) and cache_v4.pkl
(eval, true cuts) with a shared names list. PF channels come from pf_cuts_*.npz
(fracs<1) and pf_preds_*.npz (frac=1.0); missing PF -> NaN (mask channel covers).
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
EXT = ["pf_delta"] + [f"cv_{int(o)}" for o in OFFS]


def load_pf():
    pf = {}
    for p in glob.glob(os.path.join(HERE, "pf_cuts_*.npz")):
        d = np.load(p)
        for k in d.files:
            pf[k] = d[k]
    for p in glob.glob(os.path.join(HERE, "pf_preds_*.npz")):
        d = np.load(p)
        for w in d.files:
            v = d[w]
            pf[f"{w}@1.0"] = (v[1:] - v[0]).astype(np.float32)
    return pf


def costvol(hw, tw, ps, anchor, npost):
    """gr_h - ref(anchor+offset) over the post region; wide band to cover ±80."""
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
    pf = load_pf()
    log(f"pf channels loaded: {len(pf)} cuts")
    wells = sorted(os.path.basename(p).replace("__horizontal_well.csv", "")
                   for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))
    aug, base = {}, {}
    names_ref = None
    nomatch = 0
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
            cv = costvol(h, tw, p2, anchor, npost)
            key = f"{w}@{f}"
            pfv = pf.get(key)
            if pfv is None or len(pfv) != npost:
                pfv = np.full(npost, np.nan, np.float32)
                nomatch += 1
            Xv = np.hstack([X, pfv[:, None], cv]).astype(np.float32)
            names_ref = names + EXT
            cuts.append(dict(X=Xv, y=label.astype(np.float32), frac=f))
            if f == 1.0:
                base[w] = dict(X=Xv, y=label.astype(np.float32),
                               anchor=np.float32(anchor),
                               dmd=(hw["MD"].to_numpy()[p2:] - hw["MD"].to_numpy()[p2 - 1]).astype(np.float32))
        if cuts:
            aug[w] = cuts
        if every(i, len(wells)):
            log(f"ext: {i}/{len(wells)} wells ({nomatch} pf-missing)")
    with open(os.path.join(HERE, "aug_cache_v4.pkl"), "wb") as fo:
        pickle.dump(dict(names=names_ref, wells=aug), fo, protocol=4)
    with open(os.path.join(HERE, "cache_v4.pkl"), "wb") as fo:
        pickle.dump(dict(names=names_ref, wells=base), fo, protocol=4)
    log(f"done: {len(aug)} wells, pf-missing cuts: {nomatch}, {len(names_ref)} channels")


if __name__ == "__main__":
    main()
