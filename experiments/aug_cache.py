"""Build an AUGMENTED feature cache: each train well re-cut at multiple PS points.

True TVT is known everywhere in train, so any position can serve as a synthetic
Prediction Start. Each cut yields (pointwise X,y) + (well-summary row, mean-residual)
exactly like the real task. Evaluation always uses the well's TRUE PS (base cache).
"""
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
from eda import DATA, ps_index
from features import well_features, well_summary
from model import load_well
from progress import every, log

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aug_cache.pkl")
MIN_PS = 50
FRACS = (0.5, 0.65, 0.8, 1.0)   # synthetic PS as fraction of the true PS


def recut(hw, ps_new):
    """Copy of hw with TVT_input known only up to ps_new."""
    h = hw.copy()
    tvt = h["TVT"].to_numpy().astype(float).copy()
    tvt[ps_new:] = np.nan
    h["TVT_input"] = tvt
    return h


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
        cuts = []
        for f in FRACS:
            p = int(round(f * ps))
            if p < MIN_PS or p >= len(hw):
                continue
            h = recut(hw, p) if p != ps else hw
            X, names, label, p2, anchor = well_features(h, tw, with_label=True, seq=True)
            s = well_summary(h, tw, X, names, p2, anchor)
            names_ref = names
            cuts.append(dict(X=X.astype(np.float32), y=label.astype(np.float32),
                             s=s.astype(np.float32), frac=f))
        if cuts:
            cache[w] = cuts
        if every(i, len(wells)):
            log(f"aug: {i}/{len(wells)} wells, {sum(len(c) for c in cache.values())} cuts")
    with open(OUT, "wb") as f:
        pickle.dump(dict(names=names_ref, wells=cache), f, protocol=4)
    tot = sum(len(c["y"]) for cuts in cache.values() for c in cuts)
    log(f"cached {len(cache)} wells, {sum(len(c) for c in cache.values())} cuts, {tot} points -> {OUT}")


if __name__ == "__main__":
    main()
