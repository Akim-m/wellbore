"""PF predictions for the augmented PS cuts (training channels for TCN v4).

For each well and frac in (0.5, 0.65, 0.8): recut TVT_input at frac*ps, run the
PF (16 seeds — channel quality, cheaper than eval's 32), store residual-vs-anchor.
frac=1.0 equals the true cut and is already in pf_preds_*.npz.

Usage: pf_cuts.py START END   (well index shard)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
from eda import DATA
from progress import log

import pf as pfmod

FRACS = (0.5, 0.65, 0.8)
SEEDS = 16


def main():
    wells = sorted(os.path.basename(p).replace("__horizontal_well.csv", "")
                   for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    wells = wells[lo:hi]
    out = {}
    for n, w in enumerate(wells, 1):
        hw = pd.read_csv(os.path.join(DATA, "train", f"{w}__horizontal_well.csv"))
        tw = pd.read_csv(os.path.join(DATA, "train", f"{w}__typewell.csv"))
        known = hw["TVT_input"].notna().to_numpy()
        if not known.any():
            continue
        ps = int(np.max(np.nonzero(known)[0])) + 1
        tvt = hw["TVT"].to_numpy(float)
        for f in FRACS:
            p = int(round(f * ps))
            if p < 50 or p >= len(hw):
                continue
            h = hw.copy()
            t2 = tvt.copy()
            t2[p:] = np.nan
            h["TVT_input"] = t2
            try:
                r = pfmod.pf_well(h, tw, n_seeds=SEEDS)
            except Exception as e:
                log(f"{w}@{f}: PF failed {e}")
                continue
            if r is None:
                continue
            ev, pred = r
            anchor = t2[p - 1]
            out[f"{w}@{f}"] = (pred - anchor).astype(np.float32)
        if n % 10 == 0 or n == len(wells):
            log(f"{n}/{len(wells)} wells, {len(out)} cuts")
    np.savez_compressed(os.path.join(HERE, f"pf_cuts_{lo}_{hi}.npz"), **out)
    log(f"saved {len(out)} cuts -> pf_cuts_{lo}_{hi}.npz")


if __name__ == "__main__":
    main()
