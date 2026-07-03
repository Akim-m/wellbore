"""Beam-search DP TVT paths (port of the public LB-7.295 beam_search) as TCN-v5
channels. 3 diverse configs per cut; stores (3, npost) residual-vs-anchor.

Covers fracs 0.5/0.65/0.8/1.0 (1.0 = eval side too, unlike pf which splits files).
Usage: beam_cuts.py START END   (well index shard) -> beam_cuts_{lo}_{hi}.npz
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
from eda import DATA
from progress import log

HERE = os.path.dirname(os.path.abspath(__file__))
FRACS = (0.5, 0.65, 0.8, 1.0)
CONFIGS = ((10, 20.0, 144.0, 2),   # public default
           (25, 6.0, 50.0, 3),     # wide beam, cheap moves (wiggly)
           (10, 50.0, 400.0, 0))   # stiff, no smoothing


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r):
    n, nt = len(hgr), len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])
    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr
    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    MOVES = np.array([-2, -1, 0, 1, 2], np.int64)
    MC = mc * np.array([2.0, 1.0, 0.0, 1.0, 2.0])
    bidx = np.full(bs, si, np.int64)
    bcost = np.full(bs, np.inf)
    bcost[0] = 0.0
    bn = 1
    result = np.zeros(n)
    for step in range(n):
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        tot = bcost[:bn, None] + (sgr[step] - tw_gr[ci]) ** 2 / es + MC[None, :]
        tot = np.where((ni >= 0) & (ni < nt), tot, np.inf)
        ni_f, tot_f = ni.ravel(), tot.ravel()
        ok = np.isfinite(tot_f)
        ni_f, tot_f = ni_f[ok], tot_f[ok]
        order = np.argsort(tot_f)
        ni_s, tot_s = ni_f[order], tot_f[order]
        _, first = np.unique(ni_s, return_index=True)
        kept = min(bs, len(first))
        top = first[np.argsort(tot_s[first])][:kept]
        bidx[:kept], bcost[:kept] = ni_s[top], tot_s[top]
        if kept < bs:
            bidx[kept:], bcost[kept:] = bidx[kept - 1], np.inf
        bn = kept
        result[step] = tw_tvt[bidx[0]]
    return result


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
        tw_s = tw.dropna(subset=["TVT"]).sort_values("TVT")
        tw_tvt = tw_s["TVT"].to_numpy(float)
        tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
        gr_all = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy(float)
        tvt = hw["TVT"].to_numpy(float)
        for f in FRACS:
            p = int(round(f * ps))
            if p < 50 or p >= len(hw):
                continue
            anchor = tvt[p - 1]
            hgr = gr_all[p:]
            try:
                paths = [beam_search(hgr, tw_tvt, tw_gr, anchor, *c) for c in CONFIGS]
            except Exception as e:
                log(f"{w}@{f}: beam failed {e}")
                continue
            out[f"{w}@{f}"] = (np.stack(paths) - anchor).astype(np.float32)
        if n % 10 == 0 or n == len(wells):
            log(f"{n}/{len(wells)} wells, {len(out)} cuts")
    np.savez_compressed(os.path.join(HERE, f"beam_cuts_{lo}_{hi}.npz"), **out)
    log(f"saved {len(out)} cuts -> beam_cuts_{lo}_{hi}.npz")


if __name__ == "__main__":
    main()
