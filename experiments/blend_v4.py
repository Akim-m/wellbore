"""Combine TCN-v4 fold OOFs and compute the final fold-honest blends."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
from exp import load_cache, smooth_within
from stack import build_common

names, cache = load_cache()
allw = sorted(cache)

parts = {}
for k in range(5):
    d = np.load(os.path.join(HERE, f"oof_seq_v4_f{k}.npz"), allow_pickle=True)  # own file
    R, ws = d["R"], list(d["wells"])
    off = 0
    for w in sorted(ws):
        n = len(cache[w]["y"])
        parts[w] = R[off:off + n]
        off += n
    assert off == len(R)
R4 = np.concatenate([parts[w] for w in allw])
np.savez_compressed(os.path.join(HERE, "oof_seq_v4.npz"), R=R4.astype(np.float32))

T, W, D, F = build_common()
ramp = np.interp(D, [50, 175, 375, 750, 1500], [0.103, 0.384, 0.698, 0.904, 1.0])

import glob
pf = {}
for p in glob.glob(os.path.join(HERE, "pf_preds_*.npz")):
    d = np.load(p)
    for w in d.files:
        v = d[w]
        pf[w] = v[1:] - v[0]
Rpf = np.concatenate([pf[w] for w in allw])
Rpf_s = smooth_within(Rpf, W, 101)

S4 = smooth_within(R4.astype(float), W)
tcn_old = [smooth_within(np.load(os.path.join(HERE, f"oof_{n}.npz"))["R"].astype(float), W)
           for n in ("seq", "seq_v2", "seq_v2s1")]


def honest(cols, tag):
    P = np.column_stack(cols)
    pred = np.zeros_like(T)
    ws = []
    for k in range(5):
        tr, te = F != k, F == k
        w = np.linalg.lstsq(P[tr], T[tr], rcond=None)[0]
        pred[te] = P[te] @ w
        ws.append(w)
    print(f"{tag}: honest = {np.sqrt(np.mean((T - pred) ** 2)):.4f}  mean_w = {np.round(np.mean(ws, 0), 3)}")


print("v4 alone raw pooled:", round(float(np.sqrt(np.mean((T - R4) ** 2))), 4))
honest([S4], "v4_smoothed")
honest([S4 * ramp], "v4_ramped")
honest([S4 * ramp, Rpf_s], "v4_ramped + pf")
honest([S4 * ramp, Rpf_s] + [t * ramp for t in tcn_old], "v4_ramped + pf + tcn3_ramped")
honest([S4, Rpf_s] + tcn_old, "all_unramped")
