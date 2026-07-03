"""Fold-honest blend with the 128-seed multiscale PF (4 scale outputs)."""
import glob
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

ms = {}
for p in glob.glob(os.path.join(HERE, "pf_ms_*.npz")):
    d = np.load(p)
    for k in d.files:
        if not k.endswith("__a"):
            ms[k] = (d[k], float(d[f"{k}__a"][0]))
print(f"multiscale pf wells: {len(ms)}")

T, W, D, F = build_common()
scales = []
for si in range(4):
    R = np.concatenate([(ms[w][0][si] - ms[w][1]) if w in ms and ms[w][0].shape[1] == len(cache[w]["y"])
                        else np.zeros(len(cache[w]["y"])) for w in allw])
    print(f"scale {si}: pooled {np.sqrt(np.mean((T - R) ** 2)):.4f}")
    scales.append(smooth_within(R, W, 101))

ramp = np.interp(D, [50, 175, 375, 750, 1500], [0.103, 0.384, 0.698, 0.904, 1.0])
S4 = smooth_within(np.load(os.path.join(HERE, "oof_seq_v4.npz"))["R"].astype(float), W) * ramp
tcn_old = [smooth_within(np.load(os.path.join(HERE, f"oof_{n}.npz"))["R"].astype(float), W) * ramp
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
    return pred


honest([S4] + scales, "v4 + pf_ms4")
honest([S4] + scales + tcn_old, "v4 + pf_ms4 + tcn3")
pred = honest([S4, scales[0], scales[2]] + tcn_old, "v4 + pf(s3,s8) + tcn3")
