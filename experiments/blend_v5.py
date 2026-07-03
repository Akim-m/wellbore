"""Combine TCN-v5 fold OOFs and re-run the fold-honest blends vs/with v4+PF+tcn3."""
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

parts = {}
for k in range(5):
    d = np.load(os.path.join(HERE, f"oof_seq_v5_f{k}.npz"), allow_pickle=True)
    R, ws = d["R"], list(d["wells"])
    off = 0
    for w in sorted(ws):
        n = len(cache[w]["y"])
        parts[w] = R[off:off + n]
        off += n
    assert off == len(R)
R5 = np.concatenate([parts[w] for w in allw])
np.savez_compressed(os.path.join(HERE, "oof_seq_v5.npz"), R=R5.astype(np.float32))

T, W, D, F = build_common()
ramp = np.interp(D, [50, 175, 375, 750, 1500], [0.103, 0.384, 0.698, 0.904, 1.0])

ms = {}
for p in glob.glob(os.path.join(HERE, "pf_ms_*.npz")):
    d = np.load(p)
    for k in d.files:
        if not k.endswith("__a"):
            ms[k] = (d[k], float(d[f"{k}__a"][0]))
scales = []
for si in range(4):
    R = np.concatenate([(ms[w][0][si] - ms[w][1]) if w in ms and ms[w][0].shape[1] == len(cache[w]["y"])
                        else np.zeros(len(cache[w]["y"])) for w in allw])
    scales.append(smooth_within(R, W, 101))

S5 = smooth_within(R5.astype(float), W)
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


print("v5 alone raw pooled:", round(float(np.sqrt(np.mean((T - R5) ** 2))), 4))
honest([S5], "v5_smoothed")
honest([S5 * ramp], "v5_ramped")
honest([S5 * ramp, scales[0], scales[2]], "v5 + pf(s3,s8)")
honest([S5 * ramp, S4, scales[0], scales[2]], "v5 + v4 + pf(s3,s8)")
honest([S5 * ramp, S4, scales[0], scales[2]] + tcn_old, "v5 + v4 + pf(s3,s8) + tcn3")
honest([S5 * ramp] + scales + [S4] + tcn_old, "v5 + pf_ms4 + v4 + tcn3")
