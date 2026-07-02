"""Fold-honest blend of the PF tracker with the 3-generation TCN ensemble."""
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

pf = {}
for p in glob.glob(os.path.join(HERE, "pf_preds_*.npz")):
    d = np.load(p)
    for w in d.files:
        v = d[w]
        pf[w] = v[1:] - v[0]          # residual vs anchor
missing = [w for w in allw if w not in pf or len(pf[w]) != len(cache[w]["y"])]
print(f"pf wells: {len(pf)}, missing/mismatched: {len(missing)}")
Rpf = np.concatenate([pf[w] if w in pf and len(pf[w]) == len(cache[w]["y"])
                      else np.zeros(len(cache[w]["y"])) for w in allw])

T, W, D, F = build_common()
ramp = np.interp(D, [50, 175, 375, 750, 1500], [0.103, 0.384, 0.698, 0.904, 1.0])
tcn = [smooth_within(np.load(os.path.join(HERE, f"oof_{n}.npz"))["R"].astype(float), W)
       for n in ("seq", "seq_v2", "seq_v2s1")]

print("PF alone pooled:", round(float(np.sqrt(np.mean((T - Rpf) ** 2))), 4),
      " hold:", round(float(np.sqrt(np.mean(T ** 2))), 4))

for tag, cols in (("tcn3+pf", tcn + [Rpf]),
                  ("tcn3_ramped+pf", [t * ramp for t in tcn] + [Rpf]),
                  ("tcn3_ramped+pf_smooth", [t * ramp for t in tcn]
                   + [smooth_within(Rpf, W, 101)])):
    P = np.column_stack(cols)
    pred = np.zeros_like(T)
    ws = []
    for k in range(5):
        tr, te = F != k, F == k
        w = np.linalg.lstsq(P[tr], T[tr], rcond=None)[0]
        pred[te] = P[te] @ w
        ws.append(w)
    print(f"{tag}: honest = {np.sqrt(np.mean((T - pred) ** 2)):.4f}  "
          f"mean_w = {np.round(np.mean(ws, 0), 3)}")
