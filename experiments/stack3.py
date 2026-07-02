"""Combine seed-1 OOF, honest 3-model stack, ship weights, ramp refit."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
from exp import load_cache, smooth_within
from stack import build_common

names, cache = load_cache()
parts = {}
for k in range(5):
    d = np.load(os.path.join(HERE, f"oof_seq_v2s1_f{k}.npz"), allow_pickle=True)  # own file
    R, ws = d["R"], list(d["wells"])
    off = 0
    for w in sorted(ws):
        n = len(cache[w]["y"])
        parts[w] = R[off:off + n]
        off += n
    assert off == len(R)
allw = sorted(cache)
R = np.concatenate([parts[w] for w in allw])
np.savez_compressed(os.path.join(HERE, "oof_seq_v2s1.npz"), R=R.astype(np.float32))

T, W, D, F = build_common()
Rs = [smooth_within(np.load(os.path.join(HERE, f"oof_{n}.npz"))["R"].astype(float), W)
      for n in ("seq", "seq_v2", "seq_v2s1")]
P3 = np.column_stack(Rs)

pred = np.zeros_like(T)
for k in range(5):
    tr, te = F != k, F == k
    w = np.linalg.lstsq(P3[tr], T[tr], rcond=None)[0]
    pred[te] = P3[te] @ w
    print(f"fold {k} weights", np.round(w, 3))
print("3-model stack honest =", round(float(np.sqrt(np.mean((T - pred) ** 2))), 4))

wfull = np.linalg.lstsq(P3, T, rcond=None)[0]
print("full-OOF ship weights =", np.round(wfull, 4))
Pw = P3 @ wfull
for lo, hi in [(0, 100), (100, 250), (250, 500), (500, 1000), (1000, 1e18)]:
    m = (D >= lo) & (D < hi)
    s = float(np.dot(T[m], Pw[m]) / np.dot(Pw[m], Pw[m]))
    print(f"ramp bin {lo}-{hi}: scale={s:.3f}")

ramp = np.interp(D, [50, 175, 375, 750, 1500], [0.114, 0.408, 0.720, 0.911, 1.0])
pred2 = np.zeros_like(T)
for k in range(5):
    tr, te = F != k, F == k
    w = np.linalg.lstsq(P3[tr] * ramp[tr, None], T[tr], rcond=None)[0]
    pred2[te] = (P3[te] * ramp[te, None]) @ w
print("3-model stack honest WITH ramp =", round(float(np.sqrt(np.mean((T - pred2) ** 2))), 4))
