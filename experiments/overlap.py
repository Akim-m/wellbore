"""Hunt for train<->test overlap and shared typewells (the step-change levers).

1. Do the 3 local test wells exist in train (id and/or content)?
2. Are typewells shared across wells (content hash groups)?
3. Are there duplicate physical wells within train under different ids?
"""
import glob
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
from eda import DATA

def wid(p):
    return os.path.basename(p).split("__")[0]

train_h = sorted(glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))
test_h = sorted(glob.glob(os.path.join(DATA, "test", "*__horizontal_well.csv")))
train_ids = {wid(p) for p in train_h}

print("=== 1. test wells vs train ===")
for p in test_h:
    w = wid(p)
    intrain = w in train_ids
    line = f"test {w}: id in train = {intrain}"
    if intrain:
        te = pd.read_csv(p)
        tr = pd.read_csv(os.path.join(DATA, "train", f"{w}__horizontal_well.csv"))
        same_len = len(te) == len(tr)
        cols = ["MD", "X", "Y", "Z", "GR"]
        k = min(len(te), len(tr))
        eq = all(np.allclose(te[c].to_numpy()[:k], tr[c].to_numpy()[:k], equal_nan=True) for c in cols)
        line += f", rows test={len(te)} train={len(tr)}, XYZGR identical(first {k})={eq}"
    print(line)

print("\n=== 2. typewell content groups ===")
def th(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

tw_train = {wid(p): th(p) for p in glob.glob(os.path.join(DATA, "train", "*typewell*.csv"))}
tw_test = {wid(p): th(p) for p in glob.glob(os.path.join(DATA, "test", "*typewell*.csv"))}
from collections import Counter
cnt = Counter(tw_train.values())
sizes = Counter(cnt.values())
print(f"train typewells: {len(tw_train)} files, {len(cnt)} distinct contents")
print(f"group-size distribution (size: n_groups): {dict(sorted(sizes.items()))}")
hits = sum(1 for h in tw_test.values() if h in cnt)
print(f"test typewells matching a train typewell exactly: {hits}/{len(tw_test)}")

print("\n=== 3. duplicate physical trajectories within train ===")
fp = {}
for p in train_h:
    hw = pd.read_csv(p, usecols=["MD", "X", "Y", "Z"])
    key = (len(hw), round(float(hw["X"].iloc[0]), 1), round(float(hw["Y"].iloc[0]), 1),
           round(float(hw["MD"].iloc[-1]), 1))
    fp.setdefault(key, []).append(wid(p))
dups = {k: v for k, v in fp.items() if len(v) > 1}
print(f"trajectory-fingerprint duplicate groups in train: {len(dups)}")
for k, v in list(dups.items())[:10]:
    print("  ", k, v)

print("\n=== 4. near-duplicate check: same start XY (2 ft) across different ids ===")
starts = []
for p in train_h:
    hw = pd.read_csv(p, usecols=["X", "Y"], nrows=1)
    starts.append((float(hw["X"].iloc[0]), float(hw["Y"].iloc[0]), wid(p)))
S = np.array([[s[0], s[1]] for s in starts])
ids = [s[2] for s in starts]
close_pairs = 0
examples = []
for i in range(len(S)):
    d = np.hypot(S[i + 1:, 0] - S[i, 0], S[i + 1:, 1] - S[i, 1])
    for j in np.where(d < 2.0)[0]:
        close_pairs += 1
        if len(examples) < 10:
            examples.append((ids[i], ids[i + 1 + j], float(d[j])))
print(f"pairs of distinct train wells starting within 2 ft: {close_pairs}")
for e in examples:
    print("  ", e)
