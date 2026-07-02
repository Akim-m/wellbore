"""Score the model on a sample of train wells (they have true TVT beyond PS)."""
import glob
import os
import sys

import numpy as np

from eda import DATA, ps_index
from model import load_well, predict_tvt
from progress import every, log

N = int(sys.argv[1]) if len(sys.argv) > 1 else 80

wells = sorted(
    os.path.basename(p).replace("__horizontal_well.csv", "")
    for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv"))
)
sample = wells[::len(wells) // N][:N]   # evenly spread, deterministic

log(f"scoring DP model on {len(sample)} wells...")
model_rmse, base_rmse, wins = [], [], 0
for i, w in enumerate(sample, 1):
    hw, tw = load_well("train", w)
    ps = ps_index(hw)
    if ps >= len(hw) or ps < 5:
        continue
    true = hw["TVT"].to_numpy()[ps:]
    pred = predict_tvt(hw, tw)[ps:]
    mr = np.sqrt(np.mean((true - pred) ** 2))
    br = np.sqrt(np.mean((true - hw["TVT"].to_numpy()[ps - 1]) ** 2))
    model_rmse.append(mr)
    base_rmse.append(br)
    wins += mr < br
    if every(i, len(sample)):
        log(f"scored {i}/{len(sample)} wells")

m, b = np.array(model_rmse), np.array(base_rmse)
print(f"wells scored: {len(m)}")
print(f"model      mean={m.mean():7.2f}  median={np.median(m):7.2f}")
print(f"hold-TVT   mean={b.mean():7.2f}  median={np.median(b):7.2f}")
print(f"model beats baseline on {wins}/{len(m)} wells")
