"""Well-level holdout: fit on 80% of wells, score the held-out 20% vs hold-TVT."""
import glob
import os
import sys

import numpy as np

from eda import DATA, ps_index
from model import load_well
from progress import every, log
from train_residual import MIN_PS, fit_model, predict_well

feat_kw = {"seq": True} if "seq" in sys.argv else {}

wells = sorted(
    os.path.basename(p).replace("__horizontal_well.csv", "")
    for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv"))
)
holdout = set(wells[::5])                      # deterministic 20%
train = [w for w in wells if w not in holdout]

log(f"train={len(train)} holdout={len(holdout)} feat_kw={feat_kw}  fitting...")
reg = fit_model(train, feat_kw)

hold_list = sorted(holdout)
log(f"scoring {len(hold_list)} holdout wells...")
model_rmse, base_rmse, wins = [], [], 0
for n, w in enumerate(hold_list, 1):
    hw, tw = load_well("train", w)
    ps = ps_index(hw)
    if not (MIN_PS <= ps < len(hw)):
        continue
    true = hw["TVT"].to_numpy()[ps:]
    pred = predict_well(reg, hw, tw, feat_kw)[ps:]
    anchor = hw["TVT_input"].to_numpy()[ps - 1]
    mr = np.sqrt(np.mean((true - pred) ** 2))
    br = np.sqrt(np.mean((true - anchor) ** 2))
    model_rmse.append(mr)
    base_rmse.append(br)
    wins += mr < br
    if every(n, len(hold_list)):
        log(f"scored {n}/{len(hold_list)} wells")

m, b = np.array(model_rmse), np.array(base_rmse)
print(f"scored {len(m)} holdout wells")
print(f"residual   mean={m.mean():7.3f}  median={np.median(m):7.3f}")
print(f"hold-TVT   mean={b.mean():7.3f}  median={np.median(b):7.3f}")
print(f"beats hold-TVT on {wins}/{len(m)} wells")
print(f"mean improvement: {b.mean() - m.mean():+.3f} ft")
