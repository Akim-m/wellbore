"""Is the sample_submission default a strong organizer baseline?"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
from eda import DATA, ps_index

sam = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
w = sam["id"].str.rsplit("_", n=1).str[0]
errs, herrs = [], []
for wid in w.unique():
    tr = pd.read_csv(os.path.join(DATA, "train", f"{wid}__horizontal_well.csv"))
    ps = ps_index(tr) if tr["TVT_input"].isna().any() else None
    rows = sam[w == wid]
    idx = rows["id"].str.rsplit("_", n=1).str[1].astype(int).to_numpy()
    true = tr["TVT"].to_numpy()[idx]
    e = rows["tvt"].to_numpy() - true
    errs.append(e)
    # hold-TVT for comparison, anchor = TVT at first predicted idx - 1
    anchor = tr["TVT"].to_numpy()[idx.min() - 1]
    herrs.append(anchor - true)
    print(f"{wid}: sample rmse={np.sqrt(np.mean(e**2)):8.3f}   hold rmse={np.sqrt(np.mean(herrs[-1]**2)):8.3f}   "
          f"sample tvt range {rows['tvt'].min():.1f}..{rows['tvt'].max():.1f}, true {true.min():.1f}..{true.max():.1f}")
e, h = np.concatenate(errs), np.concatenate(herrs)
print(f"pooled: sample={np.sqrt(np.mean(e**2)):.3f}  hold={np.sqrt(np.mean(h**2)):.3f}")
