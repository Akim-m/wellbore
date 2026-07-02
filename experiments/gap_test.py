"""Gap-robustness test: leading NaNs + interior gaps in TVT_input must not
move the suffix predictions, and gap rows must interpolate near-exactly."""
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
os.environ["KNB_WEIGHTS"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tcn_dataset")
import kaggle_notebook as knb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp import load_cache

_, cache = load_cache()
wells = sorted(cache)[::40][:15]
models = knb.load_models(knb.find_weights())
rng = np.random.default_rng(7)

suf_diff, gap_err, base_rmse, gap_rmse_old_style = [], [], [], []
for w in wells:
    hw, tw = knb.load_well("train", w)
    ps0 = knb.ps_index(hw)
    clean = knb.predict_full(models, hw, tw)

    g = hw.copy()
    t = g["TVT_input"].to_numpy().astype(float).copy()
    t[: rng.integers(50, 300)] = np.nan                     # leading NaNs
    gaps = []
    for _ in range(rng.integers(2, 5)):                     # interior gaps
        st = rng.integers(400, ps0 - 100)
        ln = rng.integers(5, 60)
        t[st:st + ln] = np.nan
        gaps.append((st, st + ln))
    g["TVT_input"] = t
    assert knb.ps_index(g) == ps0, (knb.ps_index(g), ps0)   # anchor unchanged
    pred = knb.predict_full(models, g, tw)

    true = hw["TVT"].to_numpy()
    suf_diff.append(np.abs(pred[ps0:] - clean[ps0:]).max())
    ge = np.concatenate([pred[a:b] - true[a:b] for a, b in gaps])
    gap_err.append(float(np.sqrt(np.mean(ge ** 2))))
    base_rmse.append(float(np.sqrt(np.mean((pred[ps0:] - true[ps0:]) ** 2))))

print(f"wells tested: {len(wells)}")
print(f"suffix prediction shift due to gaps: max={max(suf_diff):.4f} ft (want ~0)")
print(f"interior-gap interpolation RMSE:     max={max(gap_err):.4f} ft (want <1)")
print(f"suffix RMSE vs truth (sanity):       mean={np.mean(base_rmse):.2f} ft")
