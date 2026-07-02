"""Fit the well-level ridge on ALL wells and print standardized coefficients."""
import sys

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
from exp import load_cache
from wellstage import AGG, CONST, well_matrix

names, cache = load_cache()
ws, F, Y, N = well_matrix(cache, names)
cols = CONST + [f"med_{c}" for c in AGG] + ["log_n", "log_dmd"]

med = np.nanmedian(F, axis=0)
F = np.where(np.isfinite(F), F, med)
mu, sd = F.mean(0), F.std(0) + 1e-9
Fz = (F - mu) / sd
reg = Ridge(alpha=10.0).fit(Fz, Y, sample_weight=N)
order = np.argsort(-np.abs(reg.coef_))
print(f"target: n-weighted mean residual  | mean(Y)={np.average(Y, weights=N):+.3f}")
for i in order:
    print(f"  {cols[i]:>20s}  coef={reg.coef_[i]:+7.3f}")
print(f"  intercept={reg.intercept_:+.3f}")
