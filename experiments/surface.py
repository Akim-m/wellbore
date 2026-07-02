"""Formation-surface spatial prior (port of the public FormationPlaneKNN idea).

For each train well and formation top: median depth at median (X,Y). To predict
a well: weighted local plane fit over k nearest OTHER wells (leave-self-out),
then per-well datum offset b = tail-weighted median of (TVT + Z - surf) on the
known prefix. TVT_hat = -Z + surf(X,Y) + b.

Evaluates pooled RMSE of the surface prior alone on all train wells' eval zones.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\aydhi\OneDrive\Documents\Coding\wellbore")
from eda import DATA
from progress import log

HERE = os.path.dirname(os.path.abspath(__file__))
FORMS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
K = 10


def build_index():
    rows = []
    for p in sorted(glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv"))):
        w = os.path.basename(p).split("__")[0]
        hw = pd.read_csv(p, usecols=["X", "Y"] + FORMS)
        rows.append([w, hw["X"].median(), hw["Y"].median()]
                    + [hw[f].median() for f in FORMS])
    df = pd.DataFrame(rows, columns=["wid", "x", "y"] + FORMS)
    return df


def impute(idx, wid, X, Y):
    """Per-formation surface depth at points (X,Y), plane fit over K nearest
    other wells, inverse-distance weighted."""
    others = idx[idx["wid"] != wid]
    cx, cy = float(np.median(X)), float(np.median(Y))
    d = np.hypot(others["x"] - cx, others["y"] - cy).to_numpy()
    nn = others.iloc[np.argsort(d)[:K]]
    dd = np.hypot(nn["x"] - cx, nn["y"] - cy).to_numpy()
    wgt = 1.0 / np.maximum(dd, 1.0)
    A = np.column_stack([nn["x"], nn["y"], np.ones(len(nn))])
    out = {}
    for f in FORMS:
        z = nn[f].to_numpy()
        ok = np.isfinite(z)
        if ok.sum() >= 4:
            W = np.diag(wgt[ok])
            try:
                coef = np.linalg.lstsq(W @ A[ok], W @ z[ok], rcond=None)[0]
                out[f] = coef[0] * X + coef[1] * Y + coef[2]
                continue
            except np.linalg.LinAlgError:
                pass
        out[f] = np.full(len(X), np.average(z[ok], weights=wgt[ok]) if ok.any() else np.nan)
    return out, float(np.median(np.sort(dd)[:3]))


def well_surface_pred(idx, wid, hw):
    """TVT_hat per formation + calibration diagnostics. Returns dict of arrays."""
    known = hw["TVT_input"].notna().to_numpy()
    if known.sum() < 20 or (~known).sum() == 0:
        return None
    X, Y, Z = (hw[c].to_numpy(float) for c in ("X", "Y", "Z"))
    tin = hw["TVT_input"].to_numpy(float)
    surf, nn_dist = impute(idx, wid, X, Y)

    kn = np.where(known)[0]
    tail_w = np.exp(0.02 * np.arange(len(kn)))          # tail-weighted (recent matters)
    preds, rmses = {}, {}
    for f in FORMS:
        s = surf[f]
        if not np.isfinite(s).all():
            continue
        resid = tin[kn] + Z[kn] - s[kn]
        b = float(np.average(resid, weights=tail_w))
        tvt_hat = -Z + s + b
        preds[f] = tvt_hat
        rmses[f] = float(np.sqrt(np.average((tvt_hat[kn] - tin[kn]) ** 2, weights=tail_w)))
    if not preds:
        return None
    # consensus: rmse-weighted mean over formations
    P = np.stack([preds[f] for f in preds])
    r = np.array([rmses[f] for f in preds])
    wgt = 1.0 / np.maximum(r, 0.5) ** 2
    cons = (wgt[:, None] * P).sum(0) / wgt.sum()
    return dict(cons=cons, best=P[np.argmin(r)], spread=P.std(0),
                rmse_best=float(r.min()), nn_dist=nn_dist)


def main():
    idx = build_index()
    wells = list(idx["wid"])
    se = sh = c = 0.0
    saved = {}
    for n, w in enumerate(wells, 1):
        hw = pd.read_csv(os.path.join(DATA, "train", f"{w}__horizontal_well.csv"))
        r = well_surface_pred(idx, w, hw)
        if r is None:
            continue
        known = hw["TVT_input"].notna().to_numpy()
        kn = np.where(known)[0]
        ev = np.where(~known)[0]
        ev = ev[ev > kn[-1]]
        true = hw["TVT"].to_numpy(float)[ev]
        anchor = hw["TVT_input"].to_numpy(float)[kn[-1]]
        e = r["cons"][ev] - true
        se += float(np.sum(e * e))
        sh += float(np.sum((anchor - true) ** 2))
        c += len(ev)
        saved[w] = np.stack([r["cons"], r["best"], r["spread"]]).astype(np.float32)
        if n % 100 == 0 or n == len(wells):
            log(f"{n}/{len(wells)} pooled: surf={np.sqrt(se/c):.3f} hold={np.sqrt(sh/c):.3f}")
    np.savez_compressed(os.path.join(HERE, "surf_preds.npz"), **saved)
    log(f"FINAL surf={np.sqrt(se/c):.4f} hold={np.sqrt(sh/c):.4f}")


if __name__ == "__main__":
    main()
