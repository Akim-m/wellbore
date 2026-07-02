"""Holdout evaluation with the *competition* metric: pooled RMSE over all points.

The leaderboard scores RMSE across every predicted dTVT point pooled together, so
long wells and far-from-PS points dominate. We report that (primary) plus the
per-well median (secondary), and expose raw residuals so shrink can be swept
without refitting.
"""
import glob
import os

import numpy as np

import model as dp
from eda import DATA, ps_index
from features import well_features
from model import load_well
from progress import every, log
from train_residual import MIN_PS, fit_model

BAND = dp.BAND


def _wells():
    return sorted(os.path.basename(p).replace("__horizontal_well.csv", "")
                  for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))


def collect(feat_kw=None):
    """Fit on 80% of wells; return pooled (true, anchor, raw_residual, well_id, dmd)
    arrays over the 20% holdout. raw_residual is the clipped, UNSHRUNK GBM output."""
    feat_kw = feat_kw or {}
    wells = _wells()
    holdout = set(wells[::5])
    train = [w for w in wells if w not in holdout]
    reg = fit_model(train, feat_kw)

    hold = sorted(holdout)
    log(f"collecting holdout points from {len(hold)} wells...")
    trues, anchors, raws, wids, dmds = [], [], [], [], []
    for n, w in enumerate(hold, 1):
        hw, tw = load_well("train", w)
        ps = ps_index(hw)
        if not (MIN_PS <= ps < len(hw)):
            continue
        true = hw["TVT"].to_numpy()[ps:]
        anchor = hw["TVT_input"].to_numpy()[ps - 1]
        md = hw["MD"].to_numpy()
        X, _, _, _, _ = well_features(hw, tw, with_label=False, **feat_kw)
        raw = np.clip(np.mean([g.predict(X) for g in reg["gbms"]], axis=0), -BAND, BAND)
        trues.append(true)
        anchors.append(np.full(len(true), anchor))
        raws.append(raw)
        wids.append(np.full(len(true), n))
        dmds.append(md[ps:] - md[ps - 1])
        if every(n, len(hold)):
            log(f"collected {n}/{len(hold)} wells")
    return (np.concatenate(trues), np.concatenate(anchors), np.concatenate(raws),
            np.concatenate(wids), np.concatenate(dmds))


def pooled_rmse(true, pred):
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def per_well_median(true, pred, wid):
    out = []
    for w in np.unique(wid):
        m = wid == w
        out.append(np.sqrt(np.mean((true[m] - pred[m]) ** 2)))
    return float(np.median(out))


def report(true, anchor, raw, wid, shrinks=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)):
    base = pooled_rmse(true, anchor)
    log(f"POOLED RMSE  hold-TVT baseline = {base:.4f}  ({len(true)} points, "
        f"{len(np.unique(wid))} wells)")
    best = (base, 0.0)
    for s in shrinks:
        pred = anchor + s * raw
        pr = pooled_rmse(true, pred)
        pw = per_well_median(true, pred, wid)
        flag = "  <-- best" if pr < best[0] else ""
        if pr < best[0]:
            best = (pr, s)
        log(f"  shrink={s:.2f}  pooled={pr:.4f} ({pr-base:+.4f})  per_well_median={pw:.3f}{flag}")
    log(f"BEST pooled={best[0]:.4f} at shrink={best[1]:.2f}  (base {base:.4f})")
    return best


def smooth_within(raw, wid, win):
    """Smooth the predicted residual along MD within each well (order preserved)."""
    if win <= 1:
        return raw
    out = raw.copy()
    for w in np.unique(wid):
        m = wid == w
        out[m] = dp._smooth(raw[m], win)
    return out


if __name__ == "__main__":
    import sys
    feat_kw = {"seq": True} if "seq" in sys.argv else {}
    log(f"evaluate feat_kw={feat_kw}")
    true, anchor, raw, wid, dmd = collect(feat_kw)
    for win in (1, 51, 151, 301, 501):
        log(f"--- residual-smoothing window = {win} ---")
        report(true, anchor, smooth_within(raw, wid, win), wid)
