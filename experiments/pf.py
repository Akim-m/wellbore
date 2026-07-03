"""Particle-filter TVT tracker (port of sunnywu27's public physical model),
vectorized across seeds. Honest by construction (no training) — evaluated
pooled over all train wells and saved as an OOF-style raw for stacking.

Usage: pf.py [n_wells]   (default all; small number for a quick check)
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
SEEDS = int(os.environ.get("PF_SEEDS", 32))
NPART = 500
MOM, VN, PN, RP, RR, RESAMP = 0.998, 0.002, 0.005, 0.1, 0.001, 0.5
LIK_SCALE = 5.0
SCALES = (3.0, 5.0, 8.0, 12.0)   # per-scale outputs when MULTISCALE=1
MULTISCALE = os.environ.get("PF_MULTISCALE", "0") == "1"


def pf_well(hw, tw, n_seeds=SEEDS, npart=NPART):
    tw_s = tw.dropna(subset=["TVT"]).sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)

    known = hw["TVT_input"].notna().to_numpy()
    ev = np.where(~known)[0]
    kn = np.where(known)[0]
    if len(ev) == 0 or len(kn) < 5:
        return None
    last = kn[-1]
    ev = ev[ev > last]                      # suffix only (interior gaps interp'd elsewhere)
    if len(ev) == 0:
        return None

    tin = hw["TVT_input"].to_numpy(float)
    z = hw["Z"].to_numpy(float)
    md = hw["MD"].to_numpy(float)

    tw_at_k = np.interp(tin[kn], tw_tvt, tw_gr)
    gr_kn = hw["GR"].to_numpy(float)[kn]
    gs = float(np.clip(np.nanstd(np.nan_to_num(gr_kn) - tw_at_k), 10.0, 60.0))

    t30 = kn[-30:]
    dt, dz, dm = np.diff(tin[t30]), np.diff(z[t30]), np.diff(md[t30])
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    gr_interp = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean())
    gr_v = gr_interp.to_numpy(float)[ev]
    md_v, z_v = md[ev], z[ev]

    S, N = n_seeds, npart
    rng = [np.random.default_rng(s) for s in range(S)]
    ls = tin[last] + z[last]
    pos = np.stack([ls + 2.0 * r.standard_normal(N) for r in rng])       # (S,N)
    rate = np.stack([ir + 0.01 * r.standard_normal(N) for r in rng])
    w = np.full((S, N), 1.0 / N)
    loglik = np.zeros(S)
    res = np.empty((S, len(ev)))
    g = np.random.default_rng(12345)        # shared process noise across seeds is fine
    prev = md[last]

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev, 1.0)
        rate = MOM * rate + VN * g.standard_normal((S, N))
        pos = pos + rate * dm_step + PN * g.standard_normal((S, N))
        tvt_p = np.clip(pos - z_v[i], tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]

        eg = np.interp(tvt_p.ravel(), tw_tvt, tw_gr).reshape(S, N)
        d = (gr_v[i] - eg) / gs
        lk = np.maximum(np.exp(-0.5 * np.minimum(d * d, 600.0)), 1e-300)
        avg = (w * lk).sum(1)
        loglik += np.log(np.maximum(avg, 1e-300))
        w = w * lk
        ws = w.sum(1, keepdims=True)
        w = np.where(ws > 0, w / np.maximum(ws, 1e-300), 1.0 / N)

        neff = 1.0 / (w * w).sum(1)
        need = neff < RESAMP * N
        if need.any():
            for s in np.where(need)[0]:
                cum = np.cumsum(w[s])
                u0 = g.uniform(0, 1.0 / N)
                idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
                pos[s] = pos[s][idx] + RP * g.standard_normal(N)
                rate[s] = rate[s][idx] + RR * g.standard_normal(N)
                w[s] = 1.0 / N
        res[:, i] = (w * (pos - z_v[i])).sum(1)
        prev = md_v[i]

    ln = loglik - loglik.max()
    if MULTISCALE:
        outs = []
        for sc in SCALES:
            sw = np.exp(ln / sc)
            sw /= sw.sum()
            outs.append((sw[:, None] * res).sum(0))
        return ev, np.stack(outs)        # (4, n)
    sw = np.exp(ln / LIK_SCALE)
    sw /= sw.sum()
    return ev, (sw[:, None] * res).sum(0)


def main():
    wells = sorted(os.path.basename(p).replace("__horizontal_well.csv", "")
                   for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))
    if len(sys.argv) == 3:                       # shard: pf.py START END
        lo, hi = int(sys.argv[1]), int(sys.argv[2])
        wells, shard = wells[lo:hi], f"_{lo}_{hi}"
    else:
        lim = int(sys.argv[1]) if len(sys.argv) > 1 else len(wells)
        wells, shard = wells[:lim], ""
    parts, errs = {}, []
    for n, w in enumerate(wells, 1):
        hw = pd.read_csv(os.path.join(DATA, "train", f"{w}__horizontal_well.csv"))
        tw = pd.read_csv(os.path.join(DATA, "train", f"{w}__typewell.csv"))
        try:
            r = pf_well(hw, tw)
        except Exception as e:
            log(f"{w}: PF failed {e}")
            r = None
        true = hw["TVT"].to_numpy(float)
        anchor_row = np.where(hw["TVT_input"].notna().to_numpy())[0][-1]
        anchor = hw["TVT_input"].to_numpy(float)[anchor_row]
        if r is None:
            continue
        ev, pred = r
        parts[w] = (ev, pred, anchor)
        e = (pred[1] if pred.ndim == 2 else pred) - true[ev]   # scale-5 row for progress
        h = anchor - true[ev]
        errs.append((float(np.sum(e * e)), float(np.sum(h * h)), len(e)))
        if n % 25 == 0 or n == len(wells):
            se, sh, c = map(sum, zip(*errs))
            log(f"{n}/{len(wells)} pooled: pf={np.sqrt(se/c):.3f} hold={np.sqrt(sh/c):.3f}")
    se, sh, c = map(sum, zip(*errs))
    log(f"FINAL pooled: pf={np.sqrt(se/c):.4f} hold={np.sqrt(sh/c):.4f} ({c} pts, {len(parts)} wells)")
    tag = "pf_ms" if MULTISCALE else "pf_preds"
    out = {}
    for w, (e, p, a) in parts.items():
        if p.ndim == 2:
            out[w] = p.astype(np.float32)
            out[f"{w}__a"] = np.array([a], np.float32)
        else:
            out[w] = np.concatenate([[a], p]).astype(np.float32)
    np.savez_compressed(os.path.join(HERE, f"{tag}{shard}.npz"), **out)


if __name__ == "__main__":
    main()
