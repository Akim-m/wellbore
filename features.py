"""Per-point features for the learned-residual TVT model.

For each horizontal-well point beyond PS we build features that describe where
the point is (geometry), what its GR says (correlation to the reference curve),
and the pre-PS context. Label = TVT - anchor (train only).
"""
import numpy as np

import model
from eda import ps_index

SEARCH = 40.0   # half-band (ft) for the GR best-match offset feature
PRE = 200       # pre-PS window for trend/variability context

NAMES = ["dmd", "dz", "lat_dist", "incl", "gr", "gr_res_anchor",
         "gr_offset", "pre_tvt_std", "pre_tvt_slope"]


SEQ_NAMES = ["gr_grad", "gr_lag100", "gr_lag250", "gr_roll_std",
             "inter_slope_dmd", "dmd_frac"]

# well-level summary features (one row per well) for the bias ridge
SUMMARY_NAMES = ["pre_tvt_std", "pre_tvt_slope", "ref_slope", "align_delta",
                 "align_slope", "align_ev", "mean_gr_res", "med_implied_off",
                 "med_gr_offset", "med_gr_res_anchor", "log_n", "log_dmd"]
ALIGN_DELTAS = np.arange(-20.0, 20.5, 1.0)        # constant TVT offset (ft)
ALIGN_SLOPES = np.arange(-0.004, 0.0042, 0.0004)  # TVT drift per ft of MD


def well_features(hw, tw, with_label=True, seq=False):
    ps = ps_index(hw)
    n = len(hw)
    md = hw["MD"].to_numpy()
    x, y, z = hw["X"].to_numpy(), hw["Y"].to_numpy(), hw["Z"].to_numpy()
    anchor = hw["TVT_input"].to_numpy()[ps - 1]

    grid = np.arange(anchor - model.BAND, anchor + model.BAND + model.GRID, model.GRID)
    gr_h, ref = model._reference(grid, hw, ps, tw)   # smoothed z-scored GR + reference
    a_idx = int(round((anchor - grid[0]) / model.GRID))
    ref_anchor = ref[a_idx] if np.isfinite(ref[a_idx]) else np.nanmean(ref)

    post = slice(ps, n)
    dmd = md[post] - md[ps - 1]
    dz = z[post] - z[ps - 1]
    lat = np.hypot(x[post] - x[ps - 1], y[post] - y[ps - 1])
    incl = np.gradient(z, md)[post]
    gr = gr_h[post]
    gr_res = gr - ref_anchor

    # GR best-match offset: TVT near anchor whose reference GR best matches gr[i].
    lo = max(0, a_idx - int(SEARCH / model.GRID))
    hi = min(len(grid), a_idx + int(SEARCH / model.GRID) + 1)
    sub = ref[lo:hi]
    cost = (gr[:, None] - sub[None, :]) ** 2
    cost[:, ~np.isfinite(sub)] = np.inf
    cost[~np.isfinite(gr), :] = 0.0
    gr_offset = (lo + np.argmin(cost, axis=1)) * model.GRID + grid[0] - anchor

    # pre-PS context (last PRE known points), broadcast to all post points.
    tvt_pre = hw["TVT_input"].to_numpy()[:ps]
    k = min(ps, PRE)
    seg, mdseg = tvt_pre[ps - k:ps], md[ps - k:ps]
    pre_std = np.nanstd(seg)
    pre_slope = np.polyfit(mdseg, seg, 1)[0] if k > 2 else 0.0

    npost = n - ps
    cols = [dmd, dz, lat, incl, gr, gr_res, gr_offset,
            np.full(npost, pre_std), np.full(npost, pre_slope)]
    names = list(NAMES)

    if seq:   # local GR-sequence context: shape/trend/texture around each point
        gr_grad = np.gradient(gr_h, md)[post]
        gr_lag100 = np.interp(md[post] - 100, md, gr_h) - gr        # GR 100 ft back
        gr_lag250 = np.interp(md[post] - 250, md, gr_h) - gr        # GR 250 ft back
        roll = np.sqrt(np.clip(model._smooth(gr_h ** 2, 101)
                               - model._smooth(gr_h, 101) ** 2, 0, None))[post]
        cols += [gr_grad, gr_lag100, gr_lag250, roll,
                 pre_slope * dmd,                # pre-PS TVT trend continued
                 dmd / max(dmd[-1], 1.0)]        # position within predicted region
        names = names + SEQ_NAMES

    X = np.column_stack(cols).astype(float)
    X[~np.isfinite(X)] = np.nan   # HGB handles NaN natively

    label = None
    if with_label and "TVT" in hw.columns:
        label = hw["TVT"].to_numpy()[post] - anchor
    return X, names, label, ps, anchor


def well_summary(hw, tw, X, names, ps, anchor):
    """One row of well-level features for the bias ridge (SUMMARY_NAMES order).

    GR-alignment evidence pooled over the whole post-PS series: a parametric
    fit TVT = anchor + delta + slope*dmd matched against the reference GR
    curve, plus medians of the pointwise GR-mismatch features.
    """
    dmd = X[:, names.index("dmd")]
    gr = X[:, names.index("gr")]
    gr_res = X[:, names.index("gr_res_anchor")]
    gr_off = X[:, names.index("gr_offset")]
    pre_std = X[0, names.index("pre_tvt_std")]
    pre_slope = X[0, names.index("pre_tvt_slope")]

    grid = np.arange(anchor - model.BAND, anchor + model.BAND + model.GRID, model.GRID)
    _, ref = model._reference(grid, hw, ps, tw)
    a_idx = int(round((anchor - grid[0]) / model.GRID))

    # local slope of the reference GR at the anchor (z-units per ft of TVT)
    lo, hi = max(0, a_idx - 4), min(len(grid), a_idx + 5)
    seg, gseg = ref[lo:hi], grid[lo:hi]
    ok = np.isfinite(seg)
    ref_slope = np.polyfit(gseg[ok], seg[ok], 1)[0] if ok.sum() > 3 else np.nan

    # implied TVT offset from GR mismatch under local linearization
    if np.isfinite(ref_slope):
        denom = np.sign(ref_slope) * max(abs(ref_slope), 0.02)
        implied = np.clip(gr_res / denom, -model.BAND, model.BAND)
    else:
        implied = np.full(len(gr_res), np.nan)

    # parametric alignment: TVT = anchor + delta + slope*dmd, matched to ref
    paths = (anchor + ALIGN_DELTAS[:, None, None]
             + ALIGN_SLOPES[None, :, None] * dmd[None, None, :])
    ref_at = np.interp(paths.ravel(), grid, ref).reshape(paths.shape)
    d2 = (gr[None, None, :] - ref_at) ** 2
    fin = np.isfinite(d2)
    cnt = fin.sum(axis=2)
    cost = np.where(cnt > 0,
                    np.nansum(np.where(fin, d2, 0.0), axis=2) / np.maximum(cnt, 1),
                    np.inf)
    if np.isfinite(cost).any() and cnt.max() >= 0.3 * len(dmd):
        bi = np.unravel_index(np.argmin(cost), cost.shape)
        a_delta, a_slope = float(ALIGN_DELTAS[bi[0]]), float(ALIGN_SLOPES[bi[1]])
        c00 = cost[len(ALIGN_DELTAS) // 2, len(ALIGN_SLOPES) // 2]
        ev = float(c00 - cost[bi]) if np.isfinite(c00) else np.nan
    else:
        a_delta = a_slope = ev = np.nan

    def med(a):
        return np.nanmedian(a) if np.isfinite(a).any() else np.nan

    row = np.array([pre_std, pre_slope, ref_slope, a_delta, a_slope, ev,
                    np.nanmean(gr_res) if np.isfinite(gr_res).any() else np.nan,
                    med(implied), med(gr_off), med(gr_res),
                    np.log1p(len(dmd)), np.log1p(float(dmd[-1]))], dtype=float)
    row[~np.isfinite(row)] = np.nan
    return row


if __name__ == "__main__":
    from model import load_well
    hw, tw = load_well("train", "000d7d20")
    X, names, label, ps, anchor = well_features(hw, tw, seq=True)
    assert X.shape == (len(hw) - ps, len(names)), X.shape
    assert label.shape[0] == X.shape[0]
    assert not np.isinf(X[np.isfinite(X)]).any()
    s = well_summary(hw, tw, X, names, ps, anchor)
    assert s.shape == (len(SUMMARY_NAMES),)
    print(f"OK: X={X.shape}, features={names}, anchor={anchor:.1f}, "
          f"label range={label.min():.1f}..{label.max():.1f}")
    print("summary:", dict(zip(SUMMARY_NAMES, np.round(s, 3))))
