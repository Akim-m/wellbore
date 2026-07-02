"""GR-correlation TVT model.

Beyond the Prediction Start point we assign each horizontal-well point a TVT
value that (a) makes its GR match a reference GR(TVT) curve and (b) stays
smooth. Solved as a dynamic program over a TVT band centred on the last known
TVT. The reference curve blends the well's own pre-PS GR (same logging tool,
best match) with the assigned typewell where pre-PS gives no coverage.
"""
import os

import numpy as np
import pandas as pd

from eda import DATA, ps_index

GRID = 0.5      # TVT grid spacing (ft)
BAND = 40.0     # half-width of TVT search band around last known TVT (ft)
STEP = 0.5      # max |TVT change| between adjacent points (ft)
LAM = 2.0       # smoothness weight (penalty on TVT movement)
ALPHA = 0.1     # anchor prior (pull toward hold-TVT)
SMOOTH = 25     # GR moving-average window (points ~= ft)


def _zscore(x):
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / s if s > 0 else x - m


def _smooth(x, w):
    """NaN-aware centred moving average."""
    if w <= 1:
        return x
    v = np.where(np.isfinite(x), x, 0.0)
    m = np.isfinite(x).astype(float)
    k = np.ones(w)
    num = np.convolve(v, k, "same")
    den = np.convolve(m, k, "same")
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def _reference(grid, hw, ps, tw):
    """Reference GR (z-scored) on the TVT grid, blending pre-PS horizontal and typewell."""
    gr_h = _smooth(_zscore(hw["GR"].to_numpy()), SMOOTH)  # one tool -> one scale, denoised
    tvt_pre = hw["TVT_input"].to_numpy()[:ps]  # TVT_input == known TVT before PS
    gr_pre = gr_h[:ps]

    # typewell, z-scored on its own, interpolated onto the grid
    tw = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    tw_ref = np.interp(grid, tw["TVT"], _zscore(tw["GR"].to_numpy()),
                       left=np.nan, right=np.nan)

    # pre-PS horizontal binned onto the grid (mean per cell)
    ok = np.isfinite(tvt_pre) & np.isfinite(gr_pre)
    idx = np.round((tvt_pre[ok] - grid[0]) / GRID).astype(int)
    inb = (idx >= 0) & (idx < len(grid))
    pre_sum = np.zeros(len(grid))
    cnt = np.zeros(len(grid))
    np.add.at(pre_sum, idx[inb], gr_pre[ok][inb])
    np.add.at(cnt, idx[inb], 1)
    pre_ref = np.where(cnt > 0, pre_sum / np.maximum(cnt, 1), np.nan)

    # blend: prefer pre-PS where available, else typewell
    ref = np.where(np.isfinite(pre_ref), pre_ref, tw_ref)
    return gr_h, ref


def _dp(gr_post, ref, start_state):
    """Viterbi over TVT states. Returns state index per post-PS point."""
    n, m = len(gr_post), len(ref)
    k = int(round(STEP / GRID))
    offs = np.arange(-k, k + 1)
    move_cost = LAM * (offs * GRID) ** 2         # penalty for each TVT step

    emit = np.where(np.isfinite(gr_post)[:, None] & np.isfinite(ref)[None, :],
                    (gr_post[:, None] - ref[None, :]) ** 2, 0.0)
    emit[:, ~np.isfinite(ref)] = 5.0             # discourage states with no reference
    anchor_pen = ALPHA * ((np.arange(m) - start_state) * GRID) ** 2
    emit = emit + anchor_pen[None, :]            # pull toward the hold-TVT value

    BIG = 1e12
    cost = np.full(m, BIG)
    cost[start_state] = 0.0
    back = np.empty((n, m), dtype=np.int16)

    for i in range(n):
        # for each target state t, min over source t-off of cost[t-off]+move
        cand = np.full((len(offs), m), BIG)
        for j, o in enumerate(offs):
            lo, hi = max(0, o), min(m, m + o)     # target range reachable from valid source
            src_lo, src_hi = lo - o, hi - o
            cand[j, lo:hi] = cost[src_lo:src_hi] + move_cost[j]
        best = np.argmin(cand, axis=0)
        cost = cand[best, np.arange(m)] + emit[i]
        back[i] = np.arange(m) - offs[best]       # source state chosen
        back[i] = np.clip(back[i], 0, m - 1)

    path = np.empty(n, dtype=np.int64)
    t = int(np.argmin(cost))
    for i in range(n - 1, -1, -1):
        path[i] = t
        t = int(back[i, t])
    return path


def predict_tvt(hw, tw):
    """Return TVT for every row: known TVT_input before PS, predicted after."""
    ps = ps_index(hw)
    out = hw["TVT_input"].to_numpy().copy()
    if ps >= len(hw):
        return out
    anchor = out[ps - 1]
    grid = np.arange(anchor - BAND, anchor + BAND + GRID, GRID)

    gr_h, ref = _reference(grid, hw, ps, tw)
    start = int(round((anchor - grid[0]) / GRID))
    path = _dp(gr_h[ps:], ref, start)
    out[ps:] = grid[path]
    return out


def load_well(split, well):
    hw = pd.read_csv(os.path.join(DATA, split, f"{well}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(DATA, split, f"{well}__typewell.csv"))
    return hw, tw


if __name__ == "__main__":
    hw, tw = load_well("test", "000d7d20")
    pred = predict_tvt(hw, tw)
    ps = ps_index(hw)
    print(f"predicted {len(hw) - ps} points; TVT {pred[ps:].min():.1f}..{pred[ps:].max():.1f}")
