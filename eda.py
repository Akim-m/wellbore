"""Explore the wellbore data and measure simple TVT baselines on train.

Task: predict TVT (stratigraphic depth) along a horizontal well beyond the
Prediction Start (PS) point. TVT is known up to PS (that's TVT_input); NaN after.
Metric: RMSE of (trueTVT - predTVT) over predicted points, per foot.
"""
import glob
import os

import numpy as np
import pandas as pd

from progress import every, log

DATA = os.path.expanduser(
    "~/.cache/kagglehub/competitions/rogii-wellbore-geology-prediction"
)


def ps_index(hw):
    """Row index of the Prediction Start point = first NaN in TVT_input."""
    nan = hw["TVT_input"].isna()
    return int(nan.idxmax()) if nan.any() else len(hw)


def summarize():
    wells = sorted(
        os.path.basename(p).replace("__horizontal_well.csv", "")
        for p in glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv"))
    )
    print(f"train wells: {len(wells)}")

    rows, ps_fracs, rmse_const, rmse_linear = [], [], [], []
    for i, w in enumerate(wells, 1):
        hw = pd.read_csv(os.path.join(DATA, "train", f"{w}__horizontal_well.csv"))
        ps = ps_index(hw)
        n = len(hw)
        if every(i, len(wells)):
            log(f"eda: {i}/{len(wells)} wells")
        if ps >= n or ps < 2:
            continue
        rows.append(n)
        ps_fracs.append(ps / n)

        true = hw["TVT"].to_numpy()
        # Baseline A: hold last known TVT constant.
        const = true[ps - 1]
        rmse_const.append(np.sqrt(np.mean((true[ps:] - const) ** 2)))

        # Baseline B: linear extrapolation of TVT vs MD from the known segment.
        md = hw["MD"].to_numpy()
        k = min(ps, 200)  # fit on the last 200 known points
        slope, intercept = np.polyfit(md[ps - k:ps], true[ps - k:ps], 1)
        pred = slope * md[ps:] + intercept
        rmse_linear.append(np.sqrt(np.mean((true[ps:] - pred) ** 2)))

    rmse_const = np.array(rmse_const)
    rmse_linear = np.array(rmse_linear)
    print(f"usable wells: {len(rmse_const)}")
    print(f"horizontal rows: median={int(np.median(rows))}, max={max(rows)}")
    print(f"PS fraction of well: median={np.median(ps_fracs):.2f}")
    print(f"predicted-region span (feet) ~ per-point RMSE below\n")
    print(f"Baseline A (hold constant):   mean RMSE={rmse_const.mean():8.2f}  median={np.median(rmse_const):8.2f}")
    print(f"Baseline B (linear extrap):   mean RMSE={rmse_linear.mean():8.2f}  median={np.median(rmse_linear):8.2f}")

    # Inspect one well's TVT excursion to gauge scale.
    hw = pd.read_csv(os.path.join(DATA, "train", f"{wells[0]}__horizontal_well.csv"))
    ps = ps_index(hw)
    print(f"\nexample {wells[0]}: rows={len(hw)}, PS={ps}")
    print(f"  TVT range whole well: {hw['TVT'].min():.1f}..{hw['TVT'].max():.1f}")
    print(f"  TVT beyond PS moves by {hw['TVT'].iloc[-1]-hw['TVT'].iloc[ps]:+.1f} ft")


if __name__ == "__main__":
    summarize()
