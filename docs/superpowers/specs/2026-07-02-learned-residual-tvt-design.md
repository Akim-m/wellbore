# Learned-Residual TVT Model — Design

**Date:** 2026-07-02
**Competition:** `rogii-wellbore-geology-prediction`
**Goal:** Beat the hold-TVT baseline on a local well-level holdout (RMSE, ft).

## Problem recap

Predict `TVT` (stratigraphic depth) at each horizontal-well point beyond the
Prediction Start (PS) point. `TVT_input` gives the known TVT up to PS (NaN after).
Metric: RMSE of (trueTVT − predTVT) in feet, over predicted points.

Established facts (measured):
- The driller steers to follow the dipping bed, so **TVT is nearly constant**
  beyond PS. Hold-last-TVT is a strong baseline: ~12.3 mean / ~10 median RMSE.
- Geometric priors that assume TVT tracks depth fail badly (hold-shift ~91,
  dip-plane extrapolation ~1283). The residual `TVT − anchor` is small and
  non-planar.
- A GR-correlation DP model only ties the baseline.
- Wells cluster tightly (766/773 have a neighbour within 5000 ft), so offset-well
  references are available.

## Approach

Predict the **residual** `r = TVT − anchor` where `anchor = TVT_input[PS-1]`, then
`TVT = anchor + r`. Centring on the baseline means a regularised model that
predicts `r ≈ 0` recovers hold-TVT — it cannot do meaningfully worse. The model's
only job is to capture whatever small signal exists.

**Model:** `sklearn.ensemble.HistGradientBoostingRegressor` (one new dependency).
Trained on post-PS points pooled across training wells, subsampled ~1-in-5 for
speed. Loss: squared error (matches the RMSE metric).

## Features (per post-PS point)

v1 (self-features only):
- `dmd` = MD − MD[PS]  (distance drilled past PS)
- `dz` = Z − Z[PS]
- `lat_dist` = horizontal distance from the PS location
- `incl` = local inclination proxy dZ/dMD (windowed)
- `gr` = smoothed GR
- `gr_res_anchor` = GR − reference_GR(anchor): mismatch if TVT were still at anchor
- `gr_xcorr_offset` = TVT offset (±band) that best matches local smoothed GR to the
  reference GR(TVT) curve — injects the geosteering signal as a feature
- `pre_tvt_std`, `pre_tvt_slope` = pre-PS TVT variability / trend context

v2 (add offset-well features):
- `nbr_tvt_level` = TVT of the k nearest *training* wells interpolated at the
  target's lateral position
- `nbr_dip` = local dip inferred from neighbours

Reference GR(TVT): the calibrated curve from `model.py` (typewell + pre-PS blend).

## Validation (the ship gate)

Well-level holdout: split wells 80/20 (deterministic). Train on 80%, predict the
20% held-out wells' post-PS points, compute per-well RMSE. Report mean, median,
and win-rate vs hold-TVT. **A version ships only if it beats hold-TVT on holdout.**
Well-level split (never point-level) prevents within-well leakage. Neighbour
features for a held-out well are drawn only from the training-side wells, matching
test-time conditions.

## Components

- `features.py` — build the per-point feature matrix + residual label for one well;
  reuses `ps_index`, the GR reference, and `_smooth` from `model.py`.
- `neighbors.py` (v2) — nearest-well lookup and neighbour TVT interpolation.
- `train_residual.py` — assemble the pooled training matrix over a well set, fit
  the regressor, persist it.
- `validate_residual.py` — well-level holdout scoring vs hold-TVT.
- `submit.py` (extended) — `--model {hold,residual}`, writes `submission_vN.zip`,
  logs to `versions.json`, repoints `submission.zip` to the best.

## Error handling / edge cases

- Wells with tiny pre-PS (`ps < ~50`): fall back to hold-TVT (too little context).
- All-NaN GR segments: GR features NaN → HGB handles NaN natively.
- Predicted TVT clipped to anchor ± band so a bad prediction can't explode RMSE.
- Neighbour set empty (isolated well): neighbour features NaN → model falls back to
  self-features.

## Out of scope

Deep nets, image (.png) inputs, per-well online learning, external data.
