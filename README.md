# Wellbore Geology Prediction

Predicting stratigraphic depth along horizontal oil wells with temporal deep
learning and Bayesian state-space tracking.

Built for the **Rogii "Wellbore Geology Prediction"** Kaggle code competition
(2026). The task is to predict true vertical thickness (TVT) — the depth of a
geological marker — along the horizontal section of a well, *beyond* a known
"Prediction Start" point, given the trajectory and gamma-ray (GR) logs up to
that point plus an offset "typewell." Submissions are scored by pooled RMSE in
feet across **773 training wells**.

## Approach

The pipeline evolved through several model generations, each validated against
an honest cross-validation gate before being kept:

1. **Two-stage residual model** — a `HistGradientBoostingRegressor` predicts a
   coarse depth, with a Ridge stage correcting per-well bias.
2. **Dilated Temporal Convolutional Network (PyTorch)** — a 405k-parameter TCN
   of 16 residual conv blocks with exponentially increasing dilations (1–128),
   GroupNorm/GELU, trained in bf16. It reads the along-well GR and geometry
   sequence and regresses the depth track directly.
3. **Likelihood-weighted particle filter** — a vectorized, 32–128-seed
   sequential Monte Carlo tracker that treats the marker depth as a latent
   state and updates it against a GR-vs-typewell cost volume, giving a
   physically plausible, monotonicity-aware path.
4. **Ensemble** — the TCN, particle filter, and residual model are blended with
   leave-fold-out OLS weights, exploiting their decorrelated error tails.

Supporting techniques include PS-augmentation (re-cutting wells at synthetic
prediction-start points to enlarge the training set), beam-search dynamic-
programming path trackers, and a trajectory-content lookup to detect
train/test overlap.

## Results

All figures are honest 5-fold, well-level, out-of-fold (OOF) pooled RMSE in
feet — no point-level leakage:

| Model | OOF RMSE (ft) |
| --- | --- |
| Hold baseline | 15.91 |
| Gradient-boosting residual | 15.04 |
| Dilated TCN | 12.63 |
| **TCN + particle-filter ensemble** | **9.27** |

The local OOF score (15.04) tracked the public leaderboard (15.32) to within
0.3 ft, confirming the cross-validation is trustworthy rather than optimistic.

## Rigor and validation

- **Leakage-free CV gate** (`cv.py`): a 5-fold, *well-level* OOF harness over
  all 773 wells that reproduces the exact shipped pipeline. Splitting on whole
  wells (never individual points) prevents a well's own data leaking across the
  train/test boundary.
- **Documented scoreboard**: every model generation is recorded with its pooled
  RMSE, alongside an explicit ledger of *measured negative results* ("don't
  retry") to keep the search space pruned.
- The repository is competition/research code (~3,600 LOC Python). It relies on
  the CV harness rather than a formal unit-test suite.

## Tech stack

Python · PyTorch (dilated 1D-TCN, bf16 autocast) · scikit-learn
(`HistGradientBoostingRegressor`, Ridge) · NumPy / pandas (vectorized particle
filter, beam-search DP, cost-volume features).

## Layout

```text
features.py          GR / geometry / typewell feature construction
train_residual.py    two-stage gradient-boosting + Ridge model
evaluate.py          scoring and submission assembly
cv.py                5-fold well-level OOF gate
experiments/         research scripts for the TCN, particle filter, ensembling
```
