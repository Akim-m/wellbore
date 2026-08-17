# wellbore

Predicting stratigraphic depth (TVT) along horizontal wells from logging data — a solution to the Kaggle code competition `rogii-wellbore-geology-prediction`.

## The problem

Geosteering a horizontal well means knowing where the drill bit sits *inside the rock layers* — the true vertical thickness, or TVT — at every point along a trajectory that can run for thousands of feet. TVT is known while logging tools have coverage, then has to be inferred.

Each well has a **Prediction Start** (PS) point. Everything before it is measured; everything after it is what the model has to produce.

Scoring is **pooled RMSE in feet across all predicted points**, not a per-well average. Long wells and points far past PS therefore dominate the metric, and a single badly-tracked well can outweigh many good ones.

## The data

773 training wells. Per well:

| File | Contents |
|---|---|
| `{id}__horizontal_well.csv` | `MD, X, Y, Z, GR, TVT_input` (+ `TVT` and markers in train) |
| `{id}__typewell.csv` | The reference log for that well |

Two properties drive the whole design:

- **`TVT_input == TVT` exactly before PS.** The anchor is noise-free, so error comes entirely from what happens after PS.
- **67.7% of pooled MSE is per-well bias** — each well's mean offset from its anchor. A perfect per-well constant alone would take 15.91 → 9.04. Only the remaining third is within-well wiggle.

That split is the reason the shipped model has two stages: a pointwise model cannot see its own per-well bias, because the bias is only visible when you aggregate the well.

## Approach

**Shipped model (v4) — two-stage stack.**

1. **Pointwise stage.** A gradient-boosted model (three-seed ensemble) over 15 engineered per-point features — geometry (`dmd`, `dz`, lateral distance, inclination), GR correlation against a reference curve, and pre-PS context (trend, variability). This captures local variation along the well.
2. **Well-level stage.** A ridge regression over 12 aggregate per-well features, correcting the systematic offset the pointwise stage is blind to.
3. The two are combined with fitted weights against a smoothed anchor.

**GR correlation (`model.py`).** Past PS, each point is assigned the TVT that makes its gamma-ray reading match a reference GR(TVT) curve while staying smooth. It is solved as a dynamic program over a ±40 ft TVT band around the last known TVT, on a 0.5 ft grid, with a smoothness penalty on TVT movement and a prior pulling toward hold-TVT. The reference curve blends the well's own pre-PS GR — same logging tool, so same scale — with the assigned typewell wherever pre-PS gives no coverage.

**Sequence models.** `kaggle_notebook.py` is a later, inference-only submission: 15 dilated-TCN fold models over the same 15 pointwise channels, combined as an OLS-weighted sum across three TCN generations, damped toward hold near PS, with `TVT = anchor + residual`. It also carries a trajectory-content lookup (id-independent X/Y match) so a test well physically present in train gets its exact TVT, and a per-well try/except with hold-TVT fallback so no single well can break a submission run.

## Results

Pooled RMSE in feet, honest 5-fold OOF over all 773 train wells unless noted.

| Model | Pooled RMSE | Note |
|---|---|---|
| Hold-TVT baseline | 15.910 | no model |
| v3 — single GBM | 15.399 | previous ship |
| Pointwise GBM, 3 seeds | 15.266 | |
| **v4 — GBM + ridge stack** | **15.036** | honest leave-fold-out 15.130 |
| v4 on the public leaderboard | **15.318** | submitted and calibrated |
| TCN 3-generation ensemble | 11.699 | |
| Particle-filter tracker, 32 seeds | 10.956 | full 773 wells |
| TCN3 (ramped) + PF blend | **9.431** | fold-honest OLS |

Local OOF 15.04 against LB 15.318 confirms the cross-validation is trustworthy — the two track each other, so local improvements are real rather than fold-fitting. The trajectory lookup matched nothing hidden, which also establishes there is **no train↔test overlap** to exploit.

## Repository layout

```
model.py            GR-correlation dynamic program
features.py         per-point feature construction
train_residual.py   learned-residual model
cv.py               5-fold cross-validation over all 773 wells
validate.py         held-out validation
evaluate.py         scoring
tune.py             hyperparameter search
submit.py           submission assembly
kaggle_notebook.py  the inference-only notebook submission (TCN)
make_notebook.py    builds the notebook from source
download.py         fetches competition data via kagglehub
eda.py              data loading, PS index
experiments/        TCN, particle filter, caches, blending
docs/               design notes and specs
```

## Running it

```bash
pip install numpy pandas scikit-learn torch
python download.py          # fetches the competition data via kagglehub
python cv.py                # 5-fold CV over all 773 wells
python submit.py            # builds submission.csv
```

The Kaggle notebook runs inference only — no internet, no training — against pre-trained fold weights attached as a dataset.

## Notes

- `HANDOFF.md` carries the project fundamentals and the full error analysis; `PROGRESS.md` is the resume state with the experiment scoreboard.
- Long training jobs should be launched detached — the sequence models take well past 20 minutes.
