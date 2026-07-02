# Wellbore Geology Prediction — Handoff

Status as of 2026-07-02 (evening). Read this first, then `docs/superpowers/` for older design notes.

## TL;DR
- Kaggle **Code Competition** `rogii-wellbore-geology-prediction`. Predict `TVT`
  (stratigraphic depth) along each horizontal well beyond the Prediction Start (PS) point.
- **Metric = pooled RMSE over all predicted points** (ft). Not per-well average.
- **Current best model = v4 (two-stage)**: shipped-pipeline OOF pooled **15.036**
  (honest leave-fold-out stack weights: 15.130) vs 15.910 hold-TVT, validated by
  5-fold CV over all 773 train wells (`cv.py`).
- **SUBMITTED & CALIBRATED (2026-07-02): public LB = 15.318** (kernel
  `aydhin/wellbore-v4-two-stage` v3, submission ref 54264023). Local OOF 15.04 ≈ LB
  15.32 — the CV is trustworthy. The lookup matched nothing hidden → **no
  train↔test overlap**; the top of the LB (5.262, pack 5.4–6.5) is genuine
  within-well TVT tracking, ~3× better than hold everywhere.
- **The gap to 5 is pure modeling.** Priority: windowed GR correlation (see
  "Plan to attack ~5"). Final deadline 2026-08-05 (~5 weeks).
- Submit via kaggle CLI (`kaggle kernels push` + MCP submit tool). NOTE: the MCP
  `save_notebook` tool silently drops competition data sources — use the CLI
  (metadata in scratchpad pattern: kernel-metadata.json with `competition_sources`).

## The competition
- **Code competition**: submit a Kaggle Notebook that outputs `submission.csv`;
  no CSV upload. CPU/GPU ≤ 9h, **internet disabled**, external public data allowed.
- **Metric**: RMSE of `(trueTVT − predTVT)` across ALL predicted points, pooled.
  Long wells and far-from-PS points dominate.
- **Submission format**: `id,tvt`, `id = {well}_{rowindex}`, one row per point beyond PS.
- **Deadlines**: entry 2026-07-29, final submission 2026-08-05.

## The data
- Local: `~/.cache/kagglehub/competitions/rogii-wellbore-geology-prediction/`
  (via `download.py`). On Kaggle auto-mounts under `/kaggle/input/...`.
- Per well: `{id}__horizontal_well.csv`, `{id}__typewell.csv`; train also has `{id}.png`.
  773 train wells; local `test/` = 3 sample wells (hidden test larger).
- Horizontal columns: `MD,X,Y,Z,GR,TVT_input` (+`TVT`, markers in train).
  `TVT_input` = TVT up to PS, NaN after (PS = first NaN). **`TVT_input == TVT`
  exactly pre-PS** (verified — the anchor is noise-free).
- **Test spans are full-length** (sample: 3.8k–6k ft beyond PS; train median 4.8k),
  so the hidden test is NOT short-horizon-easy.
- **Overlap facts (verified locally)**: the 3 sample test wells exist in train
  **byte-identical, same ids**. Typewells are 752/773 distinct (one group of 10,
  12 pairs) — no useful sharing. No duplicate trajectories within train; 10 pairs
  share a surface location (sidetracks).

## Error structure (drives everything)
- **67.7% of pooled MSE is per-well bias** (each well's mean offset from anchor);
  perfect per-well constant ⇒ 15.91 → 9.04. Only 32% is within-well wiggle.
- Residual drifts slightly positive with distance (+1.5–1.8 ft mean at 1–6k ft).
- To beat ~9 (and reach 5) you must track within-well variation: that means making
  GR-vs-reference correlation actually work, not incremental GBM features.

## The model (v4, what ships)
Pipeline: `features.py` → `train_residual.py`; `kaggle_notebook.py` = self-contained copy.

1. Anchor = last known TVT. Predict residual `r = TVT − anchor`;
   `TVT = anchor + W_PT*smooth(gbm) + W_WELL*ridge_const`, `W_PT=0.467, W_WELL=0.560`
   (full-OOF OLS stack weights).
2. **Pointwise GBM** (3-seed ensemble, mean): 15 features = v3's 13
   (geometry `dmd,dz,lat_dist,incl`; GR-match `gr,gr_res_anchor,gr_offset`;
   pre-PS `pre_tvt_std,pre_tvt_slope`; sequence `gr_grad,gr_lag100,gr_lag250,
   gr_roll_std`) + **`inter_slope_dmd` (pre_slope×dmd) + `dmd_frac`**.
   HP: `max_iter=200, lr=0.05, l2=5.0, min_samples_leaf=1000`. Residual path
   clipped ±40 then smoothed ×301 within well.
3. **Well-bias Ridge** (`well_summary` in features.py): 12 well-level features —
   pre-PS stats, `ref_slope` at anchor, **parametric GR alignment** (best δ, slope,
   evidence of `TVT = anchor+δ+s·dmd` matched to the reference GR over ±20 ft /
   ±0.004), mean/median GR-residual stats, log n, log dmd. α=10, impute
   train-medians, standardize, sample_weight = points/well. Predicts the well's
   n-weighted mean residual. **This alone (15.366) beats the entire v3 (15.399).**
4. Wells with `PS < 50` fall back to hold-TVT.
5. **Notebook extra — trajectory-content lookup**: a test well whose X/Y trajectory
   matches a train well (id-independent; rows equal, |ΔX|,|ΔY|<0.1 at every 200th
   point) gets the train well's exact TVT (~0 error). Covers the sample leak and
   any hidden-test overlap; harmless otherwise. The LB score of this submission
   diagnoses whether hidden-test overlap exists (score ≪ local ⇒ overlap).

## Results (pooled OOF over all 773 wells)
| Approach | Pooled RMSE | Note |
|---|---|---|
| hold-TVT (baseline) | 15.910 | carry last known TVT |
| v3 (single GBM, shrink 0.5) | 15.399 | previous ship |
| well-Ridge alone | 15.366 | one constant per well! |
| pointwise GBM +inter feats, 3 seeds | 15.266 | |
| **v4: stack (GBM + Ridge)** | **15.036** (honest 15.130) | ships; `cv.py` gate |

Negative results (measured, don't retry): align/refslope features as POINTWISE GBM
inputs (15.58–15.59 — they only pay at well level); well-level GBM instead of Ridge
(15.76 — overfits); dmd-binned recalibration after stacking (hurts); 3-model stack
with leads (unstable weights). Older dead ends unchanged: neighbor-well transfer
(~97), dip-plane (~1283), hold-shift (~91), GR-DP pointwise (ties), trajectory
features (overfit), more GBM capacity (worse).

## Plan to attack ~5 — SUPERSEDED by the TCN (2026-07-02 evening)
- **Windowed GR correlation is measured DEAD** (`diag2.py`): free window matching
  lands median 24–28 ft from truth vs hold's 5–11. Don't retry correlation search.
- **PS augmentation works**: re-cut each train well at 0.5/0.65/0.8/1.0×PS
  (true TVT known everywhere) → 3,092 cuts / 16.5M pts (`aug_cache.py`).
  Aug-GBM 14.88 at shrink 1.0; aug hurts the ridge (keep ridge on true cuts).
- **TCN sequence model (scratchpad `seq.py`) — honest 5-fold OOF 12.63 ALONE**,
  classical stack adds nothing. 405k-param dilated TCN over the 15 feature
  channels, trained on aug cuts, ~4 min/fold on the local RTX 4060.
  Progression: 15.91 hold → 15.13 v4 → 14.81 aug-stack → **12.63 TCN**.
- **Deployment**: train locally, ship weights (~1.6 MB) as a private Kaggle
  dataset, notebook = CPU inference only. No Kaggle-side training.
- Next: v2 (60 ep, 96 ch, length-prop sampling), pre-PS context channels,
  denser cuts, seed ensembles. Iterate fold-0-gated, 5-fold for the honest number.

## File inventory
- `features.py` — `well_features` (15 pointwise feats) + `well_summary` (12
  well-level feats incl. parametric alignment) + constants.
- `train_residual.py` — `fit_model(wells)` → bundle {3 GBMs, ridge, imputation},
  `predict_well(bundle, hw, tw)`; stack weights `W_PT/W_WELL`.
- `cv.py` — **the gate**: 5-fold OOF of the exact shipped pipeline (~7 min).
- `evaluate.py` — 80/20 pooled-RMSE harness + shrink/smoothing sweeps (updated for bundle).
- `validate_residual.py` — per-well-mean holdout (secondary lens).
- `model.py` — legacy GR-DP; provides `_reference`, `_smooth`, `load_well`, constants.
- `submit.py` — versioned local submissions + `versions.json` promotion (pure model,
  no lookup — the notebook carries the lookup).
- `kaggle_notebook.py` / `make_notebook.py` — self-contained submission + ipynb wrapper.
- `eda.py`, `progress.py`, `tune.py`, `validate.py` — unchanged utilities/legacy.
- `.mcp.json` — Kaggle MCP (`mcp-remote`); token via `KAGGLE_API_TOKEN` env var
  (never in the file). Restart Claude Code from a fresh terminal + approve server;
  then MCP tools handle upload/submission.

## Environment / auth
- venv: `.\.venv\Scripts\python.exe` (Python 3.14).
- Kaggle `KGAT_` token in `KAGGLE_API_TOKEN` (User env var). Fresh PowerShell:
  `$env:KAGGLE_API_TOKEN = [Environment]::GetEnvironmentVariable("KAGGLE_API_TOKEN","User")`.

## How to reproduce / validate / submit
```powershell
.\.venv\Scripts\python.exe cv.py                     # gate (~7 min) → 15.036
.\.venv\Scripts\python.exe submit.py --model residual --tag v4 --mean 15.036
.\.venv\Scripts\python.exe make_notebook.py          # → kaggle_notebook.ipynb
.\.venv\Scripts\python.exe kaggle_notebook.py        # local end-to-end check
```
Kaggle UI path: Code → New Notebook → File → Import Notebook → `kaggle_notebook.ipynb`
→ Run All → Save Version → Submit (internet off). Or via Kaggle MCP once loaded.
