# Learned-Residual TVT Model — Implementation Plan

> **For agentic workers:** executed inline in this session. Steps use checkbox
> (`- [ ]`) syntax. Verification is the holdout-RMSE gate + inline `assert`
> self-checks (no pytest suite — YAGNI). No git repo, so no commit steps.

**Goal:** A gradient-boosted model predicting the TVT residual past PS that beats
hold-TVT on a well-level holdout, shipped as versioned submissions.

**Architecture:** Predict `r = TVT − anchor` from per-point features (self, then
offset-well), add back the anchor, clip to a band. Well-level 80/20 holdout gates
each version.

**Tech Stack:** Python 3.14 (`.venv`), numpy, pandas, scikit-learn (new).

## Global Constraints

- Run everything via `.\.venv\Scripts\python.exe`.
- Data root: `~/.cache/kagglehub/competitions/rogii-wellbore-geology-prediction`.
- Reuse `eda.ps_index`, `model._reference`, `model._smooth`, `model.load_well`.
- Anchor = `TVT_input[ps-1]`. Final TVT clipped to `anchor ± model.BAND`.
- Wells with `ps < 50` → hold-TVT fallback.
- A version ships only if holdout mean RMSE ≤ hold-TVT mean RMSE.

---

### Task 1: Feature builder (`features.py`)

**Files:** Create `features.py`; reuses `model.py`, `eda.py`.

**Interfaces:**
- Produces: `well_features(hw, tw, *, with_label) -> (X: np.ndarray[n_post, F], names: list[str], label: np.ndarray|None, ps: int, anchor: float)`.
  `n_post = len(hw) - ps`. `label = TVT[ps:] - anchor` when `with_label` else None.

- [ ] **Step 1:** Install scikit-learn: `.\.venv\Scripts\python.exe -m pip install --quiet scikit-learn`.
- [ ] **Step 2:** Implement `well_features`. Features per post-PS point:
  `dmd, dz, lat_dist, incl (windowed dZ/dMD), gr (smoothed), gr_res_anchor
  (gr − ref[anchor]), gr_xcorr_offset (argmin over ±band of |gr − ref| in a local
  window, in ft), pre_tvt_std, pre_tvt_slope`. Use `model._reference` for `ref`.
- [ ] **Step 3:** Self-check in `__main__`: load train well `000d7d20`, assert
  `X.shape[0] == len(hw)-ps`, `X.shape[1] == len(names)`, label length matches,
  no inf. Run it; expect a printed shape line and no assertion error.

---

### Task 2: Train + holdout validation (`train_residual.py`, `validate_residual.py`)

**Files:** Create both.

**Interfaces:**
- `train_residual.fit_model(wells) -> HistGradientBoostingRegressor` — pools
  `well_features(..., with_label=True)` over `wells`, subsamples 1-in-5 rows, fits.
- `train_residual.predict_well(model, hw, tw) -> np.ndarray[len(hw)]` — full TVT
  vector: known `TVT_input` before ps, `anchor + clip(model.predict(X))` after;
  hold-TVT fallback when `ps < 50`.
- `validate_residual` — 80/20 well split by hash of name; fit on train split,
  score holdout per-well RMSE vs hold-TVT; print mean/median/win-rate.

- [ ] **Step 1:** Implement `fit_model` and `predict_well`.
- [ ] **Step 2:** Implement `validate_residual.py` (deterministic split:
  `hash(name) % 5 == 0` → holdout).
- [ ] **Step 3:** Run `validate_residual.py`. Expected: prints model vs hold-TVT
  mean/median RMSE and win-rate. Gate: model mean ≤ hold-TVT mean.

---

### Task 3: Versioned submission (`submit.py` extended)

**Files:** Modify `submit.py`; create/append `versions.json`.

- [ ] **Step 1:** Add `--model {hold,residual}` and `--tag vN`. For `residual`,
  fit on ALL train wells, predict the 3 test wells, write `submission.csv`.
- [ ] **Step 2:** Zip to `submission_<tag>.zip`; append a record to `versions.json`
  (`tag, model, local_mean, local_median, win_rate, ts`); copy the best to
  `submission.zip`.
- [ ] **Step 3:** Verify: rows == sample rows, ids match, no blanks (reuse existing
  checks). Generate v1 (self-features).

---

### Task 4: Offset-well features → v2 (`neighbors.py`)

**Files:** Create `neighbors.py`; extend `features.py` with optional neighbour block.

**Interfaces:**
- `neighbors.nearest(well, k, allowed) -> list[str]` — k nearest training wells by
  centroid, restricted to `allowed` (the train split, to avoid holdout leakage).
- `neighbors.nbr_tvt_level(hw, nbr_wells) -> np.ndarray[n_post]` — neighbours' TVT
  interpolated at the target's lateral position.

- [ ] **Step 1:** Implement `neighbors.py` (centroid KD-lookup via numpy).
- [ ] **Step 2:** Add `nbr_tvt_level`, `nbr_dip` to `well_features` behind a flag;
  thread `allowed` through so holdout wells only see train neighbours.
- [ ] **Step 3:** Re-run `validate_residual.py` with neighbours on. Compare to v1.
- [ ] **Step 4:** If it beats v1 on holdout mean, generate v2 and repoint
  `submission.zip`; else log the result and keep v1 as best.

## Self-Review

- Spec coverage: target/safety (T2 clip+fallback), model (T2), v1 features (T1),
  v2 offset features (T4), well-level holdout (T2/T4), versioning (T3) — all mapped.
- No placeholders: feature list and interfaces are concrete.
- Type consistency: `well_features`/`predict_well`/`fit_model` signatures reused
  verbatim across tasks.
