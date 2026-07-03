# PROGRESS — paused 2026-07-02 late evening

Machine handed back to the user (they're training another model). All wellbore
compute was stopped. This file is the exact resume state. Read HANDOFF.md for
the project fundamentals; this file is the delta.

## Standing orders
- **DO NOT SUBMIT** anything until the user says so.
- **Target: LB 5–7 ≈ honest local OOF ~7–9** (LB ≈ CV × 0.5–0.8; the public
  dual-pipeline notebook is CV 9.21 → LB 7.2).
- Long jobs (>20 min): launch DETACHED (`Start-Process ... python -u ... -RedirectStandardOutput`)
  — harness background commands were killed twice mid-training.

## Scoreboard (pooled RMSE, honest 5-fold OOF unless noted)
| What | Score |
|---|---|
| hold-TVT | 15.910 |
| v4 GBM+ridge (submitted) | 15.04 → **LB 15.318** |
| TCN 3-generation ensemble + near-PS ramp (kernel v6, STAGED not submitted) | 11.699 → LB'd earlier version 11.677 |
| PF tracker alone (32-seed port, full 773 wells) | 10.956 |
| **TCN3(ramped) + PF(smoothed 101) blend, fold-honest OLS** | **9.431** (weights ≈ [.04,.26,.30] TCNs + .56 PF) |
| public dual-pipeline reference | CV 9.21 → LB 7.2 |
| LB leader | 5.262 |

PF full-OOF artifacts: `pf_preds_*.npz` (4 shards) in the session scratchpad;
blend recipe in `experiments/blend_pf.py` (tcn3_ramped+pf_smooth variant).

## Where I stopped (mid-flight, killed)
1. **PF full evaluation** — 4 detached shards (`experiments/pf.py START END`) were
   ~60% done and were killed (they only save `pf_preds_*.npz` at exit → progress
   lost, rerun costs ~40 min wall with 4 shards). Early pooled numbers: 7.1/7.1/10.0/7.2
   vs hold 12.5/17.8/15.3/16.0.
2. **TCN v8c gate** (v2 config on the 8-cut aug cache) — killed at ep 10/60, rerun
   is `SEQV=v8c SEQ_EPOCHS=60 SEQ_CH=96 SEQ_LENPROP=1 SEQ_AMP=1 SEQ_AUGCACHE=aug_cache8.pkl python -u experiments/seq.py 0` (needs caches, see below).
3. **`experiments/blend_pf.py` is written and ready** — fold-honest OLS of
   [3 TCN raws (ramped) + PF] the moment `pf_preds_*.npz` (4 shards) exist.

## Resume sequence (in order)
```powershell
# 0. caches live in the OLD session scratchpad and may be gone. Rebuild if missing (~20 min total):
#    experiments/build_cache.py  -> cache.pkl        (base features, eval side)
#    experiments/aug_cache.py    -> aug_cache.pkl    (4-cut training data)
#    experiments/aug_cache8.py   -> aug_cache8.pkl   (8-cut training data)
#    NOTE: scripts reference the scratchpad path via HERE; they write next to themselves.
#    The OOF raws (oof_seq*.npz), TCN weights (seq*.pt), norm.npz are ALSO in that scratchpad;
#    weights are safe on Kaggle dataset aydhin/wellbore-tcn-weights (v2, 15 models + norm.npz).
# 1. DONE — PF shards rerun (10.956 pooled standalone)
# 2. DONE — blend = 9.431 honest (new best)
# 3. DONE — v8c gate TIED v2 (11.82 vs 11.78): 8-cut data lever saturated; folds 1-4 NOT trained.
# 4. IN FLIGHT — TCN v4 (tracker channels). Pipeline:
#    a) pf_cuts.py shards (PF for aug cuts, 16 seeds) -> pf_cuts_*.npz   [launched]
#    b) ext_cache.py -> aug_cache_v4.pkl + cache_v4.pkl (27 ch: 15 base + pf_delta + 11 costvol)
#    c) gate: SEQV=v4 SEQ_EPOCHS=60 SEQ_CH=96 SEQ_LENPROP=1 SEQ_AMP=1
#       SEQ_AUGCACHE=aug_cache_v4.pkl EXP_CACHE=cache_v4.pkl python -u seq.py 0
#       (exp.py now honors EXP_CACHE for the eval cache)
#    d) if gate clearly beats 11.78: all folds -> re-blend with PF (blend_pf.py pattern)
# 5. Later: PF 128-seed/multi-scale for the blend member; MDN/MTP head; formation-surface
#    gated channel; savgol sweep.
# 4. THE BIG BUILD (next model generation, TCN v4 with tracker channels):
#    channels to add per well/cut: PF path delta + per-point std + loglik stats;
#    GR-vs-typewell cost volume (GR - tw_gr(anchor+o) at o = ±80,±40,±20,±10,±5,0);
#    beam-search DP paths (3 diverse configs from public BEAM_CONFIGS);
#    dip tangents sin/cos(dZ/dMD), sin/cos(dX/dY);
#    formation-surface prior (experiments/surface.py: cons/best/spread + rmse trust —
#    ALONE it's useless (81 pooled) but it's the standard drift-well anchor as a gated channel).
#    Then retrain TCN (v2 config) on these channels; target honest <= 9.
# 5. Later: MDN/MTP multi-modal head (Alyaev arXiv 2201.01871, MTP loss alpha_class=0.1,
#    L1 on best mode); noise-matched GR augmentation; savgol(61,3) output smoothing sweep.
```

## Key research facts (full details in memory + agent reports)
- The ~200-team LB cluster at 7.17–7.19 forks the public dual-pipeline notebook.
  Its recipe: GBM stack over ~250 tracker features, blended 60/40 with a
  **128-seed likelihood-weighted particle filter** (state = TVT+Z structure +
  dip rate; MOM .998 VN .002 PN .005; GR-likelihood sigma = clip(prefix GR-vs-typewell
  std, 10, 60); seeds weighted exp(loglik/scale), scales 3/5/8/12).
- PF + learned model have decorrelated error tails — the proven big drop.
- Dead ends (measured by others, don't retry): learned CNN GR-matchers, mode
  rankers (leak), DTW-as-engine, synthetic-only pretraining, AEON features.
- Hidden test wells all < 12k points. Some test wells may be train copies but
  NOT row-aligned — any lookup must validate by MD-interp prefix RMSE < 1 ft
  (our kernel v6 lookup matches on X/Y content; consider adding the MD guard).
- Hidden wells have TVT_input NaN gaps → kernel v6 carries the gap-robust
  anchor + interpolation + near-PS damping ramp (RAMP_X/RAMP_Y in kaggle_notebook.py).

## Assets
- Kernel: `aydhin/wellbore-v4-two-stage` v6 = TCN ensemble, verified save-run, NOT submitted.
- Weights dataset: `aydhin/wellbore-tcn-weights` v2 (15 .pt + norm.npz).
- Repo: https://github.com/Akim-m/wellbore (private).
- Public-notebook teardowns + web research: see the memory file and the
  session's agent reports (PF/beam/NCC/FormationPlaneKNN specifics, constants, line refs).

## v4 gate result (2026-07-03): fold-0 = 9.74 raw (vs 11.78 plain TCN) � tracker channels WORK. Folds 1-4 launching; on completion: combine oof_seq_v4_f*.npz, re-blend with PF + old TCNs (blend_pf.py pattern), expect honest ~8.5-9.

## Milestone (2026-07-03): honest OOF 9.353 = v4(ramped) + PF(smooth101) + tcn trio(ramped), weights [.20 .45 .01 .23 .28]. v4 folds: 9.74/10.85/10.29/12.38/9.34; v4 alone 10.62. Next: 128-seed multiscale PF (pf_ms shards launched) -> re-blend + v5 channels.
