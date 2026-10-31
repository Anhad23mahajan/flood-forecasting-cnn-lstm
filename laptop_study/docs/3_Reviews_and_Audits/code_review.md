# Adversarial code review (2026-09-20, ~13:20 IST)

Scope: `completion/engine.py`, `model_v2.py`, `completion/b_baselines.py`, `completion/severity_study.py`, `completion/d_aggregate.py`,
`completion/g_report.py`, `completion/run_queue.py`, against `completion/PREREGISTRATION.md` and `completion/STATE.md`.
Method: line-by-line reading plus tiny single-threaded checks (`OMP_NUM_THREADS=1`, `TF_THREADS=1`, at most 4-8 windows, memmap scans; nothing edited,
no training, no process touched). Scratch scripts live in the session scratchpad, not in the project. Raw-data facts already confirmed in
`independent_verification.md` (persistence numbers, pool contents, rain timing) were not redone.

**Bottom line: no CRITICAL finding.** I found no path by which target/future depth reaches a model input, no train/held-out tile or (in blocked mode) start-time
overlap, no unmasked padded pixel in any metric, and no arithmetic error in skill / per-lead skill / RMSE / CSI or in the bootstrap statistic. The headline
point estimates (never-seen skill of `local`/`clstm` in the causal/hybrid regimes) are not invalidated by anything below. There ARE five MAJOR findings, two of
them (F1, F2) about the reliability of the numbers that will appear in the final tables/intervals, and F3 about a verdict rule that differs from the pre-registration.

---

## Findings

### F1. MAJOR - aggregation group key ignores batch, lr and seed: batches OPT and F will silently merge into (or knock out) the headline cells
* `completion/d_aggregate.py:33` (`key = (exp, arch, regime, variant, augment, blocked, epochs, n_train)`), consumed by `completion/g_report.py:35-46`.
* What is wrong: the queue (`run_queue.py:60-61` OPT = `local`/`clstm`, hybrid, full, batch 8, lr 3e-3; `run_queue.py:64-67` F = `local`, hybrid, full, seeds 11-14) produces run JSONs whose
  `arch/regime/variant/augment/blocked/epochs/n_train` are IDENTICAL to the frozen-recipe runs (`epochs=30`, `n_train=300`). `batch`, `lr` and `seed` are stored in the JSON
  (`engine.py:468`) but are not part of the key. `chain_after_A.sh` does start OPT and F.
* Consequence when they finish: the `local`/`clstm` hybrid/full group of every experiment holds 6 runs (OPT; `local` 18 with F as well) instead of 3.
  `d_aggregate.main` prints "(incomplete)" because `len(runs) == K` fails (visible), but `g_report.stat` only tests `len(runs) < 3` (`g_report.py:42`), so it
  computes `test_mean`, `held_pooled` and both bootstraps over the mixed set (frozen-recipe folds + batch-8/lr-3e-3 runs + extra seeds) without any warning:
  the headline table, H1, H2 and H5 silently average over the mixed set, while H3, H4, H6 (`len(a) == 3`, `g_report.py:121, 132, 153`) and the test-time ablation table
  (`len(runs) == 3`, `g_report.py:170`) silently DROP the hybrid cells. `tile_bootstrap` additionally counts each tile 2-5 times.
* Evidence: today `results/runs` has 40 files and none of the `_b8`, `_lr`, `_s11..14` variants, so the bug has not fired yet (checked with `ls | grep`).
* Smallest fix: put `int(r.get("batch", 1))`, `float(r.get("lr", 1e-3))` and `r["seed"] == 1000 + 10 * r["fold"] + r["K"]` (default-seed flag) in the key (or skip non-default
  runs in `load_runs` and give OPT/F their own loader); make `g_report.stat` return None unless `len(runs) == runs[0]["K"]` and the folds are `{0..K-1}` exactly once.

### F2. MAJOR - the start-time bootstrap treats strongly overlapping windows as independent clusters: never-seen intervals are too narrow
* `completion/d_aggregate.py:70-90` (`start_bootstrap`), used for every "test [95% CI]" in `summary.md`/`REPORT.md` and for H1 (`g_report.py:91`).
* What is wrong: the statistic and the pairing across folds are right, but the resampling unit (one start time) is not independent. Measured start spacing of the
  60 test start times versus the 54-depth-index span of one window (5 input + 5 forecast frames at stride 6):

  | test region | depth files | start spacing (idx) | overlap of neighbouring windows |
  |---|---|---|---|
  | Mozambique 480 m | 1729 | 28 | ~48 % of the span with the nearest neighbour (26/54 idx), 1 neighbour each side |
  | UK 60 m / UK 30 m | 865 | 14 | ~74 % with the nearest neighbour (40/54 idx), 3 neighbours each side (overlaps 40, 26, 12 idx) |
  | (Australia training pool, 80 starts) | 2881 | 36 | 18/54 idx, 1 neighbour each side |

  Effective number of non-overlapping windows is about 1729/55 = 31 (Mozambique) and 865/55 = 16 (UK), not 60, and flood errors are autocorrelated far beyond one window
  (a single event per region). The same applies to the tile bootstrap (`tile_bootstrap`, line 52): tiles are correct clusters for within-tile correlation but all tiles share the same storm timeline.
* Effect: intervals are anti-conservative. The H1 verdict "lower bound > 0" will not change (skill ~0.8), but reported widths (and any claim of "significant" differences between regimes/architectures) are overstated.
* Smallest fix: moving-block bootstrap over the time-ordered start times with block length >= ceil(54 / spacing) + 1 (Mozambique 3, UK 5, Australia 2), or resample contiguous time thirds; state in the text that each never-seen region is ONE event and E60/E30 (and the 3 folds) share the same UK event and Australia training event, so they are not independent replications.

### F3. MAJOR - `g_report.py` implements different verdict rules from the ones pre-registered for H3 and H4
* `completion/g_report.py:116-126` (H3), `:127-138` (H4) versus `PREREGISTRATION.md:35-36`.
* H4 pre-registered: EQUIVALENT if |difference| < 0.02 with an interval inside +-0.03, otherwise the larger wins. Code: "EQUIVALENT if the largest absolute per-FOLD difference < 0.03" (`mx < 0.03`, line 136); no interval on the difference, no 0.02 test on the mean, no "larger wins".
* H3 pre-registered: skill without depth history lower "by more than the fold-to-fold sd". Code: prints the mean difference and "all folds positive" (line 126); the sd comparison is never computed.
* The pre-registration also says test-time ablations are run on every run, but batch F uses `--no-ablate` (`run_queue.py:66`); harmless (F is a seed repeat) but should be listed as a deviation.
* Effect: a verdict reported as "pre-registered" is not the pre-registered one; the recurrence conclusion (clstm vs local) is exactly the one this rule decides.
* Smallest fix: compute a paired start-time (block) bootstrap of the mean fold difference clstm - local and apply the frozen rule literally, or write "deviation: looser rule (max per-fold |diff| < 0.03)" next to the H3/H4 lines.

### F4. MAJOR (interpretation) - the "forced" layout handicaps the recurrent model, so H5 fails for `clstm` for reasons unrelated to rain physics
* `completion/engine.py:186, 212-213` (5 extra steps appended AFTER the observed history, depth channel = repeated last frame), `model_v2.py:184-199` (`clstm` reads only the final ConvLSTM state).
* Evidence (finished runs, never-seen skill, mean of 3 folds): E480 `clstm` causal 0.810, hybrid 0.811, **forced 0.765** (folds 0.763 / 0.780 / 0.752), whereas E480 `local` is 0.794 / 0.807 / **0.834**.
  The forced input is a superset of the hybrid information yet `clstm` loses 0.046. `local` is immune because `LinFeatures` (model_v2.py:62-69) reads step `last_obs=4` directly; the ConvLSTM has to carry the last one-step differences through 5 further steps whose depth-difference channel is identically 0.
* This is not a leak and not a bug, but H5 ("causal <= hybrid <= forced") and any "regime effect" statement for `clstm` are confounded by the input layout. (Cause is my hypothesis, UNVERIFIED; E60/E30 clstm forced runs are not finished.)
* Smallest fix: report it as observed (H5 fails for clstm/E480) and do not attribute it to rainfall; if time allows, feed future rain as extra channels at the last observed step (or a separate 1x1 branch) for `clstm`, and re-run one fold.

### F5. MAJOR (interpretation) - the Grad-CAM and its deletion test explain/measure different quantities, and neither involves the severity head
* `completion/severity_study.py:259-271` (`cam_of`: target = signed sum of `frames[0, 0]`, i.e. lead 1 only, taken through the DEPTH decoder from the encoder hidden map), `:288-305` (deletion effect = `|delta_of(xm) - d0|` where `delta_of` sums the predicted change over ALL 5 leads and the whole patch).
* Problems: (a) heat map for lead 1, effect measured on the 5-lead patch total; (b) the effect is the absolute value of a patch-SUMMED change, so opposite-sign changes at different pixels cancel; (c) the top set is `np.argsort(cam.ravel())[-k:]` (line 289): the CAM is ReLU-ed, so with many exact zeros the "top 10 %" contains arbitrary tied zero-heat pixels (UNVERIFIED how sparse the CAMs are; no `_cams.npz` exists yet); (d) top-heat pixels form contiguous fronts while the random control is scattered, which affects a 5x5-kernel model differently even if the map were meaningless (the wet/dry matching only fixes the wet fraction).
  (e) The severity class (capsule vs dense head) is never used, so "Grad-CAM of the capsule network" is NOT what is tested (the original `grad_cam.py` had the same lead-1 depth target, so this is inherited).
* Smallest fix: use the same lead (or the 5-lead sum) for both CAM target and deletion effect, use the per-pixel L1 change instead of the summed change, break ties by adding tiny noise or use `cam > 0` only, and word the write-up as "explains the depth forecast", not the severity class. If a capsule explanation is wanted, backpropagate the class-capsule length (or the dense logit) to the same hidden map.

### F6. MINOR - `rain_shifted_cells` ablation is a no-op for a large share of test windows (clipping)
* `completion/engine.py:204-206` (`rr = clip(r + shift, 0, H - size)`) and `:405-406`.
* Evidence (measured on the cached pools): Mozambique (+1,+1): no shift at all in rows for 120/238 windows and in columns for 119/238; UK 60 m (+2,+2): rows never shifted (120/120) because the 85-row domain equals the window, columns not shifted for 60/120; UK 30 m: 120/240 in rows and 120/240 in columns. On the Australia training tiles it works (0/1271 no-ops).
* Effect: `rain_shifted_cells` = `none` to 3 decimals everywhere (e.g. E60 local 0.841 vs 0.841) is partly an artifact. `STATE.md:62` (b) cites this ablation for "skill is insensitive to crop offsets"; that sentence is not supported by it. The valid bound is `rain_removed` (costs only 0.008-0.02), which limits how much any rain misplacement can matter.
* Smallest fix: shift in the opposite direction when `r + shift` would leave the grid (or read from a shifted edge-padded copy); until then cite `rain_removed`, not `rain_shifted_cells`.

### F7. MINOR - "forced" carries one interval more than the docstring says (post-horizon rain at phase 0)
* `completion/engine.py:186` (`j + [j[-1] + m for m in 1..5]`), docstring lines 13-14.
* At phase 0 (start % 6 == 0) the last observed frame is at the start of file `j4`; the forecast horizon covers files `j4..j4+4`; forced supplies `j4` (via the hybrid last step) plus `j4+1..j4+5`, and `j4+5` begins at the instant of the last forecast frame (printed: start 0 -> forced files [0..9], horizon needs 4..8). For phase p > 0 the extra file overlaps the last 300 p s of the horizon, so the set is still the minimal complete cover. Physically this cannot leak the target (rain after t cannot change depth at t) and no depth information is involved; it only means the docstring "the rainfall of the 5 forecast intervals" is really "intervals 2..6". Impact on skill: none expected.
* Smallest fix: correct the docstring (or use `m in range(0, 5)` after the hybrid step if exact intervals are wanted).

### F8. MINOR - severity rule layer: two inputs are not strictly forecast-time, and the recalibration is fitted on truth but applied to predictions
* `completion/severity_study.py:83` (`rain = x[T_IN - 1, ..., 2].mean()`): in the hybrid regime that rain file is the interval CONTAINING t, i.e. up to 30 min of future rain (the project's own finding); there is no causal severity regime. Only `n_triggered`/reason text depend on rain (in `symbolic_rule_layer.py:55-63` the rain and slope rules only append a reason; only rules 1 and 2 change the class, and those use the model-predicted rate and the static drainage distance), so class accuracy after rules is unaffected.
* `:222-234`: recalibrated thresholds are the midpoint of moderate/severe class means of the TRUE future rate on the fold's balanced training windows, but at test time the rule sees the MODEL-predicted rate (shrunk toward 0 by an imperfect forecast) on natural-distribution windows: calibration and application distributions differ, which biases the "recalibrated" row against the rules (or for them, depending on shrinkage). Units also differ from the constants' comments (`RAINFALL_CORROBORATE_THRESHOLD` says mm; the feature is a mm/h intensity).
* `:251-255`: "reason_truth" measures sign agreement of predicted vs true patch-mean trend (asymmetric cut-offs: rising `> 0`, receding `< -1e-4`); it does not evaluate the reasons the rules actually print. Fine as a proxy, but call it that.
* Fix: use predicted rates (held-out or cross-fitted) for calibration; add `causal` to the severity regimes or footnote that rain is partly future.

### F9. MINOR - blocked hold-out has no embargo between the held-out time block and adjacent training windows
* `completion/engine.py:436-440`.
* Start times are disjoint between train and held-out windows (verified, list below), but window spans are 54 idx and the Australia start spacing is 35-36 idx, so the last held-out start and the first training start overlap by 18-19 idx (up to 4 identical frames when the spacing is exactly 36; other tiles, same times). Pakistan spacing 50-51 (overlap 3-4 idx, no shared frame). Leakage is weak (different tiles) but H6 is meant to be the strict test.
* Fix: drop training windows whose `[start, start + 54]` intersects any held-out window's span.

### F10. MINOR - design-phase runs on the never-seen UK exist, and E60 fold-0 held-out windows were reused
* `PREREGISTRATION.md:16-17` says the never-seen regions were not used for design and "the four heavy runs that did evaluate them were not consulted".
* `results/runs_design/` holds four non-blind runs with `test` present (UK skill v2 0.758, v2+aug 0.798, v2lin 0.824, v2lin+aug 0.828); `run()` prints the test line to the log (`engine.py:477-478`), so they were visible. The two architectures actually chosen (`clstm`, `local`) have only `_blind` design runs (held-out 0.711 / 0.704, no test key), which supports the claim, and the UK ranking of the heavy runs essentially matches their held-out ranking (v2 < v2+aug < v2lin ~ v2lin+aug; the top two swap by 0.001-0.004), so no decision would plausibly have changed. Not verifiable from code; state it as a disclosure rather than a fact.
* The E60 fold-0 held-out windows in the final runs are the same windows as in the design runs (same seed 1003, `n_val=300`), so the E60 fold-0 held-out number is not independent of design selection (the never-seen number is).

### F11. MINOR - UK 60 m padding shifts the patch-relative DEM channel by ~8 m
* `completion/engine.py:222-230` (edge padding of DEM to 128x128) + `model_v2.py:39` (`dem - mean(dem)` over the padded window).
* Measured: mean DEM of the real 85x85 window 22.31 m vs 30.30 m over the padded window: shift 7.995 m = 0.160 network units, a covariate shift for E60 test windows (and the S1-S3/S4 baselines, which use the same padded mean). `terrain_removed` changes skill by 0.000 (E60 local 0.841 -> 0.841, clstm 0.866 -> 0.865), so the practical impact is nil, but the "padded windows are handled" statement should mention it.
* Fix: compute the mean over `mask` only (or pad the DEM with the real-window mean).

### F12. MINOR - provenance and file-name hygiene
* `engine.py:472`: `code_md5` is taken when the run FINISHES, not when it starts. Runs that started before the 10:30-10:31 edits (kernel/batch) but finished after carry the new hash (4 runs have the current md5 and no `batch` key; 12 runs carry the older `engine fb4ce0 / model_v2 a43ab5`). The `engine.py` edits were default-preserving for batch 1 / kernel 5 (diff vs `engine_before_batch.py.bak` reviewed: only `local1/local11`, `batch`, `lr`); `model_v2.py` has no backup, so its 10:30 change (presumably the `build_local(kernel=...)` argument, param counts 4101 unchanged) could not be diffed. Results are very probably unaffected, but the stamp is not proof. Hash at start.
* `engine.py:457` and `run_queue.py:23-26`: `n_train` is not in the run id, so a run with a different `--n-train` collides with an existing file and silently returns the old result (`engine.py:460-461`). `d_aggregate.py:31` then silently drops any run whose recorded `n_train` (= `min(n_train, pool)`) is not 300. Currently all folds have >= 791 candidate windows, so nothing is lost today.
* `d_aggregate.py:21`: one global `RNG` stream, so an interval for a group changes when other groups are added or the call order changes (`REPORT.md` and `summary.md` intervals for the same group can differ). Seed per group.
* `g_report.py:42`: `v2lin` (batch D, one fold per experiment) never appears in `REPORT.md` because `stat` needs 3 runs; it appears in `summary.md` as "1/3". Mention this in the write-up.

### F13. MINOR - diagnostics/pool details
* `engine.py:295, 329, 470`: `val_small = va_idx[:48]`, the 12 windows of the dead-network guard and `train_subset = tr_idx[:100]` are the FIRST indices of a sorted list, i.e. the earliest start times; per-epoch monitoring and the reported `train_subset` skill are time-biased (no selection uses them; results are not affected).
* `engine.py:147`: window inclusion depends on the LAST FORECAST frame being wet (target). Measured effect on the test pools is negligible: Mozambique 4/238 windows enter only through the target (0.0 % of persistence SSE), UK 60 m and UK 30 m 0/120 and 0/240; no wet-at-last-observed/dry-at-last-forecast windows exist. Disclose; do not fix.
* `engine.py:210`: Pakistan (training region) forced at its very last start (3978): needs file 672, only 0-671 exist, so `min(j, nf-1)` repeats file 671 for the post-horizon step. Trivial. All other regions: max needed forced index = nf - 1 (no clamping).
* R60/R30 (unqueued) cannot run as configured: UK 60 m (85x137) has no 128x128 tile; UK 30 m has at most 2 tiles for K = 3. `severity_study.sev_labels` cache (line 37-45) is not keyed to the pool contents and is written non-atomically (first 4 parallel jobs would race); it exists now.
* `ablation_skills` `depth_hist_scrambled` (line 386-387) is a deterministic REVERSAL of frames 0-3; because the model derives differences internally it changes `diff1` to (d4 - d0) and leaves `trend` = d4 - d3, so it is a "history disordered" test, not a random scramble. Fine, name it accurately.
* Stale docstring: `engine.py:1-3` still says the engine trains "the persistence-anchored CNN-LSTM (same architecture ... as depth_experiment.py)"; it now trains v1/local/local1/local11/clstm/v2/v2lin.

### F14. MINOR (scope) - overlapping test tiles and pseudo-replication
* Mozambique (4 corner tiles of a 151x138 domain, ~76 % overlap) and UK (2 or 4 corner tiles) test windows re-use the same pixels/time, so pooled test skill weights the overlap region up to 4x. Already disclosed in `STATE.md`; it is a weighting, not an error. Also: each never-seen region is a single flood event; E60 and E30 share the same Australia event and UK event, and the three folds share the same test windows, so "3 experiments x 3 folds" is really 2 independent train->test pairs.

---

## Pre-registration vs code (item 8)

| PREREGISTRATION.md | code | status |
|---|---|---|
| Adam, peak lr 1e-3, 200-step warm-up + cosine, clip 1.0, loss MSE / persistence MSE, batch 1, 300 windows, 30 epochs, no early stopping | `engine.py:261-334`, `run_queue.py:20` (`epochs=30, n_train=300`) | matches |
| seeds `1000 + 10 fold + 3` | `engine.py:423-457` (`seed = 1000 + 10*fold + K`, K = 3) | matches |
| K = 3 tile folds fixed by seed 0, each tile held out once, <= 300 held-out windows, every never-seen window | `folds_of_tiles(..., seed=0)`, `split_for` | matches (verified) |
| CSI at 1 mm and 0.1 m, RMSE, per-lead skill | `evaluate`/`summarize` | matches |
| params: local 4,101, clstm 4,741 | built and counted: 4101 and 4741 | matches |
| H1 both 95 % intervals above 0 | `g_report.py:82-95` | matches (intervals see F2) |
| H2 beat best linear S3 | uses S3c (clipped), point comparison | matches in spirit |
| H3 lower by more than fold-to-fold sd | prints difference only | **differs (F3)** |
| H4 |diff| < 0.02, interval inside +-0.03, else larger wins | max per-fold |diff| < 0.03 | **differs (F3)** |
| H5 causal <= hybrid <= forced (means) | `g_report.py:139-147` | matches; fails for clstm/E480 (F4) |
| H6 within 0.05 | `g_report.py:148-158` | matches |
| test-time ablations on every run | F batch is `--no-ablate` | minor deviation (F3) |
| deviations after results (S4/S5/clipped rungs, K, OPT) | labelled exploratory in STATE | OPT/F additionally corrupt aggregation (F1) |

## Checked and found CORRECT (with evidence)

1. **rain_intervals index arithmetic** (`engine.py:177-187`): history frame s of a window at depth index `start` is at t = 300 (start + 6 s), file `floor(t/1800) = start//6 + s` (identical to the loader's `depth_second // 1800`); causal is that minus 1, `j < 0` gives zero rain. Printed for starts 0, 7, 13: causal [-1..3] / [0..4] / [1..5], hybrid [0..4] / [1..5] / [2..6], forced 10 steps. `selftest` asserts compose == loader and causal/forced against raw files. Causal never contains rain that had not completed by t (at phase p it lags by 300 p s, i.e. it is conservative, not leaky).
2. **End-clamping**: rain files vs depth files are Australia 481/2881, Pakistan 672/4033, Mozambique 289/1729, UK 145/865; the largest needed forced index equals `nf - 1` everywhere except Pakistan's last start (F13). Rain step = 1800 s in all regions; depth step = 300 s (asserted in `build_pool`).
3. **Forced regime has no target information in the inputs**: steps 5-9 of the input are bit-identical repeats of step 4 (`np.array_equal` for a UK window, all 5 steps); the anchor `x[:, -1, ..., 0]` is step 4 (`decode_horizon*`); `LinFeatures` with `last_obs = 4`, T = 10: `diff1 == (d4 - d3)/0.01` (max error 0.0), `trend == (d4 - d0)/0.01` (max error 9.5e-7), `d_last == d4/5` (0.0); the first difference is 0 by construction (`prev` = first frame) so the 5 constant steps add nothing.
4. **Splits**: for E480/E60/E30, folds 0-2, standard and blocked: train/held-out tile overlap 0, index overlap 0, no duplicated indices (`choice(..., replace=False)`), blocked start-time overlap 0 (train starts 53-54, held-out 26-27), `n_train = 300` in all 18 splits (candidate pools 791-3304), test pools = every wet window of the never-seen region. No use of the never-seen region in `build_and_train`, `val_skill`, `split_for` or the baselines' fitting (`fit*` see only `tr_idx`); `--blind` skips it.
5. **Padded windows** (UK 60 m, 85 of 128): `compose` zeroes padded depth and target, `mask.sum() = 7225 = 85^2`; `evaluate` masks both SSE (model and persistence) and all CSI counts. Test: a +0.1 m shift predictor gives SSE 1444.7873 from `evaluate` vs 1444.7874 by hand with the mask (3276.59 if padding were counted); persistence gives skill exactly 0.0.
6. **Metrics**: skill = 1 - sum SSE_m / sum SSE_p; per-lead skill uses per-lead sums; RMSE denominator = sum(npx) x 5; CSI = tp/(tp+fp+fn) with tp/fp/fn from `(arr>thr) & mask`, thresholds 0.001 m and 0.1 m; persistence `q` is taken from the untransformed `x` so test-time transforms cannot touch the baseline.
7. **Alignment for the bootstrap**: `_ser` uses `pool["meta"][idx]` in the same order as `evaluate` iterates `idx`; `tile_bootstrap` sums per-window SSE by (tile_r, tile_c) and is a valid tile-cluster ratio-of-sums; `start_bootstrap` resamples start times with replacement (duplicates counted), uses the same resample for all folds, and its statistic (mean over folds of per-fold pooled skill) equals the reported point estimate.
8. **Training loop**: `WarmupCosine` total steps `epochs * (n // batch)` equals the number of `step` calls (`range(0, n - batch + 1, batch)`); the loss is averaged over batch; augmentation applies the same `rot90(axes=(1,2))`/flip to x and y (verified numerically: x-depth == y under all 8 transforms); the mask is not used in training but training tiles are never padded; dead-network guard triggers on the change map spread, not on skill; seeds are set for TF/NumPy and shared across variants (paired comparisons).
9. **Ablations that do what they claim**: `depth_hist_frozen` (first 4 depth frames replaced by frame 4 before the model, which computes differences internally, so all history-derived differences become 0: skill -0.008..+0.02, i.e. ~0); `rain_removed`, `terrain_removed`, `rain_hist_frozen`, `future_rain_removed` (steps >= 5 only, forced only) act on the right channels/steps; `rain_shifted_earlier` (`regime_override="causal"`, hybrid only) does apply the one-interval-earlier mapping and keeps T = 5. (Exception: `rain_shifted_cells`, F6.)
10. **Baselines** (`b_baselines.py`): features use only depth history up to the last observed frame, DEM, rain of the regime; targets aligned with features (same `sel`, same row-major reshape); the lookup table (S5) and MLP (S4) use history only and the same `tr_idx`/`va_idx`/`te_idx` (same seed) as the neural runs; clipped variants only clip at 0.
11. **Model code** (`model_v2.py`): `InputNormalizerV2` differences `depth - prev` with `prev[0] = depth[0]`; `LinFeatures`/`build_local`/`build_clstm` zero-initialised output added to the last observed depth (untrained = persistence, `selftest`); parameter counts 4101 / 4741 as pre-registered.
12. **Pool integrity**: every cached depth pool has 0 non-finite values and no NoData sentinel (min 0 or 1.3e-5, max 4.8-26.7 m); DEM/rain caches finite and non-negative rain. `severity_study.py` fold construction is identical to the engine's (`folds_of_tiles(tiles, K, seed=0)` on the same pool), so held-out tiles are disjoint from the balanced training windows, and the fixed Grad-CAM window set (`default_rng(777)` over the fold's held-out windows) is independent of the training seed (same fold gives the same 40 windows). `depth_to_severity` thresholds 0.3/1/2 m match the docstring; the class-changing subset logic (label != last-frame class) and B1 (threshold of the predicted maximum) are implemented as documented; the rule constants are read as module globals at call time, so the monkey-patched recalibration does take effect.

## Could not verify / open
* Whether the CAMs have many exact zeros (no `severity/*_cams.npz` yet) - relevant to F5(c).
* Crop offsets of DEM/rain vs depth (already flagged inconclusive by the independent verification; F6 shows the shift ablation cannot settle it).
* Cause of the clstm forced-regime deficit (F4) - a hypothesis only.
* Magnitude of the effect of padding (F11) and of the earliest-window diagnostics (F13) on any reported number: expected nil, not measured beyond what is quoted.
* I did not re-run `selftest` (needs TF + all pools; not a tiny job) and did not review `model.py`, `a_rain_lag.py`, `c_wse_probe.py`, `e_georef_coarse.py`, `f_paper_protocol.py`.
