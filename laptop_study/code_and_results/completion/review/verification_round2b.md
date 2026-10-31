# Independent verification, round 2b (new result files)

Date: 2026-09-21. Verifier: independent agent, instructed not to trust the project's own code.
Method: my own scripts (numpy, rasterio, scipy; scripts are in the session scratchpad, NOT in this repository, because the task was read-only).
I READ `engine.py`, `reverse_transfer.py`, `b_baselines.py`, `windows.py`, `real_data_loader.py`, `d_aggregate.py`, `g_report.py` to learn definitions and file layout;
I did NOT import any of them (no project code executed, no TensorFlow, no training). One python process at a time; cached pools were memory-mapped and read in chunks
(largest in-RAM object: 10 raw 1073x1073 float32 frames, ~46 MB). Peak RSS was not instrumented. Nothing under `C:\Dev\flood-forecast-demo` was modified except this file.

Window arithmetic found (window_frame_indices(s, 5, 5, 6) in real_data_loader.py, confirmed by my bit-exact reads): depth file index i is the file `300*i .tif`;
observed frames are indices s, s+6, s+12, s+18, s+24; future frames s+30, ..., s+54. A window therefore spans 54 depth steps (frames s .. s+54);
two windows share no frame iff |s1 - s2| >= 55.
Note on a file name in the request: `E480_local_hybrid_blk_f0of3_s1003_ep30.json` does not exist; the real id is `E480_local_hybrid_full_blk_f0of3_s1003_ep30.json` (used for check 4).

## Summary table

| # | Check | Result | Key numbers |
|---|---|---|---|
| 1 | Skill in every run JSON = 1 - sum(sse_m)/sum(sse_p) | PASS | 284 files, 568 summaries (284 test + 284 heldout); max abs diff 0.0 (tolerance 1e-6); also lead_skill (5 leads) and rmse_model/persistence recomputed: 0 mismatches; all sse finite and >= 0; n_windows = array length |
| 2 | Cache vs raw TIFFs and persistence SSE, new reverse pools | PASS | 15 random windows per pool (6 pools): dh and df bit-exact in 90/90, persistence SSE equal (float32 sums: rel diff 0.0; float64 sums: <= 1.5e-7). Extended to EVERY window: 6301 windows in the 6 test-role pools bit-exact, 0 mismatches, 0 non-finite pixels, every stored sse_p of the matching run equal (rel diff 0.0). Also all 7605 windows of the 3 train-role pools (Pakistan 480, Australia 60, Australia 30) bit-exact. Also stored sse_p of ALL 284 run files (568 arrays, 191,414 windows) equals the value recomputed from the cache: worst rel diff 0.0 |
| 3 | Independent S5 lookup table, E60 hybrid fold 0 | PASS | held-out +0.673213 vs stored +0.673213 (diff -2.5e-9); never-seen (UK 60) +0.783144 vs stored +0.783144 (diff +4.2e-10). Also reproduced (diffs <= 6e-9, both parts): E60 f1, E60 f2, E30 f0, E480 f0 |
| 4 | No train / held-out leakage (strict blocked runs and reverse runs) | PASS (with the caveats in "Noteworthy") | E480 blk f0: min start gap 100, 0 shared frames, 0 shared tiles. R480 f0: min start gap 56, 0 shared frames. All 56 blocked + reverse run files: min gap 55, 0 shared frames, 0 overlapping spans |
| 5 | Headline: E60 local hybrid never-seen, 3 folds | PASS | folds +0.837932 / +0.842887 / +0.841916; mean +0.840912 -> +0.841 (REPORT +0.841); held-out pooled +0.755683 -> +0.756 (REPORT +0.756); block-bootstrap CI (mine) [+0.796..+0.798, +0.889..+0.891] vs REPORT [+0.798, +0.892]. All 9 reverse-transfer local-CNN rows (3 experiments x 3 regimes, never-seen and time-held-out) also equal the REPORT to 3 decimals |
| 6 | Internal sanity of new-type runs | PASS (2 housekeeping notes) | 5 random runs inspected (below) and ALL 284 run names parsed and compared with fields: id, exp, arch, regime, variant, fold, K, seed, epochs, n_train (learning curves), blocked, augment, batch, lr, guard, len(history) == epochs: 0 mismatches |

## Check 1 detail
Recomputed with numpy from `heldout_windows` / `test_windows` for every file in `completion/results/runs/*.json` (284 files, none "blind"). Skill, all 5 lead skills, and both RMSEs match the stored summaries.
This proves arithmetic self-consistency only; it says nothing about whether sse_m came from a correct forward pass (see "Could not verify").

## Check 2 detail
Pools (test role): Australia 60 (953 windows, 16 tiles), Australia 30 (3670, 62 tiles), Pakistan 480 (1080, 18), Mozambique 480 (238, 4), UK 60 (120, 2 padded 85x85 windows), UK 30 (240, 4).
For each: my reader = rasterio `read(1, window=Window(col,row,size,size))` of files `300*(s+6k).tif` (no reuse of project code); tile size taken from meta.npz `size` (UK 60 = 85).
The run's `test_windows` list is in the same order as the pool meta (tile_r, tile_c, start equal element-wise) for all six pools.
Pool completeness (extra, own code): for every pool the 60 start times equal `unique(linspace(0, n_files-1-54, 60).astype(int))`; every (tile, start) missing from the pool is dry (max of last observed and last future frame < 0.05 m) and every included window is wet (>= 0.05 m): 0 violations in all six pools.
Depth value range over frames read: [0, 26.7 m], no NaN/inf.

## Check 3 detail
S5 re-implemented from b_baselines.py's description: 12 depth edges x 17 last-change edges (np.searchsorted instead of np.digitize), 3000 random pixels per training window (rng seed 0), mean future change per (bin, lead), cells with fewer than 5 pixels replaced by the depth-bin marginal, prediction = last frame + table entry, skill = 1 - SSE_model/SSE_persistence.
My own split (16 Australia 60 tiles, `default_rng(0)` permutation into 3 folds, seed 1003, n_train 300, n_val 300) reproduced the stored held-out window list of `E60_local_hybrid_full_f0of3_s1003_ep30.json` exactly (order, tile, start) and its `held_out_tiles`. The agreement to ~1e-9 (float-summation level) means the binning, the marginal fill, the sampling and the scoring are all reproduced, not just approximated.
(The script's print label says "UK 60" for the E30/E480 extra cells; the test region used there was UK 30 / Mozambique 480 as appropriate.)

## Check 4 detail (split logic tested on the real pools)
For every blocked (`blocked=true`) or reverse (`R*`) file I recomputed the train / held-out index sets with my own code (tile folds `default_rng(0).permutation`, time blocks `np.array_split` of the sorted unique start times, embargo `b_lo-54 <= s <= b_hi+54`, sampling `default_rng(seed)`), then CHECKED that my reconstruction is the real one:
(i) my held-out window list equals the stored `heldout_windows` (tile, start, order) in 56/56 files;
(ii) the mean persistence MSE of my training sample equals the stored `persistence_mse_train` (max rel diff 1.0e-7) and the persistence RMSE of my first 100 training windows equals `train_subset.rmse_persistence_mm` (max rel diff 1.1e-8).
Power test: re-running the E480 blk f0 reconstruction WITHOUT embargo gives a different training pool (636 vs 624 windows) and a 0.8% mismatch in persistence_mse_train (R480 f0: 4 more windows, 3.7% mismatch), so the equality above shows the embargo really was in force in the stored runs.

| run | fold time block (start idx) | train start range | pools (train/held-out windows) | min gap train-vs-held-out (start units) | shared depth frames | window pairs with |ds|<=54 | tile overlap |
|---|---|---|---|---|---|---|---|
| E480_local_hybrid_full_blk_f0of3_s1003_ep30 | 0..1309 | 1409..3978 | 624 / 162 (300 sampled / 162) | 100 | 0 | 0 | 12 train tiles vs 6 held-out tiles: 0 overlapping, 0 shared |
| R480_local_hybrid_full_f0of3_s1003_ep30 | 0..539 | 595..1674 | 156 / 78 (156 / 78) | 56 | 0 | 0 | by design NONE held out: same 4 (mutually overlapping) Mozambique tiles in train and held-out |

Across all 56 files: global min gap 55 (R60, R30: pool start spacing ~14, first allowed training start is b_hi+55); E30/E60 blk 71; E480 blk 100-101; R480 56-57. 0 shared frames and 0 span overlaps everywhere. The embargo removed 4-82 windows per file.
Strict (E*) runs: train and held-out tiles are disjoint, non-overlapping 128-px grid cells (0 overlapping rectangle pairs in all 20 E blk files); the never-seen region is a different region altogether (trivially disjoint).
Per-epoch validation uses the first 48 held-out windows but nothing is selected on it (the final-epoch model is used; the guard can only abort), so the held-out set is not used for model selection.

## Check 5 detail
Files: `E60_local_hybrid_full_f{0,1,2}of3_s{1003,1013,1023}_ep30.json` (seed = 1000+10*fold+3, epochs 30, n_train 300, K 3).
Per fold never-seen: +0.837932, +0.842887, +0.841916 (REPORT per-fold table: +0.838/+0.843/+0.842). Mean +0.840912. Held-out per fold +0.726948/+0.784159/+0.753386 (REPORT +0.727/+0.784/+0.753); pooled +0.755683.
The three folds are three models scored on the SAME 120 UK windows, so pooled == mean. Block bootstrap (my implementation of the documented procedure: blocks of ceil(54/14)+1 = 5 start times, 12 blocks, 4000 resamples, 3 seeds) gives [+0.796..+0.798, +0.889..+0.891]; the REPORT's [+0.798, +0.892] is reproduced to within Monte-Carlo noise.
Fragility of that headline: the 5 heaviest start times hold 31% of the persistence SSE; dropping any one of the 12 time blocks moves the mean skill within +0.832..+0.870.
Reverse-transfer section of REPORT.md (local CNN row; never-seen mean over folds / time-held-out pooled): recomputed R480 -0.869/-0.884/-0.583 and +0.816/+0.825/+0.847, R60 +0.702/+0.738/+0.772 and +0.799/+0.808/+0.834, R30 +0.637/+0.662/+0.718 and +0.820/+0.821/+0.854: all 18 numbers equal the REPORT.

## Check 6 detail
Random 5 (seed 20260921) from ep90 / noguard / b8_lr0.003 / seed s11-14 / learning-curve groups (all fields agree with the file name):
- `E30_v1_hybrid_full_noguard_f0of3_s1003_ep30`: guard=False present, seed 1003, fold 0/K 3, epochs 30, history 30, n_train 300 (name says noguard: OK).
- `E30_local_hybrid_full_f0of3_s11_ep30`: seed 11, epochs 30, n_train 300, guard True.
- `E30_local_hybrid_full_f2of3_s13_ep30`: seed 13, fold 2/K 3, epochs 30, n_train 300.
- `E30_local_hybrid_full_f0of3_s4103_ep30`: n_train 50 (learning curve), fold 0, epochs 30.
- `E60_local_hybrid_full_f2of3_s4123_ep30`: n_train 50, fold 2.
Whole-directory pass (284 files) found 0 disagreements. Learning-curve runs carry no `_n25_` in the file name; they are identified by seed (s4003/s4103/s4203/s4303 for fold 0 and analogously for folds 1, 2 = n_train 25/50/100/600) and by the JSON `n_train`; the seed-to-n_train map is one-to-one over all 36 files. All 7 noguard files have `guard=False`; 15 ep90 files, 13 b8_lr0.003 files, 40 seed-repeat files parse correctly.

## Noteworthy / suspicious (none changes an arithmetic result above)
1. **The R480 negative never-seen skill is a start-index-0 artifact.** In every one of the 3 folds the 18 Pakistan windows with start = 0 (of 1080) carry 91-95% of the model SSE (but 24% of persistence SSE). Excluding them, R480 local hybrid never-seen skill is +0.819 / +0.845 / +0.837 instead of -0.837 / -1.425 / -0.388. The 10 worst windows of fold 0 are all start 0; in them the input frames already hold 8-12 m of water (max depth) that recedes to 2-7 m in the targets, and the Mozambique-trained model's error is 4-11x the persistence error per window. (Mozambique's own maximum depth is 12.4 m; 193 of 1080 Pakistan windows have input depth above that, but the damage is concentrated at start 0.) In R60 and R30 start-0 windows carry ~0% of the SSE. The REPORT describes the R480 interval as "includes 0 = unstable" but does not say why; the accurate statement is "fails only on the simulation-start windows of Pakistan".
2. **Same pattern for the strict E480 fold 0.** Held-out skill -8.099 (REPORT H6: -6.564 pooled): 6 start-0 windows carry 98.9% of the model SSE; excluding them the skill is +0.844. My split reconstruction confirms fold 0's block is starts 0..1309 and the training starts begin at 1409, so flood onset was never seen. `completion/STATE.md` already records this ("onset unseen"); REPORT.md line 59 gives the -6.564 without that caveat. Even the tile-only E480 fold 0 held-out +0.333 has 87% of its model SSE in start-0 windows (+0.870 without).
3. **Code heterogeneity across runs.** The 284 run files carry 5 different `engine.py` hashes (83 / 30 / 23 / 136 / 12 runs); only 136 match today's engine.py. The three headline frozen E60 local runs (and 9 others) were made with engine `fb4ce0ca91` (= `engine_before_batch.py.bak`, whose diff to the current engine I read: only batch/lr/guard arguments, extra architectures, the embargo (blocked only) and the guard sample differ; batch 1 maths is identical) and `model_v2.py` `a43ab5d317`, which is no longer on disk, so its difference to today's `14cbc30ee4` (kernel parameter of `build_local` etc.) could not be inspected. Indirect evidence that the default path is unchanged: the current-code seed repeats of E60 f0 (seeds 11-14: +0.8431, +0.8443, +0.8568, +0.8386, all engine `6627af098c`) sit within about one seed sd (0.008) of the frozen +0.8379 (engine `fb4ce0ca91`). `completion/results/repro_check.json` does not exist, so same-code run-to-run noise was not measured.
4. **Embargo is one window length, no more.** Min gap 55 means the first training window starts one frame (5 min) after the last target frame of the nearest held-out window (R60/R30, and R480 to within 1-2 frames): no frame is shared, but the flood state is continuous across the boundary, so held-out windows adjacent to a block edge are not independent of training. For E30/E60/E480 blk the gaps are 71-101 start units (6-8 h of simulation).
5. **Reverse runs have no spatial hold-out.** R480's 4 Mozambique tiles all overlap each other (all 16 train x held-out tile pairs overlap), R30's 4 UK tiles form 2 overlapping pairs (8 pairs incl. identical tiles), R60 has 2 overlapping padded 85x85 windows; `heldout` there is time-only, on 40-80 windows per fold, with 68-156 training windows. The REPORT header states "TIME-block folds, no tile folds"; this only confirms it.
6. **Tile filter drops wet tiles in Australia 30 m.** Two of 64 grid tiles ((768,896), (896,896); DEM exactly 0) are excluded by the "valid tile" rule but carry water >= 0.05 m in 91 of their 120 (tile, start) pairs (max depth 0.43 m, against 19 m elsewhere). Negligible for SSE, but the R30 never-seen set and E30 training set are "valid tiles", not "all wet tiles". Same rule in the forward experiments, so no inconsistency.
7. **Window inclusion depends on the target.** A window is kept if the last OBSERVED or the last FUTURE frame is wet (>= 0.05 m); windows that are dry in both are dropped. This affects model and persistence identically (checked: pool membership matches this rule exactly), but the reported skills are on "wet" windows only.
8. **Effective sample size.** The 3 never-seen "folds" score the same windows, and each never-seen region is a single flood event (E60 and E30 share the same two events); block bootstrap intervals (which I reproduced) are the honest scale of uncertainty, not the fold spread.
9. Housekeeping: `runs/E60_clstm_hybrid_full_f0of3_s1003_ep2.json` (n_train 20, epochs 2) is a leftover smoke-test result in the runs directory (kept out of the tables only by the `epochs == 30` filter in d_aggregate.py); 126 of the older run files have no `guard` field (they predate it; the aggregator treats them as guard-on).

## Could NOT verify (and why)
- **Model errors `sse_m` and everything derived from them** (every neural skill, CSI, ablation_heldout / ablation_test, history curves): recomputing needs a forward pass (TensorFlow and saved weights), which I was told not to import or run. I verified only the persistence denominators, the window sets, the arithmetic and the splits.
- Model INPUT channels other than depth (DEM block-mean, rainfall regime mapping, the geometry offsets in LAYER_GEOMETRY): I did not re-derive them. The engine's own selftest claims equality with the loader; I did not run it (it would import TensorFlow). My raw-data check covers depth frames only.
- Whether each run really trained with the stated recipe (epochs, batch, lr, seeds) beyond the internal consistency of the stored fields (history length = epochs, id/seed/n_train fields, and the training sample reconstruction above).
- Old-code equivalence: `model_v2.py` `a43ab5d317` (headline frozen runs) no longer exists to diff (note 3).
- Baselines S1-S4, S1c-S3c, and all baselines for the reverse experiments and the learning-curve baselines (only S5 for 5 forward cells was re-implemented); severity results (`results/severity/*`), SEVERITY_REPORT; the bootstrap CIs other than the E60 never-seen one; hypothesis verdicts H1-H6 beyond the E60 numbers above.
- Pre-registration timing (which analyses were decided after seeing which results): this is a documentation matter I cannot test from files.
- Memory: I kept everything memory-mapped/chunked but did not measure peak RSS.
