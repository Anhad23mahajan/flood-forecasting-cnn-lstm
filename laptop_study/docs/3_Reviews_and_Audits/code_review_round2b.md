# Adversarial code review, round 2b (2026-09-21)

Scope: only the code added or changed on 20-21 September: `completion/reverse_transfer.py` (new), `completion/engine.py` (guard switch, guard_cache, `_noguard`),
`completion/d_aggregate.py` (`load_runs`), `completion/g_report.py` (new sections), `completion/run_queue.py` (new batches), read against
`completion/results/REPORT.md`, `STATE.md`, `PREREGISTRATION.md` and the first review (`code_review.md`, not repeated here).
Method: line-by-line reading, plus single-threaded, no-TensorFlow checks on cached pool metadata and finished run JSONs (nothing edited, nothing trained, no process touched;
scratch scripts live in the session scratchpad). One quantitative side check (F8) re-fitted two cheap baselines on 74 windows.

**Bottom line: 0 critical, 3 major, 14 minor.** No path was found by which a training window shares depth frames with the held-out block (embargo verified numerically),
by which the guard change touches training, by which the R* split silently uses the wrong regions, or by which a wrong run is in a headline group TODAY (all 60 headline groups
hold each fold exactly once; only E30 clstm/causal is incomplete because it is still queued). The three majors are (1) an interpretation problem in the learning curve,
(2) a structural hole that lets guard-disabled runs sit unmarked in headline cells (and would hide a cell if a duplicate ever appeared), (3) a runner failure mode that can start every queued job at once.

---

## Findings

### F1. MAJOR (interpretation) - the "learning curve" varies the number of optimiser steps together with the number of windows
* `completion/g_report.py:334-353` (table "learning curve"), `completion/run_queue.py:99-102` (batch LC: `--n-train n --epochs 30`), `completion/engine.py:276` (`WarmupCosine(lr, epochs * (len(tr_idx) // batch), 200)`).
* What is wrong: every point of the curve trains for a fixed 30 EPOCHS, i.e. 750 / 1500 / 3000 / 9000 / 18000 optimiser steps for n = 25 / 50 / 100 / 300 / 600 (the 200-step warm-up is 27 % of the n = 25 run).
  The x-axis "number of training windows" is therefore also "number of gradient steps". The report's own training-length table shows that steps alone buy the same amount:
  local CNN 300 -> 600 windows (2x steps, 30 epochs): E60 +0.841 -> +0.861 (+0.020), E30 +0.830 -> +0.850 (+0.020), E480 +0.807 -> +0.833 (+0.026);
  300 windows for 90 instead of 30 epochs (3x steps, no new data): E60 +0.841 -> +0.862 (+0.021), E30 +0.830 -> +0.838 (+0.008), E480 +0.807 -> +0.831 (+0.024).
  The same confound applies to the per-pixel MLP row (`b_baselines.fit_mlp`: 12 epochs over n x 2000 pixels, so steps also scale with n).
  The runs also use a different seed per n (4003, 4103, ...), hence a different draw of training windows per point (not nested subsets); seed noise is 0.006-0.010 (REPORT seed table).
* Failure scenario: STATE.md 02:00 concludes "still rising at 600 (not saturated) ... the n=300 recipe under-uses the CNN's data appetite (so CNN < MLP at 300 is partly a data effect)". With the numbers above the data effect and the under-training effect are not separable; the sentence is not supported by this design (and STATE 02:40 itself says the 30-epoch CNN is under-trained).
* Fix: relabel the section "fixed 30 epochs, so steps scale with n" and drop the "data appetite" claim; or add the two missing cells that separate the effects (n = 600 for 15 epochs, n = 300 for 60 epochs, so that steps match), or equalise steps across n.

### F2. MAJOR (structural) - guard-disabled runs are indistinguishable from normal runs in every headline group, and three headline cells contain them unmarked
* `completion/d_aggregate.py:41` (`frozen = batch == 1 and lr == 1e-3 and _default_seed(r)`; no `guard` term), `:44` (headline key has no guard flag; only the `extra` key at `:46` does), `completion/g_report.py:35-39, 62-75` (`grp`, headline rows), `completion/g_report.py:426`, `completion/patch_guard.py:28-38`.
* What is wrong: STATE.md 23:25 decided that a guard-disabled run with the default seed counts as the frozen-recipe run, and the code does exactly that. Consequences:
  (a) Today 3 headline cells contain guard-disabled runs and the headline tables do not say so: E60 v1 hybrid (3 of 3 folds), E30 v1 hybrid (3 of 3), E60 clstmf forced (fold 1). REPORT.md:12, 28, 17 show them as ordinary rows,
  while REPORT.md:241 claims "the tables mark them" (only the v1-vs-final and clstmf per-fold tables do). The numbers themselves are right if the false-alarm diagnosis is right (noguard v1 runs give +0.181..+0.195, equal to the original v1 figures; the clstmf fold trained normally, val skill +0.36..+0.48 at epochs 13-30).
  (b) The rule has no way to tell a false alarm from a real collapse: a guard-disabled run whose network really died would enter the headline mean with skill ~0 and no marker. The pre-registration (line 41) says failed runs are "reported, not silently retried".
  (c) Duplicate hazard (double count): the group key omits the guard flag, so if a normal run and its `_noguard` twin of the same fold ever both exist (for example the normal job is re-run after its log has been overwritten), `runs` holds 4 entries for K = 3, `complete()` (`d_aggregate.py:59-61`) fails and `g_report.stat()` returns None: the cell silently becomes "-" in REPORT.md (only summary.md shows "4/3 (incomplete)"). The queue currently prevents this only through the `aborted()` log rule (F12).
  (d) `patch_guard.py` is documented as idempotent, but d_aggregate.py was edited afterwards (02:53). Its `old` pattern (line 28) still matches the current line 41, so re-running it silently re-adds `and r.get("guard", True)`: guard-disabled runs would flip to "extra", the v1 rows and the E60 clstmf forced cell would vanish from the headline tables, and nothing would report it.
* Failure scenario for (c): `E60_v1_hybrid_full_f0of3_s1003_ep30.json` written next to the existing `E60_v1_hybrid_full_noguard_f0of3_s1003_ep30.json` -> E60/v1/hybrid group has 4 runs -> the v1 row disappears from REPORT.md with no message.
* Fix: mark guard-disabled cells in the headline table (dagger plus footnote), make `stat()` raise/print when a group is over-complete instead of returning None, add `guard` to the frozen key only for reporting purposes (or refuse to load both twins), and delete `patch_guard.py` (its job is done and it now contradicts d_aggregate).

### F3. MAJOR (process robustness) - `count_running()` returns 0 on any error, which makes the runner start every queued job at once
* `completion/run_queue.py:171-180` (`except Exception: return 0`), used at `:209` (`while todo and count_running() < a.workers`).
* What is wrong: the worker cap is enforced only through a PowerShell/CIM query. A timeout (60 s limit), a WMI hiccup or an unparsable output makes it return 0; the inner `while` then pops and starts jobs until `todo` is empty (each iteration re-queries, fails again, returns 0). The runner's own `running` list is ignored.
* Failure scenario: 4 heavy trainings saturate the CPU, one CIM call exceeds 60 s while 30 jobs are queued -> 30 python processes at about 1.2 GB each on a 15.5 GB machine (STATE.md 22:33 records an earlier incident: "runners raced past the 4-job cap -> 5 jobs, free RAM 0.4 GB = hang risk").
* Fix: `return max(count_running_via_cim(), len(running))` (or count only the runner's own `running`, and use CIM just to add external jobs); on exception fall back to `len(running)`, never 0.

### F4. MINOR - the REPORT's explanation of the guard aborts is the superseded one
* `completion/g_report.py:250` (label "guard disabled: it fires falsely on v1's tiny early outputs"), `:426` (paragraph), `REPORT.md:123-128, 228, 241`.
* STATE.md 23:58 established the root cause: the check used the 12 EARLIEST held-out windows (near-static, so a healthy network predicts an identical, near-zero change there); the engine was then changed to sample the whole held-out set. The REPORT still says the cause is v1's tiny early outputs and never mentions that the check was changed; the dead-run table header describes the spread "across validation windows" although all 8 listed aborts came from the old earliest-12 sampling.
* Scenario: a reader concludes that the v1 model is unstable at epoch 2; the evidence says the guard was mis-specified. Fix: reword ("aborted by a check that sampled the earliest, near-static windows; the check was corrected on 21 Sep; the aborted jobs were re-run with the check disabled") and drop the v1-specific claim.

### F5. MINOR - `rain_shifted_cells` is printed in the ablation table without a caveat although the first review (F6) showed it is partly a no-op
* `completion/g_report.py:169` (key list), `:176`; `completion/engine.py:206-207` (`np.clip(r + shift, 0, H - size)` still there); `REPORT.md:63-79` (column identical to `none` to 3 decimals in all 15 rows).
* Scenario: a reader takes "rain_shifted_cells = none" as evidence of insensitivity to crop offsets, which `STATE.md`/`LOG_ROUND2_narrative.md` say must NOT be cited. Fix: drop the column or footnote it ("clipping makes the shift a no-op for a large share of windows; cite rain_removed instead").

### F6. MINOR - "held-out tiles (pooled)" columns mix two aggregators
* `completion/g_report.py:48-50` (`bl()` = mean of the three fold skills) versus `:45` / `:73` (`held_pooled` = skill of the pooled sums); same mix in the reverse table (`:387` vs `:395`), whose text at `:382` says "(pooled)".
* Example: E480 local CNN held-out is +0.430 pooled (REPORT.md:45) but +0.383 as mean of fold skills (REPORT.md:197); the E480 S4c baseline +0.868 in the same column is a mean of fold skills. No conclusion flips (0.43 vs 0.87), but the header states an estimand the baseline cells do not have. Fix: state the aggregator per row, or pool the baselines (their JSONs hold no per-window errors, so relabel instead).

### F7. MINOR (interpretation) - "seed repeats" also change the training-window draw and the held-out window draw
* `completion/g_report.py:273-291`; `completion/engine.py:453-455` (`rng = default_rng(seed)` picks `tr_idx` and then `va_idx`).
* Measured for E480 fold 0: seed 1003 and seed 12 share 191 of 300 held-out windows; the training windows differ too. The held-out "sd over seeds" (0.424, 0.358, 0.200 for E480) therefore mixes optimisation noise with evaluation-sample and training-sample variation (the never-seen column is unaffected: same test pool). The conclusion "E480 held-out is optimisation-limited" is nonetheless supported by the same-window comparisons (OPT +0.771 and 90 epochs +0.678 vs +0.430 at seed 1003). Fix: label the held-out column "seed + data draw", or fix the held-out sample across seeds.

### F8. MINOR - R60 trains and fits baselines on padded windows without a mask (assumption of the first review no longer holds)
* `completion/engine.py:274` (persistence normaliser from the raw 85x85 pool arrays), `:286` (loss over all 128x128 pixels), `:325` (`x, y, _ = compose(...)` mask dropped); `completion/b_baselines.py:58, 87, 127` (pixel samples drawn from all 128x128 pixels); `completion/reverse_transfer.py:32` (UK 60 m training pool = padded windows).
* Evidence: for R60 the logged train/persistence loss equals 0.44 x (1 - train_subset skill) in every fold (0.062 / 0.141; 0.043 / 0.097; 0.060 / 0.135; 0.44 = 85^2 / 128^2), so the training loss is scaled by the padded fraction and 56 % of sampled baseline pixels are padded zeros.
  For the neural nets the padded region is learned to zero error (that is why the ratio is exactly 0.44), so the effect is mostly loss scale (interacts with `clipnorm=1.0`). For the baselines it changes the fits: fold 0, 74 training windows, scored on every 4th never-seen window (239) / the 40 held-out windows:
  S5 lookup table 0.537 / 0.730 as run versus 0.519 / 0.764 with real pixels only; S3c linear 0.230 / 0.374 as run versus 0.283 / 0.428.
* Effect: R60 baseline cells move by 0.02-0.05 in either direction; comparisons between R60 and R30/R480 baselines at that resolution are not clean. Fix: sample only `mask` pixels in `b_baselines.fit*`, weight the loss by the mask, or mention it next to the R60 rows.

### F9. MINOR - the R ablation subsample is deterministic but its "none" is not the reported test skill, and it is not representative for R480
* `completion/reverse_transfer.py:52-56` (`linspace(0, len-1, 400)`), `completion/results/summary.json` (`ablation_test` means for R groups).
* Deterministic and evenly spread in time (pool order is start-major): yes. Representative for a ratio-of-sums metric: not for R480. `ablation_test["none"]` versus `test.skill` for R480 folds 0/1/2: -0.510 vs -0.837, -0.996 vs -1.425, -0.089 vs -0.388 (a few deep windows dominate); R30 0.657 vs 0.673, 0.632 vs 0.663, 0.639 vs 0.651; R60 0.751 vs 0.742, 0.737 vs 0.728, 0.752 vs 0.744.
  Not shown in REPORT.md today; it would mislead only if the R ablations were tabulated against the test skill. Fix: report deltas to the subsample's own `none`, and say so; or use all windows.

### F10. MINOR - guard window sampling: coverage and degenerate sizes
* `completion/engine.py:302` (`va_idx[::max(1, len(va_idx) // 12)][:12]`), `:336-339`.
* The slice keeps 12 windows but stops early whenever `len // 12` rounds down: for 300 held-out windows it covers indices 0..275 (92 %), for 78 (R480 fold 0) 0..66 (85 %), for 40 (R60) 0..33 (85 %), for 80 0..66 (84 %); worst case about 50 % (len 23 uses only the first 12). The last, latest windows are never checked; harmless for a dead-network test but "spread over the whole held-out set" is only approximately true (`np.linspace(0, len - 1, 12).astype(int)` would be exact).
* Tiny sets: `len(va_idx) == 0` -> `np.stack([])` raises ValueError; `len(va_idx) == 1` -> `np.std(D, axis=0)` is identically 0, so the guard fires at the first allowed epoch for any network. No current split is that small (smallest held-out set is 40), so latent. The absolute threshold 1e-5 m (0.01 mm) also still fires on a legitimate near-persistence model whose outputs are tiny (the v1 case), which is why the switch exists.

### F11. MINOR - JSON/table labels of the reverse runs
* `completion/reverse_transfer.py:46` (`held = set()`), `completion/engine.py:477-478` (`blocked=blocked` stays False, `held_out_tiles=[]`), `completion/d_aggregate.py:79-99, 163`.
* R* result files say `blocked: false`, `held_out_tiles: []` although the hold-out is a time block with embargo (`heldout` = time-held-out windows of the SAME tiles, which is documented in the module docstring and in the report table header). `summary.md` prints "held-out pooled [95% CI]" for R rows with `tile_bootstrap`, whose clusters are the 2-4 TRAINING tiles (heavily overlapping corner tiles) times time blocks, not held-out tiles. `REPORT.md` does not print that interval for R rows, so no reported number is wrong. Fix: add a `split: "time-block"` field and skip the tile interval for R groups.

### F12. MINOR - queue skip rule treats logs of the old guard as "reproducible collapses"
* `completion/run_queue.py:197-204` (`aborted`), `:76-97` (FXR, FXG, FXR2), `completion/g_report.py:400-426` (dead-run table).
* Any log containing "DEAD NETWORK after epoch " excludes the job for good (base v1 jobs E60/E30, FX E60 fold 1, FXR retry). All 8 such logs were produced by the old earliest-12 check, which STATE.md 23:58 shows to be a false-alarm generator, so "recorded (reproducible) collapse" (comment at `:198`) is not established; the retry (seed 1014, aborted at epoch 6) was never re-tested with the corrected check. Also the dead-run table shows "not run" for `E60_clstmf_forced_retry_f1` because `name.replace("_full_", ...)` (`g_report.py:419`) does nothing to a name without `_full_`.
* Fix: reword the comment; if the retry matters, re-run it once under the corrected check.

### F13. MINOR - smaller queue issues
* `run_queue.py:88`: `base["name"].replace("_full_", "_full_noguard_", 1)` assumes no `_aug`/`_blk` suffix (engine puts `_noguard` after `_blk`); harmless for the current CG jobs (v1/hybrid/full), wrong for any other config.
* `run_queue.py:183-185, 211-212`: `log_active` uses a 6-minute log-mtime window, but a running job can be silent much longer (after the last epoch line the R30 job evaluates 3670 never-seen windows and then 8 ablation passes over 400 windows before it prints again); a restarted runner would then start a duplicate of a job still running under the old runner. Jobs skipped as "active" are dropped silently and never retried; the final "QUEUE FINISHED; failed: []" hides them.
* `run_queue.py:86, 200`: `open(lf).read()` without closing (handle leak, many logs).
* The batch list is frozen when the runner starts (`batch()` is evaluated once in `main`); the running runner (pid 9896, started 03:00, same minute as the last `run_queue.py` edit) will not see later batch edits until it is restarted. The dynamic CG batch is likewise evaluated once.

### F14. MINOR - g_report robustness details
* `completion/g_report.py:190`: training-time ablation cells accept `len(r) == 3` instead of `A.complete(r)`; with a duplicate fold and a missing fold the mean would be wrong.
* `:224-226`: any non-frozen forced run of clstm/clstmf is labelled "RETRY after the frozen-seed run was aborted by the guard"; correct today (only the seed-1015 retry exists) but the label is inferred, not stored.
* `:209-216`: `every()` calls `A.load_runs()` (re-reads all 284 run JSONs) on each call; about 40 calls per report (performance only).

### F15. MINOR (latent) - `small_pool` in `load_runs`
* `completion/d_aggregate.py:38, 44`: for exp starting with "R", any `n_train <= 300` is accepted and mapped to key `NTR_F`. A reverse run with a deliberately smaller `--n-train` (a learning curve on R*) would land in the headline key of the same fold (the run id has no `n_train`, so it needs a different seed to be written at all), and `complete()` would hide the cell. None exists today. Fix: require `r["n_train"] == min(300, pool size)` or add `n_train` to the R group key.

### F16. MINOR (provenance) - engine hash stamps
* `code_md5` (`engine.py:481`) is taken at the END of a run. Five engine versions are stamped: `fb4ce0ca91` (12 runs, until 10:30), `e3e8e93d72` (30, to 13:18), `72eeb40485` (83, 13:44-21:57, contains the embargo fix; all strict-split runs), `923283620c` (23, 22:01-23:47, old 12-earliest guard) and `6627af098c` (136, from 00:11, the current file). Runs that started before an edit and finished after carry the new hash. The guard change alters only the abort decision (F-checks below), so results are unaffected, but the stamp does not prove which guard was active for a run. All 8 aborts and all 23 `923283620c` runs belong to the old check.

### F17. MINOR (documentation) - no `--no-guard` in the engine docstring, reverse runner cannot disable the guard
* `engine.py:20-22` docstring lists no `--no-guard`/`--batch`/`--lr`/`--blocked`; `reverse_transfer.py:73-81` exposes no `--no-guard`, so an R* run aborted by the guard has no re-run path. No R* run was aborted.

---

## What was checked and found correct

### (1) `completion/reverse_transfer.py`
* Embargo. Reproduced `time_split_for` on the cached pool metadata for all 9 experiment x fold cases. Blocks are the K = 3 `array_split` thirds of the 60 start times; training keeps `start < b_lo - 54` or `start > b_hi + 54`. A window spans [start, start + 54], so the condition is exact and slightly conservative (a shared boundary frame at distance 54 is excluded). Measured minimum start distance between any training and any held-out window: 56/57/56 (R480), 55/55/55 (R60), 55/55/55 (R30); span-intersecting pairs: 0 in all 9; sets of start times disjoint; the three held-out blocks partition all 60 starts (R480 78/80/80 windows, R60 40/40/40, R30 80/80/80; training pools 156/150/154, 74/68/74, 148/136/148, identical to the `n_train` in the run JSONs).
* Regions. Training pool = `build_pool(small region, res, "test")` (Mozambique 480: 238 windows, 4 tiles; UK 60: 120 windows, 2 padded tiles of 85 px; UK 30: 240 windows, 4 tiles); never-seen pool = `build_pool(big region, res, "test")` (Pakistan 480: 1080 windows, 18 tiles; Australia 60: 953, 16 tiles; Australia 30: 3670, 62 tiles). Different regions, different cache keys (`pool_<region>_<res>_test`), all 60 start times each.
* Naming. Run ids from `E.run` for `R*` (`R480_local_hybrid_full_f0of3_s1003_ep30`, seed `1000 + 10 fold + 3`) equal what `run_queue.reverse()` expects; all 36 RN and 27 RB outputs exist under exactly these names.
* Monkey-patching works: `E.run` and `b_baselines.main` look `split_for` and `ablation_skills` up as module globals/attributes at call time; evidence in the outputs: `n_train` 68-156 (a tile-fold split gives 300), `held_out_tiles == []`, held-out n 40-80, `ablation_test.none != test.skill` (subsample active). `_orig_ablation` is captured before the patch, so there is no recursion. Positional call signatures (`split_for(exp, fold, K, seed, n_train, n_val, log, blocked=...)`) match `time_split_for`.
* Ablation subsample: `np.linspace(0, len - 1, 400).astype(int)`; for len > 400 the step is > 1, so no duplicates; deterministic; see F9 for representativeness.

### (2) `completion/engine.py` guard
* `guard=True` default: the abort condition is `guard and ep * len(tr_idx) >= 400 and ep >= 2 and spread < 1e-5`, identical to the previous one when `guard` is True; run id gets `_noguard` only when False (placed after `_blk`, before `_b%d`/`_lr%g`); result gets an extra `guard` key; CLI `--no-guard` maps to `guard=not a.no_guard`; queue names for CG/FXG match the written files.
* RNG/training independence: `guard_cache` is built with `compose()` (pure NumPy reads, no RNG) after `tf.keras.utils.set_random_seed(seed)` and before `rng = np.random.default_rng(seed)` is created; `compose` has no random call; the per-epoch `predict()` on guard windows uses the already-traced input-signature function with `training=False` (no dropout draw, no retrace). Loss, optimiser, schedule, `perm`, augmentation draws are untouched. (Argued from the code; a bit-for-bit re-training comparison was not run.)
* Indexing: `va_idx` is a sorted Python list; the slice is valid for lists and arrays; see F10 for coverage and the degenerate sizes.

### (3) `completion/d_aggregate.py`
* Ran `load_runs()` on the 284 run files: 60 headline groups, every one holds folds {0,1,2} exactly once except E30 clstm/causal (1 fold, queued). 22 extra groups (seeds 11-14, batch-8/lr 3e-3, the seed-1015 retry) each hold one fold set, none overlaps a headline group. Runs excluded by both loaders: 52 (all `ep90`, learning-curve `n_train` 25/50/100/600, one 2-epoch smoke test), as intended.
* The 7 guard-disabled runs are in headline groups by design (F2). No blind/design run, no `epochs != 30`, no `n_train != 300` (forward experiments) run is in a headline group. R* groups: n_train 148/136/148 etc. map to the single key `n_train = 300`; no collisions.
* Guard-disabled versus normal double count: impossible with the current files; possible structurally (F2c).

### (4) `completion/g_report.py` / REPORT.md
* Filters: `fold_mean(..., frozen_seed=True/False)` keeps the default-seed run for n = 300 and the non-default (4xxx) seeds for the other n; `raw_runs` filters batch 1, lr 1e-3, epochs, `n_train`, variant, regime, augment, blocked correctly; the training-length rows use the epochs-90 default-seed runs; the paired equal-length bootstrap pairs folds in sorted order on the same start blocks. H3/H4 implement the pre-registered rules literally (H4 verdict strings verified by hand for E60/forced -0.012 [-0.031, +0.002] and E480/hybrid +0.004 [-0.011, +0.019]). H5 3/5 recomputed from the table by hand; H2 uses the clipped S3c as intended.
* Seed table order (frozen first, then seeds ascending), batch-8 row and the strict-split table (seeds 11-14 plus the batch-8 run at seed 1003) match the files. The clstmf/v1 per-fold tables show the guard-disabled rows with the note; the dead-run table regex matches the message text.
* Silent `None` handling: a missing/over-complete group becomes "-" in the headline (F2c, F14) but `tot` in H1/H2/H5 reports how many cells entered.

### (5) `completion/run_queue.py`
* Programmatic check (imported `run_queue`, no TensorFlow): for 32 batches (S BL K OPT FX FXR CG FXG FXR2 LC LCB EP EXS EPC RB RN SX SXF AX BX FXX OPTX S2 F A B B2 C E E2 D EX) rebuilt the engine run id / baseline file name from each job's command line with the engine's own format (seed, `_lr%g`, `_b%d`, `_blk`, `_noguard`, `_ep`, `_n<n_train>`) and compared with `out`: 0 mismatches over the 428 engine/baseline/reverse jobs (of 454 listed; the 26 severity jobs of S/S2/SX/SXF were checked against the existing files instead: 8/8, 6/6, 5 of 6 present under the queue's names, SXF not started); no two batches map different job names to one output or one name to different outputs; the only run file not owned by a batch is the 2-epoch smoke test. Severity outputs match `severity_study.py:100`.
* Finished/missing: LC 36/36, LCB 36/36, EP 9/9, EPC 6/6, EXS 5/5, RB 27/27, RN 36/36, OPT 12/12, F 36/36, BX 15/15; pending: D 0/6 (running), OPTX 0/3, FXX/AX/SXF/B2/EX as expected; FX 5/6 and C 3/9 are the guard-aborted base jobs (CG covers them).
* `aborted()` and `log_active()` are read-only checks on log files; the only real race found is F3/F13 (cap enforcement, silent-log window).

---

## Could not verify
* That the seven guard-disabled headline runs would have passed the corrected guard (needs a re-run), and that runs which passed the old check would also pass the new one (plausible: the new windows are more diverse; not proven).
* Bit-for-bit equality of a guard=True run before and after the 23:54 edit (no re-training; argued from code).
* The size of the F8 effect on the R60 neural results (only baselines were re-fitted); expected small because the padded region is learned to zero error.
* Whether the two majors F2(c)/F3 ever fire: both are structural; nothing in the current logs shows them firing.
* I did not review `visualize.py` and `severity_study.py` (unchanged), nor `model_v2.py`.
