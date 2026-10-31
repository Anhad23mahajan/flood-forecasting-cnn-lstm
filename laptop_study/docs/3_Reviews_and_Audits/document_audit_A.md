# Document audit A: results_narrative.md (8.1-8.15, 8b), log_established.md, README_round2.md

Auditor: independent, read-only (only this file was written). Date of audit: 2026-09-21.

Abbreviations. **N** = `completion/results/results_narrative.md`, **L** = `completion/results/log_established.md`, **R** = `completion/README_round2.md`.
**RUNS** = recomputed by me from `completion/results/runs/*.json` (310 files) with the project's own aggregation functions (`d_aggregate.load_runs / start_bootstrap / paired_start_bootstrap / tile_bootstrap / pooled_skill`, the same code `g_report.py` uses; frozen recipe = batch 1, lr 1e-3, default seed, 30 epochs, n_train 300).
**BASE** = mean of the three `baselines_<exp>_<regime>_f*of3_n300.json`. **SEV** = `SEVERITY_REPORT.md` (I also recomputed every accuracy, rule, reason-truth and Grad-CAM cell from `severity/SEV_*.json`: all matched; the paired capsule-minus-dense intervals reproduce to within Monte-Carlo noise, +0.076 [+0.051,+0.102], +0.005 [-0.012,+0.022], +0.056 [+0.029,+0.082]).
**REP:Lx** = line x of `REPORT.md`. **CR1 / CR2b** = `review/code_review.md` / `code_review_round2b.md`. **V2b** = `review/verification_round2b.md`.

## 0. Important caveat about the source of truth itself

`REPORT.md` and `report.json` on disk (both 2026-09-21 08:02) are DEGRADED: every headline table cell is "-", H1 and H2 read "0/0 cells pass", `report.json` has `"headline": {}`, the "ablation", "seed repeat", "strict split", "start-0" and "optimisation" tables are empty, and the data-warning section lists only four E60 groups "folds present [0]". Only the learning-curve, equal-steps, training-length, H4-at-90-epochs and dead-network tables are filled. `summary.md` is older still (2026-09-20 13:26). All 310 run files are present and complete today (62 headline groups, each with folds 0,1,2 exactly once, except E30 clstm/causal = fold 0 only), so the generator simply ran while the runs were not all visible, or was broken. Consequence: none of the headline numbers in N/L/R can currently be matched to a REPORT.md cell; I recomputed them from the run JSONs (RUNS/BASE). The sections that REPORT.md still contains (learning curve, equal steps, epochs-90, H4 at 90 epochs, guard table) match N exactly. **Regenerate REPORT.md (`python completion/g_report.py`) before this handoff is published.**

## 1. Statement-by-statement table (verdicts: OK / ERROR / OVERCLAIM / CONTRADICTION)

| # | Doc | Quote (short) | Claimed | Actual (source row) | Verdict |
|---|---|---|---|---|---|
| 1 | N8.1 | "straight-line extrapolation of the depth history +0.32" | +0.32 | S1 E60 hybrid 0.3232 (BASE) | OK |
| 2 | N8.1 | "linear with rain/state/trend +0.35" | +0.35 | S3c E60 hybrid 0.3447 (BASE) -> +0.34 (double rounding via 0.345) | ERROR |
| 3 | N8.1 | "lookup table (current depth x last one-step change) +0.79" | +0.79 | S5 0.7857 (BASE) | OK |
| 4 | N8.1 | "per-pixel MLP +0.85" | +0.85 | S4c 0.8464 (BASE; REP:L89 n=300 +0.846) | OK |
| 5 | N8.1 | "local CNN +0.84 [+0.80, +0.89]" | +0.84 [+0.80,+0.89] | RUNS 0.8409 [0.798, 0.892] | OK |
| 6 | N8.1 | "small ConvLSTM +0.87 [+0.81, +0.92]" | upper bound +0.92 | RUNS 0.8655 [0.8082, 0.9146] -> [+0.81, +0.91] | ERROR |
| 7 | N8.1 | "At 30 m: +0.35 / +0.36 / ..." first value (straight-line trend) | +0.35 | S1 E30 hybrid 0.3447 -> +0.34 | ERROR |
| 8 | N8.1 | "... +0.36 / +0.76 / +0.85 / +0.83 / +0.84" (30 m) | as written | S3c .3588, S5 .7629, S4c .8540, local .8304, clstm .8398 | OK |
| 9 | N8.1 | Mozambique "+0.15 / +0.18 / +0.74 / +0.83 / +0.81 / +0.81" | as written | .1509 / .1823 / .7370 / .8288 / .8070 / .8106 | OK |
| 10 | N8.1 | "original CNN-LSTM scored roughly +0.05 to +0.2" | +0.05..+0.2 | v1 hybrid .1867 / .1876 / .0614 (RUNS) | OK |
| 11 | N8.1 | "a table with a few hundred cells reaches about 90% of the best score" | ~90% | 12x17=204 cells; S5/ConvLSTM = .908 (E60), .908 (E30), .909 (E480) | OK |
| 12 | N8.1 | "adding rain, terrain, neighbouring pixels or recurrence moves it by at most a few hundredths" | <= few 0.01 | rain 0.008-0.02, terrain 0, kernels +-0.014, recurrence <= +0.025 (rows 20-46) | OK |
| 13 | N8.2 | local CNN ordering UK 60 m "+0.82 / +0.84 / +0.87" | as written | .8222 / .8409 / .8661 (RUNS) | OK |
| 14 | N8.2 | UK 30 m "+0.81 / +0.83 / +0.85" | as written | .80503 / .8304 / .8513 | OK |
| 15 | N8.2 | Mozambique "+0.79 / +0.81 / +0.83" | as written | .7938 / .8070 / .8343 | OK |
| 16 | N8.2 | "partly-future rain worth +0.01 to +0.025; known future rain +0.02 to +0.03 more" | as written | hybrid-causal .0187/.0254/.0132; forced-hybrid .0252/.0209/.0273 | OK |
| 17 | N8.2 | ConvLSTM Mozambique "causal +0.81, hybrid +0.81, forced +0.77" | as written | .8102 / .8106 / .7653 | OK |
| 18 | N8.2 | ConvLSTM UK 60 m "flat (+0.85 / +0.87 / +0.85)" | as written | .8530 / .8655 / .8541 | OK |
| 19 | N8.2 | "The code review (F4) traced this to the input layout" | cause established | CR1 F4: "Cause is my hypothesis, UNVERIFIED"; only 8.8 later supports it | OVERCLAIM |
| 20 | N8.3 | ConvLSTM UK 60 m ablations "+0.87 / +0.86 / +0.87 / +0.02 / +0.86" | as written | none .866, rain_removed .857, terrain .865, frozen .020, rain-strictly-past (`rain_shifted_earlier`) .863 | OK |
| 21 | N8.3 | local CNN UK 60 m "+0.84 / +0.83 / +0.84 / -0.01 / +0.83" | as written | .841 / .833 / .841 / -.006 / .833 | OK |
| 22 | N8.3 | ConvLSTM Mozambique "+0.81 / +0.79 / +0.81 / -0.01 / +0.79" | as written | .811 / .791 / .811 / -.007 / .791 | OK |
| 23 | N8.3 | local CNN Mozambique "+0.81 / +0.79 / +0.81 / -0.01 / +0.78" | as written | .807 / .790 / .807 / -.008 / .784 | OK |
| 24 | N8.3 | history in wrong order "-2.6 to -3.7" | -2.6..-3.7 | the four cited models -2.649 / -3.037 / -2.929 / -3.666 (E30: -1.39 / -1.14; v2lin -1.65 / -7.83) | OK |
| 25 | N8.3 | "rainfall is worth 0.01-0.03" | 0.01-0.03 | never-seen test-time: rain_removed 0.008-0.020, rain strictly past up to 0.023; 0.03 is only reached on held-out tiles (0.02-0.07) | OVERCLAIM |
| 26 | N8.3 | "`rain_shifted_cells` ... partly a no-op ... not used" | no-op | column equals `none` to 3 decimals in all runs; CR2b F5 | OK |
| 27 | N8.4 | kernels 1/5/11 UK 60 m "+0.85 / +0.84 / +0.84" | as written | .8484 / .8409 / .8414 | OK |
| 28 | N8.4 | UK 30 m "+0.84 / +0.83 / +0.83" | as written | .83500 / .8304 / .8306 | OK |
| 29 | N8.4 | Mozambique "+0.82 / +0.81 / +0.79" | as written | .8179 / .8070 / .7931 | OK |
| 30 | N8.4 | "per-pixel MLP (+0.85, +0.85, +0.83) does as well as any CNN" | as written | S4c .8464 / .8540 / .8288 (the ConvLSTM is +0.02 higher on UK 60 m only) | OK |
| 31 | N8.4 | Pakistan held-out pooled "+0.24 (k=1), +0.43 (k=5), -0.66 (k=11), intervals include 0" | as written | .2404 [-.523,.906]; .4304 [-.113,.895]; -.6566 [-2.59,.888] | OK |
| 32 | N8.4 | "ConvLSTM (+0.84) and the per-pixel MLP (+0.87) are steady" | as written | ConvLSTM pooled .8448 [.806,.901]; MLP S4c .8682 = MEAN OF FOLDS, not pooled (CR2b F6) | OK |
| 33 | N8.5 | UK 60 m hybrid "+0.025 [+0.004, +0.043]" | as written | .0246 [.0035, .0435] (paired block bootstrap) | OK |
| 34 | N8.5 | "causal +0.031" | +0.031 | .0308 [.0084,.0515] | OK |
| 35 | N8.5 | Mozambique hybrid "+0.004 [-0.011, +0.019] (equivalent)" | as written | .0036 [-.0107,.0189]; |d|<.02 and interval inside +-.03 -> equivalent | OK |
| 36 | N8.5 | Mozambique forced "-0.069" | -0.069 | -.0690 [-.095,-.038] | OK |
| 37 | N8.5 | UK 30 m hybrid "+0.009 [-0.014, +0.031]"; Moz causal "+0.016 [-0.004, +0.034]" | as written | .0095 [-.0142,.0308]; .0163 [-.0036,.0336] | OK |
| 38 | N8.5 | "(report reads 'not separable from no difference')" | report wording | g_report.py:143 prints "clstm larger (by the rule, but the interval includes 0)"; the quoted phrase exists only in N / PROJECT_LOG | ERROR |
| 39 | N8.5 | "ConvLSTM better than MLP only on UK 60 m (+0.02)" | +0.02 | clstm minus S4c: +.019 / -.014 / -.018 | OK |
| 40 | N8.5 | 90-epoch pairs "+0.004 [-0.021, +0.027]" and "-0.014 [-0.029, +0.006]" | as written | .0042 [-.0209,.0274]; -.0136 [-.0286,.0064] (REP:L117-118) | OK |
| 41 | N8.5 | "equivalent by the pre-registered rule in both independent pairs" | equivalent | both intervals inside +-0.03 and |d|<0.02 (PREREGISTRATION H4) | OK |
| 42 | N8.5 | "it reaches the same skill" | same | equivalence only within +-0.03 (E480 ConvLSTM 0.818 vs CNN 0.831 at 90 epochs) | OVERCLAIM |
| 43 | N8.6 | full model "+0.841 / +0.807"; without depth history "+0.172 / +0.086" | as written | .8409/.8070; .1715/.0863 | OK |
| 44 | N8.6 | without depth AND rain history "+0.159 / +0.085"; rain history only "+0.839 / +0.802" | as written | .1592/.0855; .8385/.8022 | OK |
| 45 | N8.6 | without any rain "+0.835 / +0.798"; without terrain "+0.842 / +0.812" | as written | .8349/.7983; .8424/.8125 | OK |
| 46 | N8.6 | "depth history worth +0.669 and +0.721 (fold sd 0.003 and 0.008); rain 0.005-0.009" | as written | .6694 / .7207; sd(ddof=1) .0026 / .0076; rain .0060 / .0087 | OK |
| 47 | N8.6 | "(a model with rain and state but no trend reaches only about +0.26)" | +0.26 | S2 hybrid never-seen: E60 .2074, E30 .1878, E480 .1001; +0.26 is the E60 HELD-OUT-tile value (.2602) | ERROR |
| 48 | N8.7 | strict split "E60 local +0.756 -> +0.716; E60 ConvLSTM +0.782 -> +0.645" | as written | pooled held-out .7557 -> .7156; .7820 -> .6454 | OK |
| 49 | N8.7 | "E30 local +0.730 -> +0.686; E480 local +0.430 -> -6.56; E480 ConvLSTM +0.845 -> +0.742" | as written | .7305 -> .6863; .4304 -> -6.5644; .8448 -> .7421 | OK |
| 50 | N8.7 / L | "H6 passes in 2 of 5 cells" | 2/5 | drops .040 (E60 local), .044 (E30 local) < .05; .137, >7, .103 fail | OK |
| 51 | N8.7 | "6 of the 480 strict held-out windows carry 98% of the plain CNN's squared error" | 6/480, 98% | 6 start-0 windows of 480; share .985 | OK |
| 52 | N8.7 | "(14,853 against 780 for persistence in that fold)" | all start-0 windows of the fold | those are the model / persistence SSE of the SINGLE worst window; the 6 start-0 windows carry 33,901 vs 1,262 (fold total 34,294 vs 3,769) | ERROR |
| 53 | N8.7 | "without them its strict score is +0.842 (ConvLSTM +0.866)" | as written | .8415 (CNN), .8665 (ConvLSTM), pooled over 3 folds | OK |
| 54 | N8.7 | tile-only "12 of 900 windows carry 82% ... (+0.865 without them)" | as written | 12/900, .821, .8652 | OK |
| 55 | N8.7 / L | never-seen "unaffected by the strict split (E480 +0.807 -> +0.803; E60 +0.841 -> +0.837)" | as written | .8070 -> .8027; .8409 -> .8367 (E30 local -.012, E60 ConvLSTM -.006 are inside the 0.02 resolution of 8.10) | OK |
| 56 | N8.7 / L | four more seeds "-6.9, -9.1, -8.1, -8.9; frozen -8.1; batch-8 -5.7; never-seen +0.76 to +0.79" | as written | -6.933, -9.082, -8.140, -8.854; -8.099; -5.730; test .763-.785 | OK |
| 57 | N8.7 | "The failure is systematic - a property of the data (the onset is absent from training), not of one unlucky run" | property of the data | reproduces across 5 seeds + 1 recipe on ONE fold for ONE architecture; the ConvLSTM on the same fold scores +0.722, the lookup table +0.509, plain-CNN folds 1,2 are +0.848 / +0.738 (RUNS, BASE strict) | OVERCLAIM |
| 58 | N8.7 | batch-8 never-seen "E60 +0.829 vs +0.841, E60 ConvLSTM +0.853 vs +0.866, E30 +0.822 vs +0.830, E480 +0.798 vs +0.807, E480 ConvLSTM +0.785 vs +0.811; 0.008-0.026 lower" | as written | .8288/.8409; .8534/.8655; .8222/.8304; .7976/.8070; .7853/.8106; diffs .0082-.0253 | OK |
| 59 | N8.7 | "held-out Pakistan-tile score goes from +0.430 to +0.771"; ConvLSTM "held-out tiles +0.837 vs +0.845" | as written | pooled .4304 -> .7714; .8368 vs .8448 | OK |
| 60 | N8.7 | 90 epochs never-seen "+0.021 (E60, +0.862), +0.008 (E30, +0.838), +0.024 (E480, +0.831)" | as written | .8618 (+.0209); .8382 (+.0078); .8313 (+.0243) | OK |
| 61 | N8.7 | "its Pakistan held-out score from +0.383 to +0.678" (two sentences after "+0.430") | one aggregator | +0.383 / +0.678 are MEAN OF FOLD SKILLS; the +0.430 quoted for the same model in the same paragraph is POOLED (pooled 90-epoch value is +0.699) | CONTRADICTION |
| 62 | N8.7 | ConvLSTM at 90 epochs "+0.866 -> +0.866, +0.811 -> +0.818; held-out tiles +0.843 -> +0.861" | as written | .8655 -> .8660; .8106 -> .8177; mean-fold .8428 -> .8610 | OK |
| 63 | N8.7 | "so the frozen recipe is not under-optimised on the never-seen score" vs "the frozen 30-epoch recipe under-trains the plain CNN slightly" (same paragraph; L repeats both) | consistent | batch/lr change hurts (-.008..-.026) but 90 epochs help (+.008..+.024): "not under-optimised" and "under-trains" read as opposites | CONTRADICTION |
| 64 | N8.7 | "sd 0.33 over five seeds"; ConvLSTM "+0.83 to +0.86", MLP "+0.87" | as written | .327; per-fold .833/.832/.863; S4c .868 (mean of folds) | OK |
| 65 | N8.7 | "the lookup table is the only method that stays sensible: +0.66" | only S5 | S5 strict E480 = .661 (mean of folds, OK) but the ConvLSTM scores +0.742 pooled (+0.722 on fold 0 vs S5 +0.509) in the SAME paragraph | CONTRADICTION |
| 66 | N8.8 | side channels "+0.765 (+0.763/+0.780/+0.752) -> +0.822 (+0.822/+0.827/+0.816)"; UK 60 m "+0.854 (+0.860/+0.855/+0.847) -> +0.880 (+0.876/+0.885/+0.880)" | as written | .7653 [.7631,.7805,.7524] -> .8216 [.8224,.8266,.8159]; .8541 [.8603,.8553,.8468] -> .8801 [.8756,.8848,.8798] | OK |
| 67 | N8.8 | "best forced-regime model on the UK (local CNN +0.866); close to local CNN on Mozambique (+0.834)" | as written | clstmf .880 > S4c .872 > local .866 > clstm .854; Moz clstmf .822 vs local .834 (S4c forced .855 is higher) | OK |
| 68 | N8.8 | removing future rain "costs side-channel 0.06-0.08 (E60 .880 -> .803, E480 .822 -> .764), local CNN 0.06-0.13" | as written | .0771 / .0575; local .1313 / .1122 / .0591 | OK |
| 69 | N8.8 | "aborted twice ...; guard disabled +0.885; seed +1 aborted; seed +2 lr 5e-4 +0.865" | as written | REP:L58-59,134-135; runs: noguard f1 .885, s1015 lr 5e-4 .865 | OK |
| 70 | N8.9 | v1 never-seen per fold "+0.184/+0.195/+0.181 (mean +0.187); +0.181/+0.193/+0.189 (+0.188); +0.060/+0.063/+0.062 (+0.062)" | as written | .1840/.1948/.1812 (.1867); .1810/.1931/.1887 (.1876); .0596/.0629/.0619 (.0614: +0.061, diff .0006) | OK |
| 71 | N8.9 | v1 held-out "+0.209/+0.259/+0.178, +0.174/+0.220/+0.184, +0.099/+0.167/+0.056" | as written | identical (RUNS, REP:L131-138) | OK |
| 72 | N8.9 | final models "+0.841/+0.866, +0.830/+0.840, +0.807/+0.811; about 0.65 (UK) and 0.75 (Moz) higher" | as written | .8409/.8655, .8304/.8398, .8070/.8106; gaps .64-.68 and .746-.750 | OK |
| 73 | N8.9 | v1 rain removed "+0.003/+0.003/+0.002"; frozen/scrambled "+0.187 -> +0.187" | as written | .003/.003/.002; scrambled .187 = none .187 | OK |
| 74 | N8.9 | UK v1 aborted "epoch 2-5"; "Mozambique runs ran with the guard on" | as written | REP:L131-133,136-138 epochs 2,2,5,2,2,2; E480 v1 runs have guard True | OK |
| 75 | N8.9 | "reproduce ... the +0.18 to +0.19 reported in the first study" | +0.18..+0.19 | PROJECT_LOG: Australia +0.190, UK +0.189 | OK |
| 76 | N8.10 | seed sd within fold "0.006-0.010 (UK 60 m 0.007, UK 30 m 0.010, Moz 0.006); between folds 0.001-0.008" | as written | .0074 / .0101 / .0058; .0014 / .0027 / .0082 | OK |
| 77 | N8.10 | "+0.025 ... only about 3 seed standard deviations" | ~3 | .0246 / .0074 = 3.3 | OK |
| 78 | N8.10 | held-out sd "0.015 at 60 m, 0.027 at 30 m, 0.33 on Pakistan" | as written | .0147 / .0266 / .3272 | OK |
| 79 | N8.10 | "single seeds range from -0.37 to +0.71" | max +0.71 | E480 held-out single seeds: min -0.372, max +0.773 (fold 2) | ERROR |
| 80 | N8.11 | local CNN learning curve, three rows, and "still rising at 600" | as written | all 15 cells equal REP:L86-97 and my RUNS recomputation | OK |
| 81 | N8.11 | MLP rows, lookup rows, trend "+0.365 -> +0.327" | as written | equal REP:L86-97 / BASE recomputation | OK |
| 82 | N8.11 | "At 600 windows plain CNN (+0.861/+0.850/+0.833) level with MLP (+0.868/+0.836/+0.844)" | as written | differences -.007 / +.014 / -.011 | OK |
| 83 | N8.11 | equal-steps cells "+0.841 -> +0.847, +0.863 -> +0.861; +0.830 -> +0.838, +0.846 -> +0.850; +0.807 -> +0.811, +0.835 -> +0.833" | as written | .8409/.8468/.8633/.8610; .8304/.8376/.8459/.8500; .8070/.8113/.8345/.8330 | OK |
| 84 | N8.11 | "doubling the data changes never-seen skill by only -0.002 to +0.008" | -.002..+.008 | -.0023 .. +.0072 | OK |
| 85 | N8.11 | MLP "(already at its level with 25-50 windows)" | at level | MLP at 25 / 50 windows vs 600: E60 .829/.857 vs .868; E30 .803/.836 vs .836; E480 .809/.807 vs .844 (up to 0.04 below) | OVERCLAIM |
| 86 | N8.11 / L | "the plain CNN is limited by the amount of TRAINING, not by the number of windows" (L: "keeps improving with more windows ONLY because it then trains for more steps") | whole range | equal-steps test exists only for 300 vs 600 windows; from 25 to 300 windows steps and data still vary together (CR2b F1) | OVERCLAIM |
| 87 | N8.12 | UK -> Australia local CNN "+0.738 [+0.571, +0.797] (60 m), +0.662 [+0.420, +0.746] (30 m)"; orderings "+0.702/+0.738/+0.772, +0.637/+0.662/+0.718" | as written | .7380 [.571,.797]; .6622 [.420,.746]; .7021/.7380/.7717; .6368/.6622/.7180 | OK |
| 88 | N8.12 | R60 MLP "+0.659", table "+0.601"; R30 "+0.588", "+0.606" | as written | S4c .659 / .588; S5 .601 / .606 (BASE R*) | OK |
| 89 | N8.12 | "folds are hidden thirds of the START TIMES of the small region's four corner tiles" | 4 tiles each | Mozambique 4 tiles, UK 30 m 4 tiles, but UK 60 m = 2 padded 85x85 squares (V2b check 2 / CR2b) | ERROR |
| 90 | N8.12 | Mozambique -> Pakistan "local CNN -0.884 [-2.665, +0.853], MLP -0.312" | as written | -.8836 [-2.665,.853]; S4c -.312 | OK |
| 91 | N8.12 | "54 of 3,240 window scores carry 93% of the squared error; without them +0.834" | as written | 54/3240, .933, .8336 (V2b note 1: 91-95% per fold) | OK |
| 92 | N8.12 | "the model's error is 4-11 times persistence's" (start-0 windows) | 4-11x | ten worst windows of fold 0: 4.3-10.6x; all 54 start-0 windows: median 7.7x, range 0.3-26.6x | OVERCLAIM |
| 93 | N8.12 | ConvLSTM "+0.444 [+0.277, +0.649]; +0.550 without the start windows"; lookup "+0.549" | as written | .4437 [.277,.649]; .5500; S5 .549 | OK |
| 94 | N8.12 | "On the training region's own hidden time thirds every model scores +0.80 to +0.88" | +0.80..+0.88 | pooled: local CNN .799-.855, ConvLSTM R60 .854, R30 .842, R480 .764; MLP .833-.881 (mean of folds); lookup .639-.741; linear baselines negative to +0.05 | ERROR |
| 95 | N8.12 | ConvLSTM "+0.695 [+0.499, +0.757] (60 m), +0.584 [+0.399, +0.661] (30 m)" | as written | .6949 [.499,.757]; .5841 [.399,.661] | OK |
| 96 | N8.12 | ConvLSTM "slightly below the local CNN" | slight | -0.043 (60 m) and -0.078 (30 m); CNN 30 m .662 vs ConvLSTM .584, i.e. 4-8x the 0.02 resolution | OVERCLAIM |
| 97 | N8.12 | "padding ... moves the R60 baselines by 0.02-0.05" | .02-.05 | CR2b F8: .018 / .034 / .053 / .054 | OK |
| 98 | N8.13 | v2lin "+0.840 [+0.790, +0.893] ... +0.802 [+0.758, +0.865]; held-out +0.770 and +0.582 [+0.242, +0.903]" | as written | .8402 [.790,.893]; .8019 [.758,.865]; .7696; .5818 [.242,.903] | OK |
| 99 | N8.13 | "652,918 parameters, about 150 times the small models"; ablations "frozen +0.006 / -0.015" | as written | logs: 652,918 / 4,741 = 138x (159x the 4,101 CNN); frozen .006 / -.015 | OK |
| 100 | N8.14 | augmentation "0.678 -> 0.803 (capsule), 0.602 -> 0.798 (dense); balanced 0.593 -> 0.697, 0.479 -> 0.660" | as written | SEV rows 3-6 (recomputed) | OK |
| 101 | N8.14 / 8b | gap "+0.076 [+0.051, +0.101] shrinks to +0.005 [-0.014, +0.022]"; causal "+0.056 [+0.029, +0.082]" | as written | SEV lines 14-16 (my recomputation +0.102 / -0.012 / +0.082) | OK |
| 102 | N8.14 | "the original 46-patch-per-fold training" | 46 patches | logs `SEV_*.log`: "62 tiles, held-out 21 (21, 20); train 400 balanced windows (100 per class)" = 41-42 training tiles per fold | ERROR |
| 103 | N8.14 / L | "the capsule design's earlier advantage was a small-sample regularisation effect" | established | only shown: the gap vanishes with augmentation; mechanism not tested | OVERCLAIM |
| 104 | N8.14 | class-changing windows "capsule 22, dense 27, forecast threshold 31" | as written | SEV row hybrid capsule+aug 47/22/31; dense+aug 47/27/30 | OK |
| 105 | N8.14 | rules after augmentation "0.803 -> 0.741 (8 fixed/119 broken), 0.798 -> 0.751 (12/96), recalibrated 0.745/0.754"; reasons "82-84%" | as written | SEV lines 28-33, 40-44 | OK |
| 106 | N8.14 | Grad-CAM "84-92% for all four models; severity map capsule 55-56%, dense 73-75%" | as written | SEV lines 56-63 (depth .88/.92/.92/.84; severity .55/.56/.73/.75) | OK |
| 107 | N8.14 | causal: "0.623 / 0.568; gap +0.056; thresholds 0.974 / 0.986-0.988; rules 0.623 -> 0.575 (0.588), 0.568 -> 0.547 (0.556); reasons 84-85%; Grad-CAM 93-97% / 57-77%" | as written | SEV lines 7-8, 16, 22-25, 39-40, 52-55 | OK |
| 108 | N8.14 | "closes code-review finding F8" | F8 closed | CR1 F8 also covers recalibration fitted on TRUE rates and applied to PREDICTED ones and the reason-truth proxy; N 8b itself says these remain | OVERCLAIM |
| 109 | N8.14 vs L | severity forced regime: N "was not run: there is no use case"; L lists it under "cells dropped after analysis (redundant)" | one reason | two different reasons; queue batch SXF (forced severity) exists and was never run | CONTRADICTION |
| 110 | N8.15 | "fired eight times: six original-model runs (UK 60/30 m, epoch 2-5) and two side-channel runs (epoch 13; retry epoch 6)" | as written | REP:L128-138 / report.json `dead_runs` | OK |
| 111 | N8.15 | "Seven of the eight verified as false alarms ...; original-model runs +0.18 to +0.195; side-channel +0.885" | as written | noguard runs: .181/.193/.189, .184/.195/.181; clstmf f1 .885; the eighth (s1014) not re-run | OK |
| 112 | N8.15 | "the check used the 12 EARLIEST held-out windows (code review finding F13)" | F13 | CR1 F13 first bullet (first 12 indices = earliest starts) | OK |
| 113 | N8.15 | fix "only changes the decision to abort, never the training, and every run that passed the old check is valid" | proven | CR2b: "argued from the code; bit-for-bit re-training comparison was not run"; "runs that passed the old check would also pass the new one: plausible, not proven"; F10: new sampling covers only 84-92% of the held-out set | OVERCLAIM |
| 114 | N8.15 | the word "collapses" | only in the correction | appears once, inside the sentence that retracts it; no misuse in N, L or R | OK |
| 115 | N8b | capsule 67.8% vs dense 60.2%, "+0.076, 95% CI [+0.051, +0.101]" over 1,800 windows | as written | SEV lines 9, 11, 14 | OK |
| 116 | N8b | "last observed frame scores 97.4% and the model's own forecast 98.7%" | as written | SEV: .974; forecast .987 (capsule; dense .986) | OK |
| 117 | N8b | "capsule right in 22 (47%), dense head in 24, forecast-threshold in 34 (72%)" | as written | 47 / 22 / 34 and 47 / 24 / 32 | OK |
| 118 | N8b | rules "161 changed: 25 fixed, 109 broken, 67.8% -> 63.1%; recalibrated 64.0% (51/119); dense 60.2% -> 56.5% / 58.4%" | as written | SEV lines 26-27, 30-31 | OK |
| 119 | N8b | reasons "84% (rising 84%, receding 91%)" | as written | .843 / .841 / .906 | OK |
| 120 | N8b | "The rules' stated depth-trend reasons are true statements about the patch" | true | true in only 84% of windows (sign agreement) | OVERCLAIM |
| 121 | N8b | Grad-CAM "88% / 92% of 40 fixed windows; wet hotter 79%/76%; ratio 3.7/3.8; rank corr 0.52/0.54; severity 55%/73%; 0.17/0.76" | as written | SEV lines 56, 60, 65-68; BUT the deletion shares are per fold over the windows with a non-zero effect (30-40 of 40; `n_with_effect`), mean of 3 folds | OVERCLAIM |
| 122 | L | "Every one of the 14,868 dataset files reads cleanly" | 14,868 files | `data/extracted`: 14,855 .tif + 12 .tfw + 1 .txt = 14,868 (count only; I did not re-read them) | OK |
| 123 | L | rain timing "correlation 1.00 at lag 0, slope 0.494-0.502" (four regions) | as written | rain_lag_*.json: r .9996-.9999; slope .4943 / .4970 / .5007 / .5019 | OK |
| 124 | L | "4,101-parameter local CNN and a 4,741-parameter ConvLSTM" | as written | run logs: 4,101 / 4,741 trainable params | OK |
| 125 | L | "skill +0.81 to +0.87 (UK 60 m +0.84 / +0.87, UK 30 m +0.83 / +0.84, Mozambique +0.81 / +0.81)" | as written | hybrid values .841/.866, .830/.840, .807/.811 | OK |
| 126 | L | "... in all three rainfall regimes" | +0.81..+0.87 in all | local CNN causal/forced .794-.866; ConvLSTM E480 forced .765; ConvLSTM 30 m exists in hybrid only (causal 1/3 folds, forced 0) | OVERCLAIM |
| 127 | L | "seed-to-seed noise of about 0.01" | 0.01 | .006-.010 | OK |
| 128 | L | "lookup table +0.74 to +0.79 and a per-pixel MLP +0.83 to +0.85" | as written | S5 .737-.786; S4c .829-.854 | OK |
| 129 | L | training-time depth-history removal "+0.84 / +0.83 / +0.81 to +0.17 / +0.16 / +0.09" | +0.16 (30 m) | E30 nodepthhist .1546 -> +0.15 | ERROR |
| 130 | L | "H3 holds by 0.67-0.72 against a fold sd of 0.003-0.008" | sd .003-.008 | fold sd (ddof 1, as g_report): E60 .0026, E480 .0076, E30 **.0131** | ERROR |
| 131 | L | "removing rain costs 0.005-0.015 and terrain nothing" | .005-.015 | norain .0060 / .0154 / .0087; noterrain +.0015 / -.0084 / +.0055 (noise) | OK |
| 132 | L | "freezing the history takes the skill to about zero, scrambling makes it strongly negative" | as written | frozen -.02..+.035; scrambled -1.1..-10.6 (never-seen) | OK |
| 133 | L | "ConvLSTM's +0.025 edge ... becomes +0.004 [-0.021, +0.027] ... (Mozambique -0.014 [-0.029, +0.006])" ; "adds nothing measurable" | as written | numbers equal rows 33, 40; "nothing measurable" = nothing beyond +-0.03 | OK |
| 134 | L | "loses +0.01 to +0.025; future rain gains +0.02 to +0.03; H5 holds for the local CNN" | as written | rows 16; local ordering holds in E60, E30, E480 | OK |
| 135 | L | side channels "gains +0.06 (Moz +0.765 -> +0.822) and +0.026 (UK +0.854 -> +0.880)" | as written | +.0563 / +.0260 | OK |
| 136 | L | "The ConvLSTM's forced-rain deficit WAS an input-layout effect" | established | CR1 F4 calls it an unverified hypothesis; N8.8 says "supported" | OVERCLAIM |
| 137 | L | "v1 uses the rain and cannot read the depth history (rain removed +0.003; frozen/scrambled: no change)" | as written | .003 / .003 / .002; frozen = scrambled = none | OK |
| 138 | L | "batch 8 / lr 3e-3: -0.008 to -0.026; 90 epochs +0.008 to +0.024" | as written | -.0082..-.0253; +.0078..+.0243 | OK |
| 139 | L | "frozen recipe is not under-optimised" next to "90 epochs helps the plain CNN" | consistent | same tension as row 63 | CONTRADICTION |
| 140 | L | strict split "all five seeds and the alternative recipe fail: -5.7 to -9.1 ... systematic"; "6 of 480 windows carry 98% ... +0.84 (CNN) and +0.87 (ConvLSTM)" | as written | numbers = rows 51, 53, 56; "systematic" = row 57 (one fold, one architecture) | OVERCLAIM |
| 141 | L | learning curve "per-pixel model at its level with 25-50 windows" | at level | see row 85 | OVERCLAIM |
| 142 | L | "150-fold larger ConvLSTM ... (UK 60 m +0.840, Mozambique +0.802)" | 150x | 138x the ConvLSTM, 159x the CNN; skills .840 / .802 | OK |
| 143 | L | reverse transfers "+0.66 to +0.74 at 60 m and 30 m; -0.88; 93% in 54 of 3,240; +0.83; ConvLSTM +0.44; lookup +0.55" | as written | rows 87, 90, 91, 93 | OK |
| 144 | L | persistence "NSE 0.9995 / CSI(1 mm) 0.983 (Mozambique) and 0.9993 / 0.980 (UK 60 m)" | as written | paper_protocol_persistence_*.json: .99951/.98316; .99935/.98006 | OK |
| 145 | L | severity paragraph: "+0.076 [+0.051, +0.101] (67.8% vs 60.2%); with augmentation both about 80%, +0.005 [-0.014, +0.022]; 97.4%; 98.6-98.7%; 0.678 -> 0.631; 0.803 -> 0.741; 84%; 84-92%; causal 0.623 / 0.568 / +0.056 [+0.029, +0.082]; 0.623 -> 0.575" | as written | SEV (all cells) | OK |
| 146 | L | "two independent train -> test pairs, not sixteen results" | 16 | 16 complete headline neural cells exist (18 minus E30 ConvLSTM causal/forced) | OK |
| 147 | L | "on the 47 windows where the class changes they are right in 22-27 against 31-34 for thresholding the model's own forecast" | 31-34 | hybrid rows: models 22 / 22 / 24 / 27; forecast threshold 34 / 31 / 32 / **30** | ERROR |
| 148 | L | "Deviations from the pre-registration (disclosed)" list | complete | omits the equal-steps learning-curve cells (added after CR2b F1) and the seed +1 / seed +2 (lr 5e-4) side-channel retries | OVERCLAIM |
| 149 | L | "The result files carry five different engine hashes" | five | six today: `6627af098c` 136, `72eeb40485` 82, `e3e8e93d72` 30, `330a342f7e` 27 (runs written 06:27-07:38 after engine.py was edited 05:48), `923283620c` 23, `fb4ce0ca91` 12 | ERROR |
| 150 | L | "seed repeats made with the current code sit within one seed standard deviation of the frozen runs" | within 1 sd | repeat minus frozen: E60 f0 seed 13 +.019 (2.5 sd); E30 f2 all four repeats +.016..+.027 (mean +.022 = 2.2 sd); 8 of 9 fold means within -.008..+.010 | OVERCLAIM |
| 151 | L | same-seed rerun "9e-6 (+0.83794 vs +0.83793) and 5e-5" | as written | repro_check.json: 8.54e-6; 4.61e-5; .83794 / .83793 | OK |
| 152 | L | "One E30 ConvLSTM causal fold-0 run ... the report's data-warning section lists it" | listed | REPORT.md on disk lists four E60 groups and NOT E30 clstm/causal (section 0) | CONTRADICTION |
| 153 | L | "two Australia 30 m tiles carrying at most 0.43 m of water are excluded" | 0.43 m | V2b note 6: max depth 0.43 m | OK |
| 154 | R5 | "a safety check ... raised eight false alarms (all original-model or side-channel runs trained normally when re-run without it)" | 8 of 8 | seven of the eight were re-run; the eighth (retry seed 1014) was not (N8.15, REPORT:L140) | CONTRADICTION |
| 155 | R3 | "ConvLSTM minus plain CNN: +0.004 [-0.021, +0.027] ... -0.014 [-0.029, +0.006] when both train 90 epochs"; "adds nothing measurable" | as written | rows 40-41 | OK |
| 156 | R3 | "the original model +0.05 .. +0.2"; "the final CNN-LSTM, 4,741 parameters"; "all reach about +0.8" | as written | rows 10, 124; lookup .737-.786, MLP .829-.854, CNN .807-.841, ConvLSTM .811-.866 | OK |
| 157 | R2 | "A per-pixel linear model of the depth history beat it by about 2x" | ~2x | S1 / v1 = 1.73 (60 m), 1.83 (30 m), 2.5 (480 m) | OK |
| 158 | R1 | "rain file j falls DURING the 30 minutes after 1800 j (measured in all four regions)" | as written | rain_lag_*.json, lag 0 unique peak in all four | OK |
| 159 | R | "Independent re-computation from the raw files (all numbers matched)" | all numbers | V2b "Could NOT verify": every neural forecast error (sse_m), model inputs other than depth, S1-S4 and reverse baselines, severity, most bootstrap intervals; first verification covered rain timing, persistence RMSE, pool integrity, paper-protocol persistence, lookup inputs | OVERCLAIM |
| 160 | R | "adversarial code review (no critical finding; five major findings fixed)" | as written | CR1: 0 critical, 5 major (all addressed); CR2b (3 major, 14 minor) not mentioned, its majors F2/F3 were fixed in code (dagger marks, `count_running`) | OK |
| 161 | R | "python completion/run_queue.py A BL K S E B FX C OPT ... the whole study (about 12 h on a 12-core CPU)" | whole study | omits batches D, E2, LC, LCB, LC2, EP, EPC, EXS, RB, RN, SX, CG, FXG, FXR, FXR2 (v2lin, learning curves, 90-epoch, strict-split seeds, reverse transfers, causal severity, guard re-runs); all 310 runs = 51.6 CPU-h = 12.9 h wall at 4 workers | OVERCLAIM |
| 162 | R | literal placeholder "@@ROUND2_RESULTS@@" (line 25) | resolved | still unresolved | ERROR |
| 163 | R | reproduce commands (`engine.py cache/selftest/run`, `--epochs 30 --n-train 300`, `b_baselines.py E60 hybrid 0 --n-train 300`, `g_report.py`, `j_severity_report.py`) | valid | subcommands and flags exist in engine.py:559-576, b_baselines.py docstring | OK |

Row count: 163 statements checked (several rows bundle values that share one source; each value inside them was compared). Verdict counts: OK 120, ERROR 15, OVERCLAIM 21, CONTRADICTION 7 (43 non-OK). The word "collapse" is used correctly (never for the false-alarm runs).

Most important problems, in order: (1) REPORT.md/report.json on disk are degraded and cannot back any headline number (section 0); (2) row 52, the 14,853 / 780 squared-error pair is one window, not the fold's start-0 windows; (3) rows 57, 65 and 63/139, the Pakistan-failure story is over- and self-contradicted (systematic "property of the data" although the ConvLSTM survives; "lookup table is the only method that stays sensible" although the ConvLSTM scores +0.742; "not under-optimised" versus "under-trains"); (4) row 154, README says all eight aborts were re-run and trained normally, only seven were; (5) rows 47, 79, 94, 102, 129, 130, 147, small numeric errors (S2 "+0.26", seed maximum "+0.71", "every model +0.80 to +0.88", "46 patches", E30 "+0.16" and "fold sd 0.003-0.008", "31-34").

## 2. Every non-OK item with an exact suggested replacement

**ERROR**

* Row 2 (N8.1). Replace "linear with rain/state/trend +0.35" with "linear with rain/state/trend +0.34".
* Row 6 (N8.1). Replace "small ConvLSTM **+0.87 [+0.81, +0.92]**" with "small ConvLSTM **+0.87 [+0.81, +0.91]**". (Interval is [+0.808, +0.9146]; REPORT prints +0.915, the narrative rounded twice.)
* Row 7 (N8.1). Replace "At 30 m: +0.35 / +0.36 / +0.76 / +0.85 / +0.83 / +0.84." with "At 30 m: +0.34 / +0.36 / +0.76 / +0.85 / +0.83 / +0.84."
* Row 38 (N8.5). Replace "(report reads \"not separable from no difference\")" with "(the report marks them \"clstm larger (by the rule, but the interval includes 0)\")".
* Row 47 (N8.6). Replace "(a model with rain and state but no trend reaches only about +0.26)" with "(a linear model with rain and state but no depth trend reaches only +0.21 / +0.19 / +0.10 on the never-seen region for UK 60 m / UK 30 m / Mozambique)".
* Row 52 (N8.7). Replace "(14,853 against 780 for persistence in that fold)" with "(in that fold the six start-0 windows carry a squared error of 33,901 against 1,262 for persistence, out of 34,294 against 3,769 for the whole fold; the single worst window alone is 14,853 against 780)". Same fix applies to `narr_8_7_on.md`, which repeats the wrong sentence.
* Row 79 (N8.10). Replace "single seeds range from -0.37 to +0.71" with "single seeds range from -0.37 to +0.77".
* Row 89 (N8.12). Replace "the small region's four corner tiles" with "the small region's corner tiles (four for Mozambique and for UK 30 m, two padded 85 x 85 squares for UK 60 m)".
* Row 94 (N8.12). Replace "On the training region's own hidden time thirds every model scores +0.80 to +0.88." with "On the training region's own hidden time thirds the local CNN scores +0.80 to +0.85 (pooled), the per-pixel MLP +0.83 to +0.88 and the ConvLSTM +0.76 to +0.85 (pooled; +0.82 as the mean of the Mozambique folds); the lookup table (+0.64 to +0.74) and the linear baselines (negative to +0.05) are lower."
* Row 102 (N8.14). Replace "the original 46-patch-per-fold training and no augmentation" with "the original training (400 class-balanced windows per fold, drawn from the 41-42 tiles that are not held out, no augmentation)".
* Row 129 (L). Replace "to +0.17 / +0.16 / +0.09" with "to +0.17 / +0.15 / +0.09".
* Row 130 (L). Replace "against a fold sd of 0.003-0.008" with "against a fold sd of 0.003-0.013".
* Row 147 (L). Replace "against 31-34 for thresholding the model's own forecast" with "against 30-34 for thresholding the model's own forecast".
* Row 149 (L). Replace "The result files carry five different engine hashes" with "The result files carry six different engine hashes (136 / 82 / 30 / 27 / 23 / 12 runs; the newest 27 were written after `engine.py` was edited on 21 September 05:48)". Also re-check "the dead-network check was changed twice" (see section 3).
* Row 162 (R). Substitute the results section for `@@ROUND2_RESULTS@@` (or delete the line) before publishing.

**OVERCLAIM**

* Row 19 (N8.2). Replace "The code review (F4) traced this to the input layout" with "The code review (F4) suspected the input layout (its own words: \"cause is my hypothesis, unverified\"); 8.8 tests it".
* Row 25 (N8.3). Replace "rainfall is worth 0.01-0.03" with "rainfall is worth 0.01-0.02 on the never-seen region (0.02-0.07 on held-out tiles)".
* Row 42 (N8.5). Replace "it reaches the same skill" with "its skill is equivalent within the pre-registered margin (difference inside +-0.03)".
* Row 57 (N8.7). Replace "**The failure is systematic - a property of the data (the onset is absent from training), not of one unlucky run.**" with "**The failure of the plain CNN reproduces across five seeds and a second recipe on this fold (-5.7 to -9.1), so it is not one unlucky run; that the cause is the missing onset is an inference: the ConvLSTM on the same fold scores +0.722, the lookup table +0.509, and the plain CNN's other two folds are +0.848 and +0.738.**"
* Row 85 (N8.11). Replace "(already at its level with 25-50 windows)" with "(within about 0.04 of its 600-window score already with 25-50 windows)".
* Row 86 (N8.11). Replace "the plain CNN is limited by the amount of TRAINING, not by the number of windows" with "between 300 and 600 windows the plain CNN is limited by the amount of TRAINING, not by the number of windows (the equal-steps test was not made at 25-100 windows)". In L replace "keeps improving with more windows only because it then trains for more steps (with the steps held equal, doubling the data changes its skill by -0.002 to +0.008), i.e. it is limited by training length, not by data" with "keeps improving with more windows mainly because it then trains for more steps (between 300 and 600 windows, with the steps held equal, doubling the data changes its skill by -0.002 to +0.007)".
* Row 92 (N8.12). Replace "the model's error is 4-11 times persistence's" with "the model's squared error in the ten worst windows of fold 0 is 4-11 times persistence's (over all 54 start-0 windows the ratio has median 7.7 and ranges from 0.3 to 27)".
* Row 96 (N8.12). Replace "slightly below the local CNN" with "below the local CNN (by 0.04 at 60 m and 0.08 at 30 m)".
* Row 103 (N8.14 and L). Replace "the capsule design's earlier advantage was a small-sample regularisation effect, not evidence that capsules suit the task" with "the capsule design's earlier advantage disappears with augmentation, which is consistent with a small-sample regularisation effect (not tested directly) and gives no evidence that capsules suit the task". In L replace "so the capsule's advantage was a small-sample effect" with "so the capsule's advantage is consistent with a small-sample effect".
* Row 108 (N8.14). Replace "closes code-review finding F8" with "addresses the rain part of code-review finding F8 (the recalibration is still fitted on true rates and applied to predicted ones)".
* Row 113 (N8.15). Replace "it only changes the decision to abort, never the training, and every run that passed the old check is valid" with "it only changes the decision to abort (argued from the code; a bit-for-bit re-training comparison was not run), so runs that finished under the old check are unaffected by the change; whether they would also pass the new check was not tested (the 12 windows are taken at a stride and cover 84-92% of the held-out set)".
* Row 120 (N8b). Replace "The rules' stated depth-trend reasons are true statements about the patch: the predicted trend has the same sign as the actual one in 84% of windows" with "The rules' stated depth-trend reasons are true in about 84% of windows (the predicted trend has the same sign as the actual one; 'rising' claims 84%, 'receding' 91%)".
* Row 121 (N8b). Replace "in 88% (capsule model) and 92% (dense model) of 40 fixed held-out windows" with "in 88% (capsule model) and 92% (dense model) of the windows in which the deletion had any effect (30-40 of the 40 fixed windows per fold; mean over the 3 folds)".
* Row 126 (L). Replace "in all three rainfall regimes" with "in all three rainfall regimes for the local CNN (+0.79 to +0.87) and for the ConvLSTM on UK 60 m and Mozambique (+0.77 to +0.87); the 30 m ConvLSTM was run in the hybrid regime only".
* Row 136 (L). Replace "The ConvLSTM's forced-rain deficit was an input-layout effect" with "The ConvLSTM's forced-rain deficit is supported as an input-layout effect".
* Row 140 (L). Replace "its failure under the strict space-and-time split on Pakistan is systematic (all five seeds and the alternative recipe fail: -5.7 to -9.1)" with "its failure under the strict space-and-time split on Pakistan (fold 0) reproduces across seeds and recipes (all five seeds and the alternative recipe fail: -5.7 to -9.1)"; and in "Not established" replace "The Pakistan failure is systematic across seeds and recipes." with "The Pakistan failure of the plain CNN reproduces across seeds and recipes on fold 0."
* Row 141 (L). Same replacement as row 85.
* Row 148 (L). Append to the deviations bullet: "the equal-steps learning-curve cells (n = 600 for 15 epochs, n = 300 for 60 epochs) and the two post-hoc retries of the side-channel fold 1 (seed +1; seed +2 with lr 5e-4) were also added after seeing results".
* Row 150 (L). Replace "(seed repeats made with the current code sit within one seed standard deviation of the frozen runs)" with "(seed repeats made with the current code scatter around the frozen runs about as much as the seeds scatter among themselves: the mean of the four repeats minus the frozen run is between -0.008 and +0.010 in 8 of 9 folds and +0.022 in E30 fold 2; single repeats differ by up to 2.6 seed standard deviations)".
* Row 159 (R). Replace "Independent re-computation from the raw files (all numbers matched)" with "Independent re-computation from the raw files of the data-side numbers (rain timing, window pools and persistence errors, the lookup-table baseline, the train/hold-out splits and the arithmetic of every stored skill) matched; the neural networks' own forecast errors were not recomputed".
* Row 161 (R). Replace "the whole study (about 12 h on a 12-core CPU)" with "the main matrix (the exploratory batches D E2 LC LCB LC2 EP EPC EXS RB RN SX CG FXG FXR FXR2 AX BX FXX OPTX F B2 EX S2 are listed in completion/run_queue.py; all 310 runs take about 51 CPU-hours, about 13 h on 4 workers)".

**CONTRADICTION**

* Row 61 (N8.7). Replace "and its Pakistan held-out score from +0.383 to +0.678" with "and its Pakistan held-out score from +0.430 to +0.699 pooled (+0.383 to +0.678 as the mean of the three fold skills)". Also make the aggregator explicit wherever the held-out column mixes pooled neural numbers with fold-mean baselines (MLP +0.87 in row 32; CR2b F6).
* Rows 63 and 139 (N8.7 and L). Replace "so the frozen recipe is not under-optimised on the never-seen score" with "so a larger batch and a higher learning rate do not improve the never-seen score (the frozen recipe is not limited by its step size), but the frozen 30-epoch recipe does under-train the plain CNN in number of steps (see Training length)"; in L replace "The frozen recipe is not under-optimised for the never-seen score (batch 8 / lr 3e-3: -0.008 to -0.026); 90 instead of 30 epochs helps the plain CNN by +0.008 to +0.024." with "A larger batch and learning rate (batch 8 / lr 3e-3) do not help the never-seen score (-0.008 to -0.026), but the plain CNN is under-trained at 30 epochs: 90 epochs give +0.008 to +0.024."
* Row 65 (N8.7). Replace "(the lookup table is the only method that stays sensible: +0.66)" with "(under the strict split the lookup table stays at +0.66 and the ConvLSTM at +0.742; the per-pixel MLP (-2.65, mean of folds) and the plain CNN (-6.56) do not)".
* Row 109 (N8.14 vs L). Use one reason in both, e.g. in L: "and severity under the forced-rain regime (not run: there is no use case for a known-future-rain severity label)"; also remove batch SXF from the "dropped as redundant" wording.
* Row 152 (L). Either regenerate REPORT.md so its data-warning section lists E30 clstm/causal (fold 0 only), or replace "(the report's data-warning section lists it)" with "(it is listed in `summary.md` as 1/3 folds, incomplete)". Check first: `summary.md` is from 2026-09-20 13:26 and may not list it either.
* Row 154 (R5). Replace "raised eight false alarms (all original-model or side-channel runs trained normally when re-run without it)" with "aborted eight runs, seven of which (six original-model runs and one side-channel run) trained normally when re-run without it; the eighth, a retry of the side-channel run with the next seed, was not re-run". Also replace "it now samples windows across the whole test period" with "it now samples windows across the whole held-out set".

## 3. What I could NOT verify

* **Neural forecast errors.** No one (me included) recomputed any `sse_m`; every neural skill in N/L/R is matched to the stored JSONs only (V2b "Could NOT verify"). I checked arithmetic, aggregation, intervals and splits, not the forward passes.
* **REPORT.md headline cells** (section 0): unavailable; matched against recomputation. `report.json` has no headline data either. H1, H2, H5 counts and the H3/H4/H6 verdict lines could not be compared to REPORT.md; I re-derived H3, H4, H6 by the pre-registered rules (PREREGISTRATION lines 35-38): consistent with N/L.
* **Paper facts**: FNO+ NSE/CSI values (0.955 / 0.934; 0.9989 / 0.860), "the paper reports no persistence baseline", "the paper's benchmark has no LSTM model", "the paper's RMSE is a relative error" and R4's "copying the initial frame matches or beats its best model on most of its own measures" (only our persistence numbers, which match `paper_protocol_persistence_*.json`, were checkable).
* **Geometry, georeferencing, crop offsets, cross-hardware, Colab** statements in L; "reads cleanly" (only the file count 14,868 = 14,855 .tif + 12 .tfw + 1 .txt was confirmed).
* **Pre-registration timing** (which analyses were post hoc) and the sentence "the dead-network check was changed twice" (CR2b F16 lists five engine versions; I found a sixth today; what changed at 05:48 was not diffed).
* **"With the original weak model the net effect had been zero on 240 windows"** (N8b): first-study number, not in these result files.
* **"Mozambique 1 valid 128-px tile, UK 30 m 2, UK 60 m none"** (N8.12): UK 60 m and UK 30 m agree with CR1 F13, the Mozambique count was not checked.
* **v1 architecture statements** were checked only for the 5 m depth scale (`model.py:40`) and the 128-unit bottleneck (`model.py:84`); "cannot read the depth history" rests on the ablations (verified).
* **Severity intervals**: reproduced only to Monte-Carlo noise (+0.102 vs +0.101; -0.012 vs -0.014 lower bound). They come from a plain window bootstrap that treats the 1,800 pooled windows as independent (no tile/time clustering, unlike the depth intervals); if the depth intervals needed block bootstraps (CR1 F2), these are probably too narrow. Not audited further.
* **Run-time claim** "about 12 h on a 12-core CPU": only summed `train_minutes` (3,096 min = 12.9 h on 4 workers); the README's batch list is not the full queue.
* **"Seven guard-disabled runs would have passed the corrected guard"** and bit-for-bit invariance of training under the guard edit: not testable without re-training (CR2b "Could not verify").
* **Other documents** (`results/narr_8_7_on.md`, `explainer_*.md`) repeat some of the same wrong sentences (for example the 14,853 / 780 sentence); I audited only the three requested files.
