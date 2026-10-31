# Document audit B: Story for Aman (D), Explainer v2 (E), deck text (F)

Auditor: independent read-only pass, 21 September 2026. Nothing under `C:\Dev\flood-forecast-demo` was modified except this file
(the report generators were executed in memory with every file write redirected to the session scratchpad; the bytecode caches in `completion/__pycache__` were already current and were not rewritten).
One light python process at a time, no TensorFlow, nothing killed.

Documents (paths as given):
* D = `.../DL Project/Project_Story_for_Aman.md` (sections 1, 4.3, 5, 6, 7 checked closely; the other sections skimmed for numbers).
* E = `.../DL Project/Project_Explainer_End_to_End_v2.md` ("What changed" table, section 10 incl. 10.1-10.15 and the severity narrative, section 12; section 14 and cross-references also read).
* F = `.../scratchpad/build/deck_v6_final_text.txt` (19 slides; slides 11-18 are audit/results/lessons; counted by the `=====` separators).

## 0. How the numbers were checked, and one important finding about the "source of truth"

**FINDING 0 (affects D section 9 and F slide 15): the on-disk `completion/results/REPORT.md` and `report.json` (both mtime 08:02) are degraded.**
Every headline table cell is "-", H1 and H2 read "0/0 cells pass", the ablation, seed-repeat, strict-split, start-window and reverse-transfer sections are empty or absent, `report.json` has `headline: {}` and `baselines: {}`, and the data-warnings list claims only fold 0 exists for the E60 groups, although all three folds are in `runs/`. Only the learning-curve, equal-steps, 90-epoch and guard-abort tables are intact.
To get a usable reference I ran `g_report.main()` in memory against the CURRENT `runs/` and `baselines_*.json` with all file writes redirected to the scratchpad (`regen_REPORT.md`, `regen_report.json`).
That full regenerated report contains 203 table/bullet lines and **all 203 are found verbatim in E section 10**. So the tables in E are consistent with the run JSONs; what is broken is the on-disk report that D and F point readers to.
Fix: run `python completion/g_report.py` once (all runs are present) and check that H1 then reads 13/16.

Sources used (abbreviations in the table):
* **REG** = the regenerated full report above and the run JSONs it reads. I also recomputed directly from run JSONs: per-fold never-seen skills, pooled held-out skills, the strict-split windows, seed ranges, per-lead persistence RMSE, engine hashes, train minutes, window counts.
* **BL** = `baselines_*.json` (including S2 and the strict-split `_blk` files, which REPORT.md does not print).
* **SEV** = `SEVERITY_REPORT.md`. I recomputed natural and balanced accuracy, last-frame and forecast thresholds, class-changing counts and the three paired differences from `severity/SEV_*.json`: identical. (Rule-layer, Grad-CAM and bootstrap-interval tables were not recomputed; `scipy` is not installed in this python.)
* **PP** = `paper_protocol_persistence_*.json`. **LOG** = `PROJECT_LOG.md`. **STATE** = `completion/STATE.md`. **VER1 / VER2** = `review/independent_verification.md` / `review/verification_round2b.md`. **CR1 / CR2** = `review/code_review.md` / `review/code_review_round2b.md`.
* Tolerance: 0.001 for 3-decimal claims; for 2-decimal claims correct rounding of the exact value.

Verdict key: OK; ERROR (number or fact is wrong); OVERCLAIM (stronger than the evidence, wrong scope, or an unsupported causal wording); CONTRADICTION (documents disagree with each other or with themselves).

Leftover markers: no `[PENDING`, TODO, TBD, XXX or FIXME in D, E or F. The word "collapse" appears only in disclosed corrections (D 4.3 quotes it as the mistake; E 10.15 says so) and for genuine early events that LOG documents (D s3 capsule softmax; E s14 "latent training-collapse mode", the 60 m run). No run that REPORT.md lists as a guard false alarm is called a collapse anywhere.

**Counts (180 statements checked; table in section 1).**
* Overall: OK 131, OVERCLAIM 27, ERROR 18, CONTRADICTION 4 (49 non-OK items, each with replacement text in section 2).
* By document: D (Story) 68 checked = OK 51, OVERCLAIM 11, ERROR 5, CONTRADICTION 1. E (Explainer) 73 checked = OK 49, ERROR 11, OVERCLAIM 11, CONTRADICTION 2. F (deck text) 39 checked = OK 31, ERROR 2, OVERCLAIM 5, CONTRADICTION 1.
* Twelve cross-document / within-document contradictions are listed separately in section 3 (several overlap with the table rows above).

**The five most important problems**
1. The on-disk `REPORT.md` / `report.json` (08:02) are degraded (empty headline tables, "0/0" hypotheses) while D s9 and F s15 tell readers to look there; regenerating from the current runs reproduces E's tables exactly, so only a re-run of `g_report.py` is needed (N01).
2. E 10.7 contains numbers that contradict E's own table: strict-split seed results "-6.9, -9.1, -10.1, -10.9 (frozen -10.1)" should be -6.9, -9.1, -8.1, -8.9 (frozen -8.1), and "14,853 against 780" cannot be reproduced (fold 0: 33,901 vs 1,262 in the six onset windows) (N02).
3. "Independent agents re-derived the numbers" is overstated in F s11 ("Every headline number ... all matched") and D s1/s4.3/s5: the verifier explicitly could not check any neural skill, only raw-data quantities, the lookup baseline and the splits (N03).
4. Claims that disagree with the record or with each other: D s8 "GPU and CPU differ only in the last decimals" (LOG: +-0.03; E s12: not established), D 6.8 and E 10.7 "only the lookup table stays sensible" (the ConvLSTM also does: +0.44 reverse, +0.74 strict), E 10.12 "every model scores +0.80 to +0.88" on the time thirds (ConvLSTM +0.76, lookup +0.64, linear negative) (N04, N05, N06).
5. Smaller numeric slips in E s12/10.x: H3 fold sd "0.003-0.008" (E30 is 0.013), "+0.26" for the no-trend linear model (never-seen is +0.21/+0.19/+0.10), "+0.71" seed maximum (is +0.77), "five engine hashes" (six), forecast-threshold "31-34" (30-34); plus the undisclosed design-phase caveat F10 (D "kept out of every design decision") and the F s9 "13 mm at 30 min" (N07-N10, N13, N15, N18).

## 1. Statement-by-statement table

| ID | Doc / place | Quote (abridged) | Claimed | Actual (source) | Verdict |
|---|---|---|---|---|---|
| D01 | D s1 | "+0.19 (roughly 10% less error than copying)" | skill 0.19 = 10% less error | 1 - sqrt(1-0.19) = 10.0% RMSE; first-study skill about +0.19 (LOG) | OK |
| D02 | D s1, 4.1, 9 | "13.5 of 26.9 GB" (half the dataset) | 13.5 / 26.9 | 13.47 / 26.87 GB (LOG, E s3) | OK |
| D03 | D s1, 6.1 | lookup table "about 90% of the best score" | ~90% | 0.786/0.866 = 91% (E60); 0.763/0.854 = 89% (E30, vs MLP); 0.737/0.829 = 89% (E480, vs MLP) (REG) | OK |
| D04 | D s1 | ConvLSTM and plain CNN "+0.81 to +0.87 ... original +0.06 to +0.19" | 0.81-0.87; 0.06-0.19 | hybrid: local 0.807/0.830/0.841, ConvLSTM 0.811/0.840/0.866, v1 0.061/0.187/0.188 (REG); "hybrid" not stated (other regimes go down to 0.765) | OK |
| D05 | D s1 | "capsule head only beat its plain control because of the small training set" | causal claim | no-aug gap +0.076 [+0.051, +0.101]; with aug +0.005 [-0.014, +0.022] (SEV); one seed per fold; aug not run in the causal regime; consistent with, not proof of, a small-sample effect | OVERCLAIM |
| D06 | D s1 | "both lose to a one-line rule" | | last-frame threshold 0.974 vs capsule 0.678/0.803, dense 0.602/0.798 (SEV) | OK |
| D07 | D s1, 6.7 | "the rulebook makes answers worse" | | 0.678->0.631, 0.803->0.741, 0.602->0.565, 0.798->0.751 (SEV) | OK |
| D08 | D s1, 5 | "separate computer agents (they re-derived the numbers from raw files ...)"; s5 "re-derived the key numbers ... (all matched)" | numbers re-derived | VER1/VER2: persistence errors, pools, splits, lookup baseline matched; "Could NOT verify: model errors sse_m and everything derived from them (every neural skill ...)" | OVERCLAIM |
| D09 | D s3 | "depth skill about +0.19 (seven runs ranged +0.04 to +0.23)" | UK range +0.04..+0.23 | UK values in LOG: +0.044 (seed 7 aug) ... +0.217 (audit run); +0.225 is an Australian score | ERROR |
| D10 | D s3 | "capsule head scored 55-70% ... trivial rule about 93%" | | LOG: "roughly 55-70%"; threshold baseline 92-93% | OK |
| D11 | D s4.2(1) | rain leak "worth only about +0.01 to +0.025" | | local CNN causal->hybrid +0.019 / +0.025 / +0.013 (REG) | OK |
| D12 | D s4.2(2) | straight-line guess "about twice" the original | ~2x | S1/v1 = 0.323/0.187 = 1.7x, 0.345/0.188 = 1.8x, 0.151/0.061 = 2.5x (BL, REG) | OK |
| D13 | D s4.2(2) | original "+0.18 to +0.19 on the UK, +0.06 on Mozambique" | | folds 0.181-0.195; Mozambique 0.0596-0.0629 (REG) | OK |
| D14 | D s4.2(4) | "NSE 0.9995 vs 0.955; CSI at 1 mm 0.98 vs 0.93" | | copy NSE 0.99951, CSI 0.9832 (PP); FNO+ values are the paper's (F s17 prints 0.9555 and 0.934) | OK |
| D15 | D s4.3 | "stopped eight runs ... (+0.18 to +0.195) ... (+0.885)" | | 8 aborts; six v1 re-runs 0.181-0.195; clstmf re-run 0.8848 (REG guard table) | OK |
| D16 | D s4.3 | "(everything matched: about 190,000 window scores, the lookup-table baseline to 1e-9, no training/test overlap in any of 56 splits)" | | 191,414 windows in 284 files; S5 differences 4e-10 to 6e-9 in 5 cells; 56 blocked/reverse files, 0 shared frames (VER2). Numbers right; scope covered by D08 | OK |
| D17 | D s4.3 | second review "no critical error; three major points, all fixed" | | CR2: 0 critical, 3 major (learning curve, guard runs unmarked, runner) | OK |
| D18 | D s4.3 | "and so did the side-channel run it had stopped (+0.885)" | the side-channel run | two side-channel runs were aborted; only the first was re-run (REPORT guard table: retry "not re-run") | OVERCLAIM |
| D19 | D s4.3 | guard "looked at the 12 earliest test windows ... samples windows across the whole test period" | test | the guard used the earliest HELD-OUT windows (tiles of the training flood), not the never-seen test set (E 10.15, CR1 F13) | ERROR |
| D20 | D s4.3 | laptop hung, "five training jobs instead of four ... 15.5 GB" | | STATE 19:57 reboot note; this machine 15.47 GB | OK |
| D21 | D s5 | "the never-seen floods were kept out of every design decision" | absolute | CR1 F10: four non-blind design runs (v2 family) printed their UK skill (0.758-0.828) in `runs_design/`; "not consulted"; E60 fold-0 held-out windows reused | OVERCLAIM |
| D22 | D s5 | "4,101 parameters ... 4,741 parameters ... 653k parameters" | | 4,101 and 652,918 in training logs; 4,741 only in README/PREREG/build script (not in any run JSON or log) | OK |
| D23 | D s5, 6.9 | "five different random seeds per fold" | | seeds 1003/1013/1023 + 11-14 present for local hybrid, all folds, E60/E30/E480 | OK |
| D24 | D s5 | protocol saved "with a checksum" | | md5 2b8b2ef69639 recorded (LOG) = md5 of the file now; file 10:12:34, first final-run file 10:19 | OK |
| D25 | D s6 table | straight-line +0.32 / +0.35 / +0.15 | | 0.3232 / 0.3447 / 0.1509 (BL); +0.35 is a double rounding, exact +0.34 | OK |
| D26 | D s6 table | v1 +0.19 / +0.19 / +0.06 | | 0.1867 / 0.1876 / 0.0614 | OK |
| D27 | D s6 table | lookup +0.79 / +0.76 / +0.74 | | 0.7857 / 0.7629 / 0.7370 | OK |
| D28 | D s6 table | per-pixel MLP +0.85 / +0.85 / +0.83 | | 0.8464 / 0.8540 / 0.8288 | OK |
| D29 | D s6 table | local CNN +0.84 / +0.83 / +0.81 | | 0.8409 / 0.8304 / 0.8070 | OK |
| D30 | D s6 table | small ConvLSTM +0.87 / +0.84 / +0.81 | | 0.8655 / 0.8398 / 0.8106 | OK |
| D31 | D 6.2 | "Freezing the history takes the skill to about zero" | | -0.018 ... +0.037 across all non-v1 models (REG ablation table) | OK |
| D32 | D 6.2 | "scrambling its order makes it strongly negative" | all models | -1.14 to -7.83 for local, kernel 11, ConvLSTM, heavy; per-pixel CNN on the UK stays +0.163 / +0.121; v1 unchanged | OVERCLAIM |
| D33 | D 6.2 | "removing rain costs only 0.01-0.03" | | test-time 0.008-0.020 (local, ConvLSTM), up to 0.039 (11x11 CNN, E30) (REG); wording differs from E table (C1) | OK |
| D34 | D 6.2 | training-time: full +0.84/+0.81; no depth history +0.17/+0.09; no rain history +0.84/+0.80; no rain +0.84/+0.80; no terrain +0.84/+0.81 | | 0.841/0.807; 0.172/0.086; 0.839/0.802; 0.835/0.798; 0.842/0.812 (REG) | OK |
| D35 | D 6.2 | "depth history worth about 0.7; rain about 0.01; terrain nothing" | | 0.669 / 0.721 (H3); rain 0.006 / 0.009; removing terrain changes skill by +0.001 / -0.008 / +0.005 | OK |
| D36 | D 6.3 | original: freeze or scramble no change; "remove the rain: skill drops to zero" | | v1 frozen = scrambled = 0.187; rain removed +0.003 / +0.003 / +0.002 (REG) | OK |
| D37 | D 6.4 | "a per-pixel network, a 5x5 and an 11x11 CNN score the same" | same | E480: 0.818 / 0.807 / 0.793 (11x11 is 0.025 below, seed sd 0.006); UK 0.848/0.841/0.841 fine | OVERCLAIM |
| D38 | D 6.4 | "Trained long enough (90 epochs) the plain CNN is level with the per-pixel network" | level | 90-epoch local 0.862/0.838/0.831 vs MLP (30 ep) 0.846/0.854/0.829: +0.016 / -0.016 / +0.002 | OK |
| D39 | D 6.5 | "+0.025 ... +0.004 (UK) and -0.014 (Mozambique)" | | 0.0250 [+0.004,+0.043]; +0.0042 [-0.021,+0.027]; -0.0136 [-0.029,+0.006] (REG H4, epochs90) | OK |
| D40 | D 6.5 | "the ConvLSTM matches the plain CNN" | | 90-epoch intervals inside +-0.03 (pre-registered equivalence rule) | OK |
| D41 | D 6.6 | past rain -> original input "+0.01 to +0.025; future rain +0.02 to +0.03 more" | general | local CNN 0.013-0.025 and 0.021-0.027; ConvLSTM causal->hybrid +0.013 (UK 60) and +0.000 (Mozambique) | OVERCLAIM |
| D42 | D 6.6 | side channels "+0.77 to +0.82 (Mozambique) and +0.85 to +0.88 (UK)" | | 0.7653->0.8216; 0.8541->0.8801 | OK |
| D43 | D 6.7 | "67.8% ... 60.2% (capsule better by 7.6 points)" | | 0.678 / 0.602 / +0.076 [+0.051,+0.101] (SEV) | OK |
| D44 | D 6.7 | "both reach about 80% ... +0.5 points, interval includes 0" | | 0.803 / 0.798; +0.005 [-0.014,+0.022] (SEV) | OK |
| D45 | D 6.7 | "one-line rule (97.4%)" | | 0.974 (SEV, recomputed) | OK |
| D46 | D 6.7 | "67.8% -> 63.1%; 80.3% -> 74.1%" | | 0.678->0.631; 0.803->0.741 (SEV) | OK |
| D47 | D 6.7 | reasons "true about 84% of the time" | | 0.843 capsule, 0.842 dense; aug 0.838 / 0.822 (SEV) | OK |
| D48 | D 6.7 | "84-92% of windows" (depth deletion test) | | hybrid 0.88 / 0.92 / 0.92 / 0.84 (SEV) | OK |
| D49 | D 6.7 | "for the severity class it is weak" | weak | capsule 0.55-0.56, dense 0.73-0.75 (SEV); E calls the dense case moderate | OVERCLAIM |
| D50 | D 6.8 | "6 of the 480 hidden windows ... 98% ... +0.84" | | 6/480; share 98.5%; 0.8415 (REG onset table) | OK |
| D51 | D 6.8 | "five different seeds and a different training recipe all fail" | | fold 0: -8.099, -6.933, -9.082, -8.140, -8.854; batch 8: -5.730 | OK |
| D52 | D 6.8 | "(-0.88, but +0.83 without the first windows)" | | -0.884; +0.834 (REG) | OK |
| D53 | D 6.8 | "Only the two-number lookup table stays sensible there." | only lookup | lookup +0.549 but ConvLSTM +0.444 [+0.277,+0.649] also positive (E 10.12 and F s15 say both hold up) | CONTRADICTION |
| D54 | D 6.8 | "UK -> Australia the models work: +0.66 to +0.74" | all models | local CNN 0.662-0.738 only; ConvLSTM 0.584 / 0.695; MLP 0.588 / 0.659 | OVERCLAIM |
| D55 | D 6.9 | "varies by about +-0.01" | | seed sd 0.006-0.010 (within-fold ranges reach 0.027) | OK |
| D56 | D 6.9 | "653k parameters, 150x larger ... +0.84 on the UK, +0.80 on Mozambique" | | 652,918 = 138x the ConvLSTM, 159x the local CNN; 0.8402 / 0.8019 | OK |
| D57 | D 6.9 | more windows "only helps because it then trains for more steps" | causal | equal steps: -0.002 to +0.008, inside seed noise (REG) | OVERCLAIM |
| D58 | D 6.10 | recipe "about the same never-seen score (0.01 lower) ... +0.785 against +0.811" | | local -0.008 to -0.012; ConvLSTM E60 -0.013, E480 -0.026 (0.785 / 0.811) | OK |
| D59 | D 6.10 | "stops the plain CNN's unstable scores ...: those wobbles were training noise" | proved | one seed (1003) at batch 8: pooled +0.771 vs +0.430 | OVERCLAIM |
| D60 | D s7 | "+0.81 to +0.87 against +0.06 to +0.19 ... stable to about +-0.01 ... two independent reviews found no critical error" | | as D04/D55; CR1 and CR2 both "0 critical" | OK |
| D61 | D s8 | "about 300 training runs, roughly a day and a half of computing" | 300 runs, 36 h | 310 run files; stored train_minutes sum 3,096 = 51.6 h on four parallel workers; first to last result 21.3 h (20 Sep 10:19 to 21 Sep 07:38) | ERROR |
| D62 | D s8 | "12-core CPU, 15.5 GB RAM" | | 12 cores / 14 threads, 15.47 GB (this machine) | OK |
| D63 | D s8 | Colab default: 60 m, local CNN + ConvLSTM, three rain settings, three folds | | notebook cell 4: E60, ["local","clstm"], 3 regimes, folds 0-2 | OK |
| D64 | D s8 | "GPU and CPU differ only in the last decimals." | last decimals | LOG: same configuration moved by about +-0.03 across CPU 6 threads, CPU 12 threads, Colab GPU; E s12 lists a cross-hardware check as NOT established | ERROR |
| D65 | D s9 | "the deck (..._v2.pptx, 19 slides)" | 19 | 19 slides (18 separators in the text) | OK |
| D66 | D s9 | "`completion/results/REPORT.md` has all the tables" | | on-disk REPORT.md is degraded (Finding 0) | ERROR |
| D67 | D s2 | "at 5-minute steps ... perfect to about 2 mm" | | LOG: 2.0 mm on Australia, 0.4 h horizon | OK |
| D68 | D 4.3 | first review found aggregation bug, narrow CIs, two hypothesis tests looser, wrong Grad-CAM test, layout handicap | | CR1 F1-F5 (five majors) | OK |
| E01 | E What-changed r1 | "correlation 1.00 at lag 0, depth gain = 0.50 x the file value" | | slope 0.494-0.502 (VER1 claim A) | OK |
| E02 | E What-changed r2 | "beat the original CNN-LSTM by ~2x" | | 1.7x-2.5x (D12) | OK |
| E03 | E What-changed r2 | "removing it costs 0.01-0.02" | | 0.008-0.020 for local/ConvLSTM (REG); wording differs from 10.3 (C1) | OK |
| E04 | E What-changed r3 | "about +0.8 to +0.87 ... 4,741 parameters" | | 0.807-0.866 hybrid | OK |
| E05 | E What-changed r4 | copying "as well as or better than its best model on most measures" | | copy >= FNO+ on all 9 numbers of F s17 (paper values external) | OK |
| E06 | E s10 main result | lookup +0.79/+0.74; MLP +0.85/+0.83; local +0.84/+0.81; ConvLSTM +0.87/+0.81; straight line +0.32/+0.15 | | 0.786/0.737; 0.846/0.829; 0.841/0.807; 0.866/0.811; 0.323/0.151 | OK |
| E07 | E s10 main | original "roughly +0.05 to +0.2 ... +0.19 UK, +0.06 Mozambique" | | 0.061-0.188 | OK |
| E08 | E s10 main | "+0.025 [+0.004,+0.043] ... equivalent on Mozambique" | | H4 E60 hybrid +0.025 [+0.004,+0.043]; E480 hybrid +0.004 [-0.011,+0.019] equivalent | OK |
| E09 | E s10 main | 90 epochs "+0.004 [-0.021,+0.027] ... -0.014 [-0.029,+0.006], i.e. equivalent" | | inside +-0.03 (rule) by 0.003 and 0.001; margin +-0.03 is 3-4x the seed sd | OK |
| E10 | E s10 main | "at 600 training windows the plain CNN is level with the per-pixel MLP" | | 0.861/0.850/0.833 vs 0.868/0.836/0.844 | OK |
| E11 | E 10.1 | all listed skills and intervals | | match REG; "+0.35" for E60 S3c and E30 S1 is 0.3447 (exact +0.34) | OK |
| E12 | E 10.1 | "a table with a few hundred cells reaches about 90%" | | 12 x 17 bin edges (VER2), about 200 cells; 89-91% | OK |
| E13 | E 10.2 | local causal/hybrid/forced: UK60 +0.82/+0.84/+0.87; UK30 +0.81/+0.83/+0.85; Moz +0.79/+0.81/+0.83 | | 0.822/0.841/0.866; 0.805/0.830/0.851; 0.794/0.807/0.834 | OK |
| E14 | E 10.2 | ConvLSTM Moz +0.81/+0.81/+0.77; UK60 +0.85/+0.87/+0.85 | | 0.810/0.811/0.765; 0.853/0.866/0.854 | OK |
| E15 | E 10.2 | leak "+0.01 to +0.025 ... +0.02 to +0.03 more" (local CNN) | | 0.013-0.025; 0.021-0.027 | OK |
| E16 | E 10.3 | all ablation numbers for ConvLSTM/local, UK60/Moz; "-2.6 to -3.7" | | table rows match; -2.649 ... -3.666 | OK |
| E17 | E 10.3 | "rainfall is worth 0.01-0.03" | | see C1 (0.008-0.020 main models, 0.039 max) | OK |
| E18 | E 10.4 | kernel 1/5/11 numbers; pooled +0.24/+0.43/-0.66; ConvLSTM +0.84, MLP +0.87 | | 0.848/0.841/0.841; 0.835/0.830/0.831; 0.818/0.807/0.793; 0.2404/0.4304/-0.6566; 0.8448; 0.8682 | OK |
| E19 | E 10.5 | H4 numbers; "ConvLSTM better than MLP only on UK 60 m (+0.02)" | | H4 lines match; +0.020 | OK |
| E20 | E 10.6 | training-time ablations and H3 numbers | | 0.841/0.172/0.839/0.159/0.835/0.842 etc.; +0.669 / +0.721 (0.003 / 0.008) | OK |
| E21 | E 10.6 | "(a model with rain and state but no trend reaches only about +0.26)" | +0.26 | S2 never-seen hybrid: +0.207 / +0.188 / +0.100 (BL; not printed in REPORT.md); +0.260 is the E60 held-out-tile mean | ERROR |
| E22 | E 10.6 | "rain is worth about 0.005-0.009" | | no-rain 0.006 (E60), 0.009 (E480) | OK |
| E23 | E 10.7 | H6 pairs; "E60 local ... within the pre-registered 0.05" | | +0.756->+0.716; +0.782->+0.645; +0.730->+0.686; +0.430->-6.564; +0.845->+0.742 | OK |
| E24 | E 10.7 | "6 of the 480 ... 98% ... +0.842 (ConvLSTM +0.866)" | | 6/480; 98.5%; 0.8415; 0.8665 | OK |
| E25 | E 10.7 | "(14,853 against 780 for persistence in that fold)" | 14,853 / 780 | fold 0: model 33,901 vs persistence 1,262 in the six start-0 windows; 34,294 vs 3,769 over all 162 windows (m^2 summed over pixels and 5 leads) | ERROR |
| E26 | E 10.7 | "12 of 900 windows carry 82% ... (+0.865 without them)" | | 0.821; 0.8652 | OK |
| E27 | E 10.7 | "Never-seen skill unaffected (E480 +0.807 -> +0.803; E60 +0.841 -> +0.837)" | | strict runs 0.8027; 0.8367 | OK |
| E28 | E 10.7 | four more seeds "-6.9, -9.1, -10.1, -10.9; the frozen seed -10.1" | | seeds 11-14: -6.933, -9.082, -8.140, -8.854; frozen -8.099 (E's own table just above agrees with these) | ERROR |
| E29 | E 10.7 | "batch-8 / lr 3e-3 recipe (-5.7)"; never-seen "+0.76 to +0.79" | | -5.730; 0.763-0.785 | OK |
| E30 | E 10.7 | "so the frozen recipe is not under-optimised on the never-seen score" | | 90 epochs add +0.021 / +0.008 / +0.024 to the plain CNN (same paragraph) | OVERCLAIM |
| E31 | E 10.7 | "+0.430 to +0.771, i.e. ... instability on Pakistan is optimisation noise, not missing data" | proved | single seed at batch 8 | OVERCLAIM |
| E32 | E 10.7 | 90 epochs "+0.021 (E60), +0.008 (E30), +0.024 (E480); held-out +0.383 to +0.678; ConvLSTM +0.866->+0.866, +0.811->+0.818, +0.843->+0.861" | | +0.0209, +0.0078, +0.0243; 0.383/0.678; ConvLSTM as stated | OK |
| E33 | E 10.7 | "(sd 0.33 over five seeds)" | | mean within-fold sd 0.327 (REG) | OK |
| E34 | E 10.7 | "(the lookup table is the only method that stays sensible: +0.66)" | only lookup | lookup strict +0.661 (BL) but ConvLSTM strict +0.742 (E's own H6 line); per-pixel MLP -9.6 in fold 0 | CONTRADICTION |
| E35 | E 10.8 | ConvLSTM forced +0.765 -> +0.822 (folds), +0.854 -> +0.880 (folds) | | 0.7631/0.7805/0.7524; 0.8224/0.8266/0.8159; 0.8603/0.8553/0.8468; 0.8756/0.8848/0.8798 | OK |
| E36 | E 10.8 | "the ConvLSTM is the best forced-regime model on the UK (local CNN +0.866)" | best | 0.880 vs MLP 0.872 vs local 0.866: within seed noise | OVERCLAIM |
| E37 | E 10.8 | removal costs "0.06-0.08" (side channel) and "0.06-0.13" (local) | | 0.077/0.058; 0.131/0.112/0.059 | OK |
| E38 | E 10.8 | aborted twice; +0.885; retry +0.865 (seed 1015, lr 5e-4) | | REG per-fold table | OK |
| E39 | E 10.9 | per-fold v1 never-seen/held-out numbers; "(mean +0.187) (+0.188) (+0.062)" | | means 0.1867 / 0.1876 / 0.0614 (E480 shown +0.062, table +0.061: trivial) | OK |
| E40 | E 10.9 | "about 0.65 (UK) and 0.75 (Mozambique) higher" | | 0.64-0.68; 0.746-0.750 | OK |
| E41 | E 10.9, s12 | v1 "reproduce ... the +0.18 to +0.19 reported in the first study" | reproduces | first study UK values +0.044 to +0.217 over seven runs and +-0.03 machine-to-machine (LOG) | OVERCLAIM |
| E42 | E 10.10 | seed sd 0.007 / 0.010 / 0.006; between-fold 0.001-0.008; "about 3 seed standard deviations" | | 0.0074 / 0.0101 / 0.0058; 0.0014 / 0.0027 / 0.0082; 0.025/0.0074 = 3.4 | OK |
| E43 | E 10.10 | "single seeds range from -0.37 to +0.71" | +0.71 | max single-seed held-out on Pakistan +0.773 (fold 2, seed 14) | ERROR |
| E44 | E 10.11 | learning-curve values for local CNN, MLP, lookup, straight line | | all match REG | OK |
| E45 | E 10.11 | MLP "already at its level with 25-50 windows" | at level | 25 windows: -0.033 to -0.039 below the 600-window value; 50 windows: 0 to -0.037 (Moz 0.807 vs 0.844) | OVERCLAIM |
| E46 | E 10.11 | equal-steps "-0.002 to +0.008" and the six cells | | +0.006 -0.002; +0.008 +0.004; +0.004 -0.002 | OK |
| E47 | E 10.12 | UK->Australia and Mozambique->Pakistan numbers; 54 of 3,240; 93%; +0.834; ConvLSTM +0.444 / +0.550 | | 0.738/0.659/0.601; 0.662/0.588/0.606; -0.884/-0.312; 54; 93.3%; 0.8336; 0.4437; 0.5500 | OK |
| E48 | E 10.12 | "8-12 m of water ... recedes to 2-7 m ... error is 4-11 times persistence's" | | VER2 noteworthy 1 (fold 0); not recomputed by me | OK |
| E49 | E 10.12 | "On the training region's own hidden time thirds every model scores +0.80 to +0.88." | every model | local 0.799-0.854, MLP 0.833-0.881, ConvLSTM 0.764 / 0.854 / 0.842, lookup 0.639-0.741, linear -1.187 to +0.054 (REG reverse table) | ERROR |
| E50 | E 10.12 | ConvLSTM "+0.695 ... +0.584 ..., slightly below the local CNN" | slightly | 0.043 and 0.078 below; intervals overlap | OVERCLAIM |
| E51 | E 10.13 | heavy: +0.840 [+0.790,+0.893], +0.802 [+0.758,+0.865]; held-out +0.770, +0.582; frozen +0.006 / -0.015; "about 150 times" | | 0.8402; 0.8019; 0.7696; 0.5818; ablation +0.006 / -0.015; 138x-159x | OK |
| E52 | E 10.14 | accuracies, balanced accuracies, gaps, rules before/after, fixed/broken counts, 82-84%, 84-92%, 55-56%, 73-75%, causal 0.623/0.568, +0.056, 84-85%, 93-97%, 57-77% | | all identical to SEV (recomputed subset: section 0) | OK |
| E53 | E 10.14 | "the capsule is right in 22, the dense head in 27, the forecast threshold in 31" | 31 | forecast-threshold right: 31 (capsule+aug row) and 30 (dense+aug row) (SEV) | ERROR |
| E54 | E 10.14 | "advantage was a small-sample regularisation effect, not evidence that capsules suit the task" | causal | as D05 | OVERCLAIM |
| E55 | E 10.14 and severity header | "The severity numbers of 8b ..."; "details in section 9" | cross-refs | there is no section 8b; section 9 is the Visualization Pipeline | ERROR |
| E56 | E 10.15 | "fired eight times: six original-model runs ... two side-channel runs ... seven re-run ... +0.18 to +0.195 ... +0.885" | | REG guard table | OK |
| E57 | E severity narrative | 67.8/60.2/+0.076; 97.4/98.7; 22 (47%) / 24 / 34 (72%); 161/25/109; 64.0% (51,119); 56.5/58.4; 84% / 84% / 91%; 88/92%; 79/76%; 3.7/3.8; 0.52/0.54; 55/73%; 0.17/0.76 | | identical to SEV | OK |
| E58 | E s12 | "skill +0.81 to +0.87 ... in all three rainfall regimes" | 0.81-0.87 in all regimes | hybrid only; across regimes local+ConvLSTM span 0.765-0.866 (0.880 with side channels) | OVERCLAIM |
| E59 | E s12 | "H3 holds by 0.67-0.72 against a fold sd of 0.003-0.008" | 0.003-0.008 | fold sd 0.003 / 0.013 / 0.008 (E30 = 0.013) (H3 line) | ERROR |
| E60 | E s12 | "removing rain costs 0.005-0.015 and terrain nothing" | | training-time rain 0.002-0.015; terrain +0.001 / -0.008 / +0.005 | OK |
| E61 | E s12 | "frozen recipe is not under-optimised ... 90 epochs helps ... +0.008 to +0.024" | | same tension as E30 | OVERCLAIM |
| E62 | E s12 | "(+0.66 to +0.74 at 60 m and 30 m)" for UK -> Australia | all networks | local CNN only; ConvLSTM 0.695 / 0.584 | OVERCLAIM |
| E63 | E s12 | persistence NSE/CSI 0.9995/0.983, 0.9993/0.980 | | PP 0.99951/0.9832; 0.9993/0.980 | OK |
| E64 | E s12 | "right in 22-27 against 31-34 for thresholding the model's own forecast" | 31-34 | 30-34 (rows 30, 30, 31, 32, 34, 34) (SEV) | ERROR |
| E65 | E s12 | "The result files carry five different engine hashes" | five | six in the current 310 files: 6627af098c x136, 72eeb40485 x82, e3e8e93d72 x30, 330a342f7e x27 (today's engine.py), 923283620c x23, fb4ce0ca91 x12 | ERROR |
| E66 | E s12 | "headline UK 60 m runs used a version of model_v2.py that no longer exists" | | 12 runs with a43ab5d317; file on disk is 14cbc30ee4 | OK |
| E67 | E s12 | repro "9e-6 (+0.83794 vs +0.83793) and 5e-5" | | repro_check.json: 8.5e-6, 4.6e-5; 0.83794054 vs 0.83793199 (file is present, so "when present" is stale) | OK |
| E68 | E s12 | "One E30 ConvLSTM causal fold-0 run ... part of no table" | | `E30_clstm_causal_full_f0of3_s1003_ep30.json` exists, group incomplete, not in tables | OK |
| E69 | E s12 | design "used only held-out tiles"; no mention of F10 | | CR1 F10 and STATE say "DISCLOSE": four non-blind design runs printed UK skill; E60 fold-0 held-out windows reused, so that held-out number is not independent of design | OVERCLAIM |
| E70 | E s14 | guard listed only as "caught by smoke tests before use" | | only the first version was; the 12-earliest-window version aborted eight real runs (10.15, LOG s8) | CONTRADICTION |
| E71 | E s14(b) | "Mozambique 238 windows, 17.0757 mm; UK at 60 m 120 windows, 15.7004 mm" | | E480 238 / 17.0757; E60 120 / 15.7004 (run JSONs) | OK |
| E72 | E s14 | five major findings in review 1 | | CR1 F1-F5 | OK |
| E73 | E severity narrative | "(With the original weak model the net effect had been zero on 240 windows.)" | zero | LOG line 518: 3 of 240 changed, 2 fixed, 1 broke (65.4% -> 65.8%): net +1 | ERROR |
| F01 | F s2 | "about 2 weeks ... (2 days to 4 hours for the others)" | | agrees with E s2 (paper figures, external) | OK |
| F02 | F s9 | "2 mm ... (13 mm at 30 min)" | 13 mm at 30 min | 13.5 mm is LOG's stride-6 RMSE over the whole 2.5 h horizon on Australia; per-lead persistence RMSE at the 30-minute lead is 4.8 / 5.6 / 5.2 mm (E60 / E30 / E480), 5-lead means 15.7 / 18.2 / 17.1 mm | ERROR |
| F03 | F s9 | "1 - (RMSE model / RMSE copy)^2" | | equals 1 - SSE ratio (metric definition) | OK |
| F04 | F s9 | "3 tile folds ... 300 windows, 30 epochs" | | frozen recipe (PREREG) | OK |
| F05 | F s10 | "Australia r = -0.28, UK r = -0.41, Mozambique r = -0.27, Pakistan r = -0.09" | | LOG s4 recovery table | OK |
| F06 | F s11 | straight-line guess "about twice" | | 1.7x-2.5x | OK |
| F07 | F s11 | "Every headline number was re-computed from the raw files by a separate agent (all matched)" | every headline number | VER2 "Could NOT verify": all neural skills, ablations, most baselines; only persistence denominators, pools, splits, S5 in 5 cells | OVERCLAIM |
| F08 | F s11 | "no critical error and five major issues, all fixed" | | CR1: 0 critical, 5 major | OK |
| F09 | F s11 vs s17 | FNO+ NSE "0.955" vs table "0.9555" | | same paper value, rounded two ways (0.0005) | OK |
| F10 | F s12 | lookup +0.79 / +0.74, "roughly 90%"; kernels +0.85/+0.84/+0.84 and +0.82/+0.81/+0.79 | | 0.786/0.737; 0.848/0.841/0.841; 0.818/0.807/0.793 | OK |
| F11 | F s12 | "+0.025 ... +0.004 (UK), -0.014 (Mozambique)" | | as D39 | OK |
| F12 | F s13 | "(-2.6 / -2.9)" | | ConvLSTM E60 -2.649, E480 -2.929 (model not named on the slide; others -1.1 to -7.8) | OK |
| F13 | F s13 | "costs about 0.01 to 0.03" (rain removed or strictly past) | | 0.008-0.020 removal, 0.013-0.025 strictly past (local CNN) | OK |
| F14 | F s13 | terrain "less than 0.01" | | largest change 0.006 (v1, E60) test-time; 0.008 (E30) training-time | OK |
| F15 | F s13 | trained without depth history "+0.17 ... +0.09, against +0.84 and +0.81" | | 0.172 / 0.086 vs 0.841 / 0.807 | OK |
| F16 | F s14 | "about +0.02 for the local CNN" | | mean of 0.019, 0.025, 0.013 = 0.019 | OK |
| F17 | F s14 | 30-epoch and 90-epoch intervals; "equivalent" | | as E08 / E09 | OK |
| F18 | F s14 | side channels "+0.85 to +0.88 (UK) and +0.77 to +0.82 (Mozambique)" | | as D42 | OK |
| F19 | F s15 | "moves by about 0.01 between seeds ... below 0.02 cannot be told apart" | | sd 0.006-0.010 | OK |
| F20 | F s15 | 90 epochs "+0.021 (UK) and +0.024 (Mozambique); ConvLSTM +0.000 / +0.007; 150x ... +0.84 UK, +0.80 Mozambique" | | +0.0209, +0.0243; +0.0004, +0.0071; 0.8402, 0.8019 | OK |
| F21 | F s15 | "Per-pixel model: +0.83 with only 25 training windows" | +0.83 | 0.829 (UK 60) but 0.803 (UK 30) and 0.809 (Mozambique) | OVERCLAIM |
| F22 | F s15 | "Plain CNN still rising at 600 (+0.86)" | | 0.861 (UK 60); 0.850 / 0.833 | OK |
| F23 | F s15 | "Batch 8 / lr 3e-3: about 0.01 lower ..., but steadier on Pakistan tiles" | | -0.008 to -0.026; +0.771 vs +0.430 (one seed) | OK |
| F24 | F s15 | "(-5.7 to -9.1 with every seed and both recipes); +0.76 to +0.79" | | -5.730 ... -9.082 (fold 0); 0.763-0.785 | OK |
| F25 | F s15 | "+0.66 to +0.74 at 60 m and 30 m" | all | local CNN only (see E62) | OVERCLAIM |
| F26 | F s15 | "(-0.88); the lookup table (+0.55) and the ConvLSTM (+0.44)" | | -0.884; 0.549; 0.444 | OK |
| F27 | F s15 | guard "fired 8 times; all runs I re-ran ... trained normally (+0.19 ... +0.885)" | | REG guard table (careful wording) | OK |
| F28 | F s15 | "Numbers: completion/results/report.json" | | on-disk report.json is degraded (Finding 0) | ERROR |
| F29 | F s16 | "68% -> 63% ... changed 161 answers, fixed 25 and broke 109" | | 0.678->0.631; 161; 25; 109 (SEV) | OK |
| F30 | F s16 | "88% of windows (severity map: 55%) ... hotter than dry in 79%" | | 0.88 / 0.55 / 0.79 (SEV) | OK |
| F31 | F s16 | "68% ... 60% ... 97% ... 80% / 80%" | | 0.678 / 0.602 / 0.974 / 0.803 / 0.798 | OK |
| F32 | F s16 | "so nothing beats that rule" | none | thresholding the model's own forecast scores 0.986-0.988, above the last-frame rule's 0.974 (SEV) | OVERCLAIM |
| F33 | F s17 | nine persistence numbers (0.0210/0.9995/0.983; 0.0243/0.9993/0.980; 0.0247/0.9993/0.975) | | identical in PP | OK |
| F34 | F s17 | "on the last 20% of each event copying is even better" | | relative error and NSE better in all three; CSI(1 mm) UK 60 m 0.979 vs 0.980 (PP) | OK |
| F35 | F s18 | "a straight-line guess beat our model for weeks" | weeks | project files begin 14 Sep; trend baseline found 20 Sep (about 6 days) | OVERCLAIM |
| F36 | F s18 | "two independent train -> test pairs, not nine" | nine | E s12 says "not sixteen" (16 = H1 cells) | CONTRADICTION |
| F37 | F s4 | "84-100% of patches are severe" | | agrees with E s3 (Mozambique 84%, UK 100%); data fact, not recomputed | OK |
| F38 | F s9 | "All 14,868 files were read" | | agrees with E, LOG | OK |
| F39 | F structure | slides 11-18 are the audit/results/lessons slides | | 19 slides; 11 = audit, 12-17 results, 18 = lessons | OK |

## 2. Every non-OK item, with exact replacement text

Priority H = must fix before use, M = fix, L = wording/precision. "Old" is copied from the document; "New" is the suggested text.

### H items

**N01 (Finding 0; D66, F28; D s9, F s15) degraded report on disk.**
Action: run `python completion/g_report.py` and confirm H1 reads "13/16 cells pass". If that cannot be done before hand-in:
* D s9 Old: "`completion/results/REPORT.md` has all the tables." New: "Section 10 of the explainer contains all the result tables; `completion/results/REPORT.md` is regenerated from the result files with `python completion/g_report.py`."
* F s15 Old: "Numbers: completion/results/report.json." New: "Numbers: the tables in the explainer (section 10), regenerated by completion/g_report.py."

**N02 (E28, E25; E 10.7) numbers that contradict E's own table.**
* Old: "with four more seeds fails in the same way (held-out skill -6.9, -9.1, -10.1, -10.9; the frozen seed -10.1)"
  New: "with four more seeds fails in the same way (held-out skill -6.9, -9.1, -8.1, -8.9; the frozen seed -8.1)"
* Old: "(14,853 against 780 for persistence in that fold)"
  New: "(in that fold the six windows hold 33,901 of the plain CNN's 34,294 squared-error units, against 1,262 of persistence's 3,769; units: m^2 summed over pixels and the five lead times)"

**N03 (D08, D16, F07; D s1, s4.3, s5; F s9, s11) "re-derived the numbers" overstates what the verifiers could do.** The verifier's own report lists as NOT verified: model errors and everything derived from them (every neural skill, CSI, ablations, history), baselines other than S5 in 5 cells, severity, CIs, H1-H6.
* F s11 Old: "Every headline number was re-computed from the raw files by a separate agent (all matched); an adversarial code review found no critical error and five major issues, all fixed."
  New: "A separate agent re-computed from the raw files the rain timing, the copy-the-last-frame errors of every never-seen window, the window pools and the train/test splits, and the lookup-table baseline (all matched); it could not re-run the neural models. An adversarial code review found no critical error and five major issues, all fixed."
* F s9 Old: "a separate agent re-derived key numbers from the raw files" New: "a separate agent re-derived the raw-data numbers, the baselines and the splits from the raw files (the trained networks were not re-run)"
* D s1 Old: "(they re-derived the numbers from raw files and reviewed the code)" New: "(they re-derived the raw-data numbers, the baselines and the train/test splits from the raw files without our code, and reviewed the code; the trained networks were not re-run)"
* D s4.3 Old: "(everything matched: about 190,000 window scores, the lookup-table baseline to 1e-9, no training/test overlap in any of 56 splits)" New: "(everything it could check matched: the stored copy-the-last-frame errors of about 190,000 window scores, the lookup-table baseline to 1e-8 or better, no training/test overlap in any of 56 blocked or reverse splits; it could not re-run the neural models)"
* D s5 Old: "separate agents re-derived the key numbers from the raw files without using our code (all matched), and reviewed the code twice." New: "separate agents re-derived the raw-data numbers, the baselines and the splits from the raw files without using our code (all matched; the trained networks were not re-run), and reviewed the code twice."

**N04 (E49; E 10.12).**
Old: "On the training region's own hidden time thirds every model scores +0.80 to +0.88."
New: "On the training region's own hidden time thirds the local CNN and the per-pixel MLP score +0.80 to +0.88, the ConvLSTM +0.76 to +0.85, the lookup table +0.64 to +0.74 and the linear models far less (-1.19 to +0.05)."

**N05 (D53, E34; "only the lookup table").**
* D 6.8 Old: "Only the two-number lookup table stays sensible there." New: "The two-number lookup table (+0.55) and the ConvLSTM (+0.44) stay sensible there; the plain CNN and the per-pixel network do not."
* E 10.7 Old: "(the lookup table is the only method that stays sensible: +0.66)" New: "(the lookup table stays sensible, +0.66, and so does the ConvLSTM, +0.74 pooled; the plain CNN does not, and neither does the per-pixel MLP, which scores -9.6 in fold 0)"

**N06 (D64; D s8) hardware claim contradicts E s12 and LOG.**
Old: "GPU and CPU differ only in the last decimals."
New: "GPU and CPU runs are not bit-identical: in the first study the same configuration moved by about 0.01 to 0.03 skill between machines and thread counts, and a cross-hardware check of the final models was not done."

### M items

**N07 (E21; E 10.6).** Old: "(a model with rain and state but no trend reaches only about +0.26)"
New: "(a linear model with rain and state but no trend reaches only +0.21 / +0.19 / +0.10 on the never-seen region at UK 60 m / UK 30 m / Mozambique, against +0.34 / +0.36 / +0.18 with the trend)"

**N08 (E59; E s12).** Old: "against a fold sd of 0.003-0.008" New: "against a fold sd of 0.003-0.013"

**N09 (E43; E 10.10).** Old: "single seeds range from -0.37 to +0.71" New: "single seeds range from -0.37 to +0.77"

**N10 (E65; E s12).** Old: "The result files carry five different engine hashes" New: "The result files carry six different engine hashes"

**N11 (E58; E s12).** Old: "skill **+0.81 to +0.87** (UK 60 m +0.84 / +0.87, UK 30 m +0.83 / +0.84, Mozambique +0.81 / +0.81), in all three rainfall regimes, with a seed-to-seed noise of about 0.01."
New: "skill **+0.81 to +0.87** with the original (hybrid) input (UK 60 m +0.84 / +0.87, UK 30 m +0.83 / +0.84, Mozambique +0.81 / +0.81) and +0.77 to +0.87 across the three rainfall regimes (lowest: the ConvLSTM with forced rain on Mozambique, +0.765), with a seed-to-seed noise of about 0.01."

**N12 (E30, E61; E 10.7 and s12) under-optimised.**
* E 10.7 Old: "so the frozen recipe is not under-optimised on the never-seen score;" New: "so a higher learning rate with batch 8 does not help on the never-seen score (but longer training does, see below);"
* E s12 Old: "The frozen recipe is not under-optimised for the never-seen score (batch 8 / lr 3e-3: -0.008 to -0.026); 90 instead of 30 epochs helps the plain CNN by +0.008 to +0.024." New: "A higher learning rate with batch 8 does not help (-0.008 to -0.026 on the never-seen score), but the frozen 30-epoch recipe slightly under-trains the plain CNN: 90 epochs give it +0.008 to +0.024."

**N13 (D21, E69; D s5, E s12) design-phase disclosure (CR1 F10, STATE "DISCLOSE").**
* D s5 Old: "the never-seen floods were kept out of every design decision." New: "the never-seen floods were not used for any design decision (four early design runs of the discarded v2 family did print their UK score; they were not consulted, and the two chosen models were designed with runs that never touched the UK)."
* E s12, add to "Not established": "- **Independence of the design phase.** Four non-blind design runs (the v2 family) printed their UK 60 m score during design; the two chosen architectures (local, clstm) have only blind design runs, and the UK ranking of the heavy runs matched their held-out ranking, so no decision plausibly changed. The E60 fold-0 held-out windows are the windows used to choose the design (same seed), so that one held-out score is not independent of it; the never-seen scores are (code review F10)."

**N14 (D61; D s8).** Old: "(about 300 training runs, roughly a day and a half of computing)" New: "(about 310 training runs, about 52 hours of training in total on four parallel jobs, roughly 21 hours from the first to the last result)"

**N15 (F02; F s9).** Old: "Copying the last frame is that close at 5-minute steps (13 mm at 30 min)." New: "Copying the last frame is that close at 5-minute steps (2 mm over 25 minutes). At our 30-minute steps over 2.5 hours the copy error is 13.5 mm on Australia."

**N16 (F32; F s16).** Old: "so nothing beats that rule." New: "so neither head comes close to that rule (thresholding the model's own forecast is slightly better still, 98.6%)."

**N17 (D05, E54; capsule wording).**
* D s1 Old: "The capsule head only beat its plain control because of the small training set;" New: "The capsule head beat its plain control only without augmentation (with augmentation the gap vanishes, consistent with a small-sample effect);"
* D 6.7 Old: "the capsule's earlier edge came from the small training set." New: "the capsule's earlier edge is consistent with a small-sample effect (one seed per fold; augmentation was not run in the causal regime)."
* E 10.14 Old: "the capsule design's earlier advantage was a small-sample regularisation effect, not evidence that capsules suit the task" New: "the capsule design's earlier advantage is consistent with a small-sample regularisation effect (one seed per fold) and is not evidence that capsules suit the task"
* E s12 Old: "so the capsule's advantage was a small-sample effect" New: "so the capsule's advantage is consistent with a small-sample effect"

**N18 (E53, E64; forecast-threshold counts).**
* E 10.14 Old: "the capsule is right in 22, the dense head in 27, the forecast threshold in 31" New: "the capsule is right in 22, the dense head in 27, the forecast threshold applied to their forecasts in 31 and 30"
* E s12 Old: "they are right in 22-27 against 31-34 for thresholding the model's own forecast" New: "they are right in 22-27 against 30-34 for thresholding the model's own forecast"

**N19 (E55; cross-references).**
* Old: "(hybrid regime; Australia 30 m; 3 folds pooled; details in section 9)" New: "(hybrid regime; Australia 30 m; 3 folds pooled; details in 10.14 and the tables below)"
* Old: "The severity numbers of 8b were obtained with the original 46-patch-per-fold training and no augmentation." New: "The severity numbers in the first severity paragraph above were obtained with the original 46-patch-per-fold training and no augmentation."

**N20 (D19; D s4.3).** Old: "It looked at the 12 *earliest* test windows, when the flood has barely started" New: "It looked at the 12 *earliest* held-out windows (tiles of the training flood), when the flood has barely started"; Old: "The check now samples windows across the whole test period." New: "The check now samples windows across the whole held-out period."

**N21 (D18; D s4.3).** Old: "and so did the side-channel run it had stopped (+0.885)" New: "and so did the first side-channel run it had stopped (+0.885; the second, a retry with the next seed, was not re-run)"

### L items

**N22 (D09; D s3).** Old: "(seven runs ranged +0.04 to +0.23)" New: "(seven runs ranged +0.04 to +0.22 on the UK)"

**N23 (D32; D 6.2, E s12).** D Old: "scrambling its order makes it strongly negative" New: "scrambling its order makes it strongly negative (for every model except the per-pixel CNN on the UK and the original model)". E s12 Old: "scrambling it makes it strongly negative" New: "scrambling it makes it strongly negative (-1.1 to -7.8; the exceptions are the per-pixel CNN on the UK, +0.12 to +0.16, and the original model, which ignores the history)".

**N24 (D37; D 6.4).** Old: "A per-pixel network, a 5×5 and an 11×11 CNN score the same." New: "A per-pixel network, a 5×5 and an 11×11 CNN score within 0.01 of each other on the UK; on Mozambique the 11×11 CNN is 0.025 below the per-pixel one, so looking at neighbours never helps."

**N25 (D41; D 6.6).** Old: "Strictly past rain → the original input adds about +0.01 to +0.025; knowing the future rain adds about +0.02 to +0.03 more." New: "For the local CNN, strictly past rain → the original input adds about +0.01 to +0.025 and knowing the future rain adds about +0.02 to +0.03 more (for the ConvLSTM the first step is +0.013 on the UK at 60 m and +0.000 on Mozambique)."

**N26 (D49; D 6.7).** Old: "for the severity class it is weak." New: "for the severity class it is weak for the capsule head (top beats both controls in 55-56% of windows) and moderate for the dense head (73-75%)."

**N27 (D54, E62, F25; scope of +0.66 to +0.74).**
* D 6.8 Old: "In the other reverse directions (UK → Australia) the models work: +0.66 to +0.74." New: "In the other reverse directions (UK → Australia) the plain CNN works: +0.66 to +0.74 (the ConvLSTM +0.58 to +0.70)."
* F s15 Old: "Train on the small UK flood, test on Australia: +0.66 to +0.74 at 60 m and 30 m." New: "Train on the small UK flood, test on Australia: the plain CNN scores +0.74 (60 m) and +0.66 (30 m)."
* E s12 Old: "Training on the small UK flood and testing on Australia works (+0.66 to +0.74 at 60 m and 30 m)" New: "Training on the small UK flood and testing on Australia works for the plain CNN (+0.74 at 60 m, +0.66 at 30 m; ConvLSTM +0.70 / +0.58)"

**N28 (D57, D59, E31; D 6.9, 6.10, E 10.7).**
* D 6.9 Old: "A learning-curve check shows that giving the plain CNN more training windows only helps because it then trains for more steps; what limits it is training time, not data." New: "A learning-curve check with the number of training steps held equal shows that doubling the training windows changes the plain CNN's score by only -0.002 to +0.008 (inside seed noise): its gain from more windows is consistent with the extra training steps, so what limits it looks like training time, not data."
* D 6.10 Old: "and, as a side effect, stops the plain CNN's unstable scores on hidden Pakistan tiles: those wobbles were training noise." New: "and, in the one run made with it, the plain CNN's scores on hidden Pakistan tiles are steady (+0.77 against +0.43), which suggests those wobbles were training noise."
* E 10.7 Old: "i.e. its tile-level instability on Pakistan is optimisation noise, not missing data" New: "which suggests (one seed) that its tile-level instability on Pakistan is optimisation noise, not missing data"

**N29 (E36; E 10.8).** Old: "the ConvLSTM is the best forced-regime model on the UK (local CNN +0.866)" New: "the ConvLSTM has the highest forced-regime skill on the UK (+0.880, against the per-pixel MLP +0.872 and the local CNN +0.866; differences of this size are within seed noise)"

**N30 (E50; E 10.12).** Old: "slightly below the local CNN" New: "below the local CNN by 0.04 (60 m) and 0.08 (30 m), with overlapping intervals"

**N31 (E41; E 10.9, s12).** E 10.9 Old: "The v1 figures reproduce, by a different code path (the study engine instead of the original notebooks), the +0.18 to +0.19 reported in the first study." New: "The v1 figures agree, by a different code path (the study engine instead of the original notebooks), with the first study's typical UK skill of about +0.19 (its seven runs spanned +0.04 to +0.22, and identical configurations moved by up to 0.03 between machines)." E s12 Old: "(reproducing the first study's +0.18 to +0.19)" New: "(consistent with the first study's typical +0.19)".

**N32 (E45, F21; MLP with few windows).** E 10.11 Old: "(already at its level with 25-50 windows)" New: "(within about 0.04 of its 600-window level already with 25-50 windows)". F s15 Old: "Per-pixel model: +0.83 with only 25 training windows." New: "Per-pixel model: +0.80 to +0.83 with only 25 training windows."

**N33 (F35; F s18).** Old: "missed that a straight-line guess beat our model for weeks." New: "missed, until our second audit, that a straight-line guess beat our model."

**N34 (F36; F s18).** Old: "two independent train → test pairs, not nine." New: "two independent train → test pairs, not sixteen."

**N35 (E70; E s14).** Old: "and, caught by smoke tests before use, a Grad-CAM deletion measure that counted the removed water itself and a dead-network guard that mistook sparse changes for a collapsed model." New: "**a dead-network guard that falsely aborted eight runs because it looked only at the 12 earliest windows, and calling those aborts "collapses" (10.15)**; and, caught by smoke tests before use, a Grad-CAM deletion measure that counted the removed water itself and an earlier version of the guard that mistook sparse changes for a collapsed model."

**N36 (E73; E severity narrative).** Old: "(With the original weak model the net effect had been zero on 240 windows.)" New: "(With the original weak model the rules changed 3 of 240 windows, 2 fixed and 1 broken.)"

**N37 (E67, small).** Old: "(a same-seed rerun on this machine is documented in `completion/results/repro_check.json` when present)" New: "(a same-seed rerun on this machine is documented in `completion/results/repro_check.json`)".

**N38 (rounding, trivial).** D s6 table and E 10.1: the E30 straight-line value 0.3447 and the E60 linear-with-rain value 0.3447 are shown as "+0.35"; exact two-decimal rounding is "+0.34". E 10.9 "(+0.062)" for the Mozambique v1 mean should read "(+0.061)" (0.0614; the table says +0.061). F s11 "0.955" vs F s17 "0.9555": use one form.

**N39 (C1; rain-removal cost).** Make the four places agree. Suggested: E "What changed" row 2 keep "0.01-0.02"; D 6.2 Old: "removing rain costs only 0.01–0.03" New: "removing rain costs only about 0.01–0.02"; E 10.3 Old: "rainfall is worth 0.01-0.03" New: "rainfall is worth about 0.01-0.02 (0.008-0.020 for the local CNN and ConvLSTM; 0.039 at most, for the 11x11 CNN at 30 m)"; F s13 Old: "costs about 0.01 to 0.03" New: "costs about 0.01 to 0.02 (0.01 to 0.025 if only strictly past rain is fed)".

## 3. Contradictions between D, E and F (and within E)

| ID | Where | Statement A | Statement B | Resolution |
|---|---|---|---|---|
| C1 | E table row 2 vs E 10.3, D 6.2, F s13 | rain removal costs "0.01-0.02" | "0.01-0.03" | N39 |
| C2 | D 6.8 vs E 10.12, F s15 | "Only the two-number lookup table stays sensible" | lookup and ConvLSTM both "hold up better" | N05 |
| C3 | D s8 vs E s12 | GPU and CPU "differ only in the last decimals" | cross-hardware check "not established" | N06 |
| C4 | F s18 vs E s12 | "not nine" | "not sixteen" | N34 |
| C5 | D 6.7 vs E 10.14 | severity Grad-CAM "weak" | weak for capsule, moderate for dense | N26 |
| C6 | F s11, D s1/s5 vs E s14(b) | "every headline number" / "the numbers" re-derived | E lists the specific quantities that were | N03 |
| C7 | D 6.6 vs E 10.2 | leak "+0.01 to +0.025" (no model named) | "for the local CNN" | N25 |
| C8 | D s5 vs CR1 F10 / STATE | never-seen floods out of "every design decision" | four design runs printed UK skill; E60 fold-0 held-out reused | N13 |
| C9 | E 10.7 text vs E 10.7 table, E s12, F s15 | strict-split seeds -6.9, -9.1, -10.1, -10.9 (frozen -10.1) | table: -6.933, -9.082, -8.140, -8.854 (frozen -8.099); s12 and slide "-5.7 to -9.1" | N02 |
| C10 | E 10.7 vs E's own H6 line | lookup "the only method that stays sensible" | ConvLSTM +0.845 -> +0.742 | N05 |
| C11 | E 10.7 vs E s12 | "frozen recipe is not under-optimised" | "90 epochs helps the plain CNN by +0.008 to +0.024" (same paragraph) | N12 |
| C12 | E s14 vs E 10.15 | guard listed as "caught by smoke tests before use" | guard aborted eight real runs before it was fixed | N35 |

## 4. What I could NOT verify

* **Any neural-network score from first principles.** Neural skills exist only as stored per-window squared errors; re-running the models (TensorFlow, weights) was out of bounds. I verified (a) that the E section-10 tables are reproduced line by line by the project's own aggregation code run on the current run JSONs, (b) the arithmetic of the stored skills (also done by VER2), and (c) my own recomputation of the numbers quoted in the narrative from those JSONs. The block-bootstrap intervals were taken from that regeneration, not independently re-implemented (VER2 reproduced the E60 never-seen interval only).
* **The paper's numbers**: FNO+ relative error / NSE / CSI in F s17 (0.0786, 0.9555, 0.934; 0.0248, 0.9989, 0.860; 0.0252, 0.9990, 0.811), "the paper reports no persistence baseline / no CNN-LSTM", the Norfolk 4-6 h to 3.2 min figure, and the simulator run times (2 weeks, 2 days, 4 hours). Only the persistence side of s17 was checked (PP).
* **The 4,741-parameter count of the small ConvLSTM** (not stored in any run JSON or engine log; 4,101 and 652,918 are in the logs). "About 150 times" is 138x (vs ConvLSTM) to 159x (vs local CNN).
* **Georeferencing and geometry claims** (E s3, F s10): only the alignment correlations (-0.28, -0.41, -0.27, -0.09, the 69% flood cover) were matched to LOG; the overlays, offsets and the 14,868-file audit were not re-run. VER1 itself could not confirm the Mozambique offset.
* **Rule-layer, Grad-CAM and bootstrap tables of SEVERITY_REPORT.md** were not recomputed (only accuracies, thresholds, class-changing counts and paired differences). The Grad-CAM stability lines need `scipy`, which this python lacks.
* **First-study numbers** (D s3, E 10.9/s12 "first study") were compared with LOG text only; the "46-patch-per-fold" training size (E 10.14) was not re-derived.
* **"8-12 m ... 2-7 m ... 4-11 times"** (E 10.12) comes from VER2 fold 0; I did not re-open the rasters.
* **Run times of the Colab notebook** ("roughly an hour" in D s8; E s8 "dataset pull ~5-8 min, training ~8 min").
* **Pre-registration timing**: file time (10:12:34) precedes the earliest final-run file (10:19) and the recorded md5 matches, but which analyses were decided after seeing which results cannot be tested from files.
* **The deck's charts and speaker notes** (slides 12-16 carry figures; only the text file was audited), so statements such as "Vertical axis starts at 0.5" are unchecked.
* **Sections of E and D outside the assigned scope** (E s1-s9 other than the "What changed" table; E s13; D s2 physics description) were skimmed, not audited line by line.

## 5. Notes on method (for reproducibility)

* Scratch scripts and outputs are in the session scratchpad: `aud1.py`, `aud2.py` (run-JSON recomputations), `regen.py` (in-memory `g_report.main()` with all writes redirected; outputs `regen_REPORT.md`, `regen_report.json`), `regen_sev.py` (stopped at the `scipy` import; nothing written to the project).
* Engine-hash count, train-minute sum, window counts: computed over all 310 files in `completion/results/runs/` (204,470 window scores in total; VER2 counted 191,414 in the 284 files it saw).
* Observation, not an error: the "equivalent" verdicts (H4 at 90 epochs) satisfy the pre-registered rule (|difference| < 0.02 and interval inside +-0.03), but the margin is 3-4 seed standard deviations wide; "no difference larger than 0.03" would be the more literal wording.
* Observation, not an error: under the strict space+time split the per-pixel MLP also fails on Pakistan (S4c held-out -9.6 in fold 0, mean -2.65), which none of D, E, F reports; it supports rather than weakens the onset conclusion.
