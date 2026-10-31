# Flood Forecasting Pipeline — Handoff

Everything needed to run this on your own machine or on Colab.

<!-- ROUND2 START -->
## READ THIS FIRST: what changed on 20 September 2026 ("round 2")

A second, deeper audit corrected several earlier results. **The original CNN-LSTM (v1) and the results printed in "Current state" further down describe the ORIGINAL model and are kept as the record.** The current, corrected study lives in `completion/` and in `flood_forecasting_study_colab.ipynb`. Full details: `PROJECT_LOG.md`, section "Completion, round 2".

1. **Rainfall timing.** Rainfall file *j* is the rain that falls *during* the 30 minutes after time 1800·*j* (mm/h intensities; measured in all four regions). The original loader gave the last observed frame up to 30 minutes of rain that had not yet fallen, so "no future information enters any input" was wrong for rainfall. Three regimes are now tested: `causal`, `hybrid` (the original) and `forced`.
2. **The original model was weak.** A per-pixel linear model of the depth history beat it by about 2x; the original could not use the depth history. The earlier finding "rainfall drives the skill / the LSTM's history is not used" was an artefact of that weakness.
3. **Where the skill really is.** A lookup table keyed on two numbers per pixel (current depth, last change), a per-pixel MLP, a 4k-parameter local CNN and a small full-resolution ConvLSTM (**the final CNN-LSTM, 4,741 parameters**) all reach about +0.8 skill on never-seen regions (0 = copying the last frame; the original model +0.05 .. +0.2). Spatial and recurrent structure adds nothing measurable at equal training length (ConvLSTM minus plain CNN: +0.004 [-0.021, +0.027] on the UK at 60 m and -0.014 [-0.029, +0.006] on Mozambique when both train 90 epochs; details in the log).
4. **The paper's "RMSE" is a relative error, not metres**, and copying the initial frame matches or beats its best model on most of its own measures.
5. **Robustness checks (seeds, training length, learning curve, optimisation recipe, strict space+time split, reverse-direction transfers) and one mistake of mine**: a safety check that aborts 'dead' networks aborted eight runs, seven of which (six original-model runs and one side-channel run) trained normally when re-run without it; the eighth, a retry of the side-channel run with the next seed, was not re-run. It now samples windows across the whole held-out set. Everything is in `PROJECT_LOG.md`, section 'Completion, round 2'.

**Protocol and checks.** Frozen before the final runs: `completion/PREREGISTRATION.md`. Independent re-computation from the raw files of the data-side numbers (rain timing, window pools and persistence errors, the lookup-table baseline, the train/hold-out splits and the arithmetic of every stored skill) matched; the neural networks' own forecast errors were not recomputed and an adversarial code review (no critical finding; five major findings fixed): `completion/review/`. Intervals are block bootstraps; each never-seen region is ONE flood event, so the evidence is two independent train -> test pairs.

**Reproduce (local, Windows venv; the queue is resume-safe):**

```bash
python completion/engine.py cache E60                     # window pools + region caches (needs the local dataset)
SELFTEST_EXPS=E60 python completion/engine.py selftest    # input builder == original loader, regimes, untrained model == persistence
python completion/b_baselines.py E60 hybrid 0 --n-train 300
python completion/engine.py run E60 hybrid 0 --arch clstm --epochs 30 --n-train 300
python completion/run_queue.py A BL K S E B FX C OPT --workers 4 --threads 3    # the main matrix (the exploratory batches are listed in completion/run_queue.py; all 310 runs take about 51 CPU-hours, about 13 h on 4 workers)
python completion/g_report.py ; python completion/j_severity_report.py          # tables + hypothesis verdicts
```
On Colab: run `flood_forecasting_study_colab.ipynb` (defaults to the 60 m experiment). Files: `completion/engine.py` (windows, regimes, folds, training, scoring), `model_v2.py` (local CNN, ConvLSTM, heavier variants), `completion/b_baselines.py`, `completion/severity_study.py`, `completion/d_aggregate.py` / `g_report.py` / `j_severity_report.py`, analyses `a_rain_lag.py`, `e_georef_coarse.py`, `f_paper_protocol.py`, `c_wse_probe.py` (a failed probe, kept as a negative result), and every result under `completion/results/`.

**Skill on the never-seen region (0 = copying the last frame), hybrid regime, mean over 3 tile folds** (full tables with intervals, other regimes and ablations: `completion/results/REPORT.md`):

| model | Australia -> UK, 60 m | Australia -> UK, 30 m | Pakistan -> Mozambique, 480 m |
|---|---|---|---|
| linear extrapolation of the depth history (S1) | +0.323 | +0.345 | +0.151 |
| linear with rain + state + trend (S3, clipped) | +0.345 | +0.359 | +0.182 |
| two-number lookup table (S5) | +0.786 | +0.763 | +0.737 |
| per-pixel MLP (S4, clipped) | +0.846 | +0.854 | +0.829 |
| original CNN-LSTM (v1), same folds and recipe | +0.187 [+0.109, +0.240] | +0.188 [+0.102, +0.246] | +0.061 [+0.030, +0.092] |
| local CNN, kernel 1 (per pixel) | +0.848 [+0.808, +0.898] | +0.835 [+0.783, +0.879] | +0.818 [+0.774, +0.884] |
| local CNN, 5x5 (4,101 parameters) | +0.841 [+0.798, +0.892] | +0.830 [+0.777, +0.876] | +0.807 [+0.766, +0.870] |
| **small ConvLSTM = the final CNN-LSTM (4,741 parameters)** | +0.866 [+0.808, +0.915] | +0.840 [+0.764, +0.904] | +0.811 [+0.761, +0.874] |

<!-- ROUND2 END -->

**Read `PROJECT_LOG.md` first**: "Completion, round 2" and the final "What is and is not established" are the CURRENT state; "Pre-flight before Colab" is the design story of the original (v1) model. It records what was built, and
several places where earlier results were wrong and were corrected - the current design differs from the
original plan in ways that matter (see "What changed from the plan" below).

## What this is

The CNN-LSTM + Capsule Network + Symbolic Rule Layer pipeline from the deck, built against real
FloodCastBench data. It has been run end to end locally at small scale, **twice at full size on a Colab
T4 GPU**, and at full size on CPU (results below and in `PROJECT_LOG.md`, "First real-scale run",
"Second Colab run" and the augmentation A/B).

## What changed from the plan

| Original plan | Now | Why (measured - see PROJECT_LOG.md) |
|---|---|---|
| Train on Australia + Pakistan, test on UK + Mozambique | **The paper's transfer tests**: Pakistan -> Mozambique at 480 m and Australia -> UK at 60 m are same-resolution (the paper's 30 m UK test is a zero-shot downscaling); we run those two plus our own Australia -> UK at 30 m | Mixing resolutions is not sensible (a 128-px patch spans 3.8 km, 7.7 km or 61 km). Pakistan/Mozambique were first left out as "not alignable" - that was wrong (see "Completing the dataset" in PROJECT_LOG.md): their geometry was recovered and cross-checked. |
| Score severity everywhere | **Severity only on Australia at 30 m** | Elsewhere an "always severe" model looks perfect (UK 100%, Pakistan ~100%, Mozambique 84%); the 60 m / 480 m experiments are depth only. |
| Judge by RMSE / accuracy | **Judge against baselines**: persistence (depth), threshold-on-last-frame (severity) | Persistence is accurate to 2 mm at 5 min; thresholding the last frame predicts severity 96.7-99.9% correctly. |
| 5-minute steps | **30-minute stride** | At 5 min the rainfall channel was constant and the task trivial. |
| Decoder rebuilds depth from an LSTM state | **Persistence-anchored decoder + skip connection** | The old decoder could not reach a copy of the last frame; without the skip it learned only a constant drift. |
| Softmax cross-entropy on capsule lengths | **Margin loss** | Softmax over bounded lengths caps at ~0.475 and collapsed to one class. |

## Setup (local machine)

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

TensorFlow has no native-Windows GPU support; local runs are CPU-only, which is fine for the checks below.

## Getting the data

The dataset is mirrored at **https://huggingface.co/datasets/mahajananhad/floodcastbench**
(CC BY 4.0, credit Xu et al. 2025, doi:10.1038/s41597-025-04725-2). Pull only what you need:

```python
from huggingface_hub import snapshot_download
DATA_ROOT = snapshot_download(
    repo_id="mahajananhad/floodcastbench", repo_type="dataset",
    allow_patterns=["High-fidelity flood forecasting/30m/Australia/*",
                    "Relevant data/DEM/Australia_DEM.*",
                    "Relevant data/Land use and land cover/Australia.*",
                    "Relevant data/Rainfall/Australia flood/*",
                    "Relevant data/Georeferenced files/*"],
)
```

Size: the notebook's pull (Australia + UK) is ~13.5GB in 4,385 files, almost all of it Australia's 2,881 depth
frames (7 min 43 s on Colab); the UK alone is 1,019 files. On **native Windows**, add
`local_dir=r"C:\hfdata"` (short path) - the default cache needs symlinks (Developer Mode) and the repo's deep
folder names can exceed the 260-character path limit. Colab (Linux) needs neither.

`dryrun_notebook.py` expects the extracted dataset at `data/extracted/FloodCastBench` (edit
`LOCAL_DATA_ROOT` at its top if yours is elsewhere) and needs Australia + UK.

## Running things locally

```bash
python verify_stage1.py            # data loader sanity check (real Australia patches)
python verify_stage3_capsule.py    # severity-loss gradients reach the shared encoder
python verify_stage3_combined.py   # one combined depth+severity step updates all 29 variables
python dryrun_notebook.py          # executes EVERY notebook cell locally at tiny size (~1 min)
DRYRUN_NOTEBOOK=flood_forecasting_multires_colab.ipynb python dryrun_notebook.py   # same for the multi-resolution notebook
python depth_experiment.py pakistan_mozambique 40 42 400   # 480 m transfer test, full size (~35 min on CPU; plug in)
python depth_experiment.py australia60_uk60 40 42 400      # 60 m transfer test, full size
python dryrun_notebook.py '{"PER_CLASS_TRAIN": 25, "PER_CLASS_VAL": 10, "EPOCHS": 25, "N_START_INDICES": 40, "N_UNIFORM_AU": 40, "UK_START_STEP": 60, "CENSUS_START_STEP": 200}'
                                   # ~5 min; the "smoke test" runs quoted in PROJECT_LOG.md
```

Whole-dataset audit: `python verification_outputs/audit/audit_full_dataset.py` reads **every one of the 14,868 files** (about 25 min).

Independent audit (`verification_outputs/audit/`; re-derives labels, splits, window indexing and metrics from raw files):

```bash
python verification_outputs/audit/audit_data_checks.py        # data-level checks (needs the local dataset)
python verification_outputs/audit/audit_rule_calibration.py   # are the rule thresholds on the right scale? (data only)
python verification_outputs/audit/investigate_uk_nodata.py    # where are the NoData pixels in the UK layers?
python verification_outputs/audit/audit_480m_alignment.py Mozambique 2   # the FIRST alignment search (no trend removal): finds nothing - kept as the record of why it failed
python verification_outputs/audit/probe_alignment_terrain.py Mozambique 24 480m   # the validated probe that did recover the geometry (see PROJECT_LOG 'Completing the dataset')
# behavioural audit of a TRAINED model (full size, ~25 min on CPU; plug the laptop in):
DRYRUN_EXTRA=verification_outputs/audit/audit_after_training.py python dryrun_notebook.py '{"PER_CLASS_TRAIN": 100, "PER_CLASS_VAL": 60, "EPOCHS": 40, "N_START_INDICES": 80, "N_UNIFORM_AU": 150, "UK_START_STEP": 12, "CENSUS_START_STEP": 24}'
```

`dryrun_notebook.py` is the important one: it runs the notebook's real code against the real data, so
errors surface here rather than in a paid Colab session. (The first draft of the notebook passed a syntax
check and was still invalid - this is why.)

## Real-scale training: the Colab notebook

Upload `flood_forecasting_colab.ipynb` to https://colab.research.google.com, **Runtime -> Change runtime type
-> GPU**, run top to bottom. It downloads Australia + UK from Hugging Face, trains on ~400 class-balanced
windows for 40 epochs, and prints depth skill vs persistence, severity vs the threshold baseline, Grad-CAM,
the symbolic-reasoning demo, a georeferenced map, and the Australia -> UK test with an automatic summary.
Read the summary as: **skill > 0 means it beats persistence**; a cross-regional claim needs UK skill > 0.

### The other regions: `flood_forecasting_multires_colab.ipynb`

The main notebook uses 13.5 GB (Australia + UK at 30 m). This second notebook covers the rest of the 26.9 GB dataset following the
paper's own protocols - **Pakistan -> Mozambique at 480 m** and **Australia -> UK at 60 m** - depth only. It downloads what it needs
(Pakistan is ~9.5 GB), runs both experiments and reports skill by lead time and what each model uses (test-time ablations).

## File guide

| File | What it is |
|---|---|
| `real_data_loader.py` | Loads real depth/DEM/rainfall at any region and resolution (30 / 60 / 480 m) via `LAYER_GEOMETRY` (scale + crop offset, recovered from the data); aligns rainfall to depth; refuses NoData windows and unregistered geometries |
| `windows.py` | Window scanning, severity census, class-balanced sampling (shared by notebook and scripts) |
| `model.py` | Shared CNN encoder, LSTM, persistence-anchored decoder with skip connection, capsule head |
| `capsule_layers.py` | PrimaryCaps, routing-by-agreement, margin loss |
| `grad_cam.py` | Direction-aware Grad-CAM for the depth branch |
| `symbolic_rule_layer.py` | Rule engine; thresholds documented with the real data they came from |
| `predict_severity.py` | `predict_severity_with_reasoning()` - capsule prediction + rules + justification |
| `build_notebook.py` | Regenerates the main notebook from the current modules (run after editing any of them) |
| `build_notebook_multires.py`, `depth_experiment.py` | The multi-resolution notebook and the depth experiment it runs (480 m and 60 m transfer tests) |
| `dryrun_notebook.py` | Local end-to-end execution of the notebook |
| `upload_to_hf.py` | Uploaded the dataset (already done; use `upload_folder` for new uploads) |
| `verify_stage1.py`, `verify_stage3_*.py` | Correctness checks |
| `verification_outputs/` | Original logs/plots; `audit/` - the independent audit scripts and logs (incl. the whole-dataset read and the alignment probes); `multires/` - the 480 m / 60 m experiment logs, results and alignment evidence; `preflight/` - the evidence behind the redesign; `augmentation_experiment/` - the full-size local A/B logs; `georeferencing/` - flood depth overlaid on satellite imagery (script + figure) |
| `archive_pre_redesign/` | Scripts from before the redesign; **they do not run** against the current model (see its README) |

## Current state of the ORIGINAL model (v1) - kept as the record; see the round-2 block at the top for the corrected results

**First full-size Colab run (T4, seed 42) - later runs on other seeds gave smaller numbers, see below:**

| Result | Model | Baseline |
|---|---|---|
| Depth, Australia held-out patches | RMSE 10.0 mm, **skill +0.19** | persistence 11.2 mm |
| Depth, **UK (never trained on)** | RMSE 17.5 mm, **skill +0.19** | persistence 19.4 mm |
| Severity, Australia held-out | capsule **70%** | threshold-the-last-frame **90%** (chance 25%) |

[Round 2: v1 statements; a per-pixel linear model of the depth history alone scores +0.15 to +0.35, the final models +0.81 to +0.87, so beating persistence is a low bar.] Depth beats "copy the last frame" in a region it never saw. Severity does **not** beat simply thresholding the
last frame - the head overfits the 46 training patches. The symbolic layer changed 0 of 8 demo classes in that
run and 1 of 8 in the second Colab run (where it turned a right answer wrong). Measured over all 240 validation
windows (the notebook's "Does the symbolic layer improve accuracy?" cell): capsule alone 62.5%, capsule + rules
62.5% - the rules changed 8 windows, 4 fixed and 4 broke, net zero. It is a justification layer, not an accuracy booster. Skill +0.19 (v1) is ~10% lower RMSE, i.e. ~1-2 mm: the simulated
floods change little over 2.5 h.

**Independent audit (September 2026).** Labels, splits, window indexing, loader output and the headline metric were re-derived
from raw files and all pass. **Finding for the ORIGINAL model (v1): its depth skill was rainfall-driven** [Round 2: this described v1's weakness, not the flood - see the block at the top; the audit also did not test rain timing]. On a trained model, removing
rainfall at test time takes skill from +0.22 to ~0 (Australia +0.013, UK +0.004), while removing terrain or scrambling/freezing
the depth history changes it by < 0.01 - so the forecast is effectively "last frame + current rainfall", and the claim that the
LSTM "captures temporal context" is **not supported**. Also: the UK layers carry a NoData padding ring outside the aligned crop
(never read by any window; the loader now refuses such windows); "Pakistan/Mozambique cannot be aligned" was wrong - it came from a
search method that could not succeed, and the dataset was only 50% used (both regions and the 60 m data were later recovered, see below); the rule thresholds came from 40 windows; and the numbers are development-set numbers. Identical
settings differ by about +-0.03 in skill from non-determinism alone. See `PROJECT_LOG.md`, "Independent audit".

**Afterwards, local A/B tests (two seeds, full size, CPU)** of an `AUGMENT` option (random flips/rotations of
training windows): severity accuracy rose on both seeds (62.9% -> 81.2% and 54.6% -> 68.1%; the threshold
baseline is ~92-93%), but on the second seed training destabilised late and depth skill fell (UK +0.21 -> +0.04).
`AUGMENT` is therefore **off by default**. Across all seven full-size runs depth beat persistence in both regions,
but its size varies with seed (UK +0.04 to +0.22; Australia +0.05 to +0.23) - quote a range. The notebook default
`PER_CLASS_VAL` is now 60 (was 25). To reproduce the first Colab run exactly: `PER_CLASS_VAL = 25`, `SEED = 42`.

**The rest of the dataset (Pakistan, Mozambique, 60 m data) - depth skill vs persistence, two seeds** (`verification_outputs/multires/`, `depth_experiment.py`):

| Evaluation | Seed 42 | Seed 7 | Rainfall removed (42 / 7) |
|---|---|---|---|
| Pakistan held-out tiles (480 m) | +0.089 | +0.098 | +0.003 / +0.019 |
| **Mozambique, never seen** (480 m) | +0.047 | +0.074 | +0.001 / +0.008 |
| Australia held-out tiles (60 m) | +0.410 | +0.235 | +0.018 / -0.007 |
| **UK, never seen** (60 m) | +0.176 | +0.183 | +0.003 / +0.005 |

Both of the paper's transfer directions beat persistence on both seeds, including the never-seen regions; the v1 behaviour is the same as at 30 m (removing rainfall took v1's skill to about zero; a v1 property, see the top of this file). Australia's held-out 60 m tiles vary with the seed (different tiles), so the unseen-region numbers are the stable ones. The model is worse than persistence at the 1 mm wet threshold (mm-scale noise in dry cells). The 30 m runs were not re-run with the learning-rate warm-up. See `PROJECT_LOG.md`, "Completing the dataset", sections 7-8.

**Caveats:** few seeds, one storm per unseen region, and small unseen-region tests (UK 30 m: 2 patches; UK 60 m: 2 padded tile positions x 60 start times = 120 windows; Mozambique: 4 overlapping tiles). Read `PROJECT_LOG.md` for what is and is not
established before quoting any number.
