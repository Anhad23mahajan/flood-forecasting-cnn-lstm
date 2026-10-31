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

@@ROUND2_RESULTS@@

