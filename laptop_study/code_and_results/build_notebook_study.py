"""Builds flood_forecasting_study_colab.ipynb: the corrected, current study (regimes, baselines, models, ablations, report).
Regenerate with:  python build_notebook_study.py     (embeds the modules verbatim, like the other builders)
Everything heavy runs through `subprocess` (not `!`), so dryrun_notebook.py can execute it locally end to end at tiny size."""
import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)})


md("""# FloodCastBench - the corrected study (Colab)

**This is the current notebook.** `flood_forecasting_colab.ipynb` and `flood_forecasting_multires_colab.ipynb` reproduce the ORIGINAL model (v1) and
its original evaluation; a later audit found three problems with them (see `PROJECT_LOG.md`): (1) the rainfall input carried up to 30 minutes of rain
that had **not yet fallen**, (2) a trivial per-pixel linear model beat the original CNN-LSTM, and (3) v1 could not use the depth history at all.
This notebook re-runs the depth-forecasting study with those problems fixed.

**What it does** (per experiment: `E60` Australia -> UK at 60 m, `E30` the same at 30 m, `E480` Pakistan -> Mozambique at 480 m):

1. builds a cache of every wet window of the training region and of the never-seen test region;
2. runs the **baseline ladder** against persistence ("copy the last frame"): linear models, a two-feature lookup table, a per-pixel MLP;
3. trains the **neural models** (a 4.1k-parameter local CNN, and a small full-resolution ConvLSTM = the project's CNN-LSTM) in **three rainfall regimes**;
4. prints skill vs persistence with confidence intervals on **held-out tiles** and on the **never-seen region**, plus what each model uses
   (test-time removal of rainfall, terrain and depth history).

**The three rainfall regimes** (rainfall file `j` is the rain falling during `[1800 j, 1800 (j+1))` s of the simulation - verified physically):
`causal` = only rain that has already fallen; `hybrid` = the original input (the interval containing each frame, partly future);
`forced` = hybrid plus the rainfall of the 5 forecast intervals (a rain-forced surrogate, the setting the paper's FNO+ works in).

**Cost.** `E60` downloads about 3.4 GB + rainfall; `E30` about 13.5 GB; `E480` about 9.5 GB (Pakistan). Defaults run `E60` only.""")

md("## Setup")
code("""import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "rasterio", "scipy"], check=False)
import warnings
import tensorflow as tf
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
print("GPU available:", tf.config.list_physical_devices('GPU'))
print("TensorFlow version:", tf.__version__)""")

md("""## Configuration

`EPOCHS = 30` and `N_TRAIN = 300` are the frozen recipe of the study (`completion/PREREGISTRATION.md`); reduce them only for a quick smoke test.""")
code("""# CONFIG (overridable)
EXPERIMENTS = ["E60"]                      # any of "E60", "E30", "E480"
REGIMES = ["hybrid", "causal", "forced"]
MODELS = ["local", "clstm"]                # any of "local1", "local", "local11", "clstm"
FOLDS = [0, 1, 2]
EPOCHS = 30
N_TRAIN = 300
THREADS = 4                                # CPU threads per training process
RUN_BASELINES = True
RUN_SEVERITY = False                       # the severity / rules / Grad-CAM study needs E30 (Australia at 30 m)""")

md("## Pull the data from Hugging Face")
code("""from huggingface_hub import snapshot_download

PATTERNS = ["Relevant data/Georeferenced files/*"]
def add(regions, fid):
    global PATTERNS
    for r in regions:
        PATTERNS += [f"{fid}/{r}/*", f"Relevant data/DEM/{r}_DEM.*", f"Relevant data/Land use and land cover/{r}.*", f"Relevant data/Rainfall/{r} flood/*"]
if "E60" in EXPERIMENTS: add(["Australia", "UK"], "High-fidelity flood forecasting/60m")
if "E30" in EXPERIMENTS or RUN_SEVERITY: add(["Australia", "UK"], "High-fidelity flood forecasting/30m")
if "E480" in EXPERIMENTS: add(["Pakistan", "Mozambique"], "Low-fidelity flood forecasting/480m")
DATA_ROOT = snapshot_download(repo_id="mahajananhad/floodcastbench", repo_type="dataset", allow_patterns=PATTERNS, max_workers=16)
print("Downloaded to:", DATA_ROOT)""")

md("## Modules (verified locally; identical to the ones in the code handoff)")
code("""import os
for d in ("completion", "completion/results", "completion/results/runs", "completion/logs", "completion/cache", "data/extracted"):
    os.makedirs(d, exist_ok=True)
if not os.path.exists("data/extracted/FloodCastBench"):
    os.symlink(DATA_ROOT, "data/extracted/FloodCastBench")
print("data linked at data/extracted/FloodCastBench")""")
for fname in ["real_data_loader.py", "windows.py", "capsule_layers.py", "model.py", "model_v2.py", "depth_experiment.py", "symbolic_rule_layer.py",
              "predict_severity.py", "grad_cam.py"]:
    code(f"%%writefile {fname}\n" + open(fname, encoding="utf-8").read())
for fname in ["engine.py", "b_baselines.py", "d_aggregate.py", "g_report.py", "severity_study.py", "visualize.py"]:
    code(f"%%writefile completion/{fname}\n" + open("completion/" + fname, encoding="utf-8").read())

md("""## Build the window caches and check the input builder

`selftest` verifies, on real windows, that the study's input builder reproduces the original loader exactly, that the causal / forced rainfall
inputs equal the raw rainfall files of the intended intervals, and that an untrained model equals persistence in every regime.""")
code("""import os, subprocess, sys
ENV = dict(os.environ, TF_THREADS=str(THREADS))
def run(args, tail=12):
    p = subprocess.run([sys.executable, "-u"] + args, env=ENV, capture_output=True, text=True)
    lines = [l for l in (p.stdout + p.stderr).splitlines() if not l.startswith(("I0000", "W0000", "WARNING"))]
    print("\\n".join(lines[-tail:]))
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {args}")
for e in EXPERIMENTS:
    run(["completion/engine.py", "cache", e], tail=4)
ENV["SELFTEST_EXPS"] = ",".join(EXPERIMENTS)
run(["completion/engine.py", "selftest"], tail=8)""")

md("""## Baselines

Persistence is skill 0 by definition. `S1` linear on depth history only; `S3` linear with rainfall, state and trend; `S5` a two-feature lookup table (current depth x last
one-step change); `S4` a small per-pixel MLP (no convolution, no recurrence). The `c` versions clip the forecast at zero depth.""")
code("""if RUN_BASELINES:
    for e in EXPERIMENTS:
        for r in REGIMES:
            for f in FOLDS:
                print(f"--- baselines {e} {r} fold {f}")
                run(["completion/b_baselines.py", e, r, str(f), "--n-train", str(N_TRAIN)], tail=3)""")

md("""## Train and evaluate the neural models

One run = one (experiment, regime, fold, model): train on the tiles outside the fold, score the held-out tiles and every window of the never-seen region, then the
test-time ablations. Weights are saved (they are tiny) so the forecast maps and Grad-CAM below can be drawn. Results are written to `completion/results/runs/*.json` (a finished run is skipped, so the cell can be re-run after a disconnect).""")
code("""for e in EXPERIMENTS:
    for r in REGIMES:
        for f in FOLDS:
            for a in MODELS:
                print(f"--- {e} {r} fold {f} model {a}")
                run(["completion/engine.py", "run", e, r, str(f), "--arch", a, "--epochs", str(EPOCHS), "--n-train", str(N_TRAIN), "--save-weights"], tail=6)""")

md("""## Results

Skill = 1 - SSE_model / SSE_persistence (0 = copying the last frame). Intervals are cluster bootstraps (tiles for the held-out score, start times for the never-seen region).
The hypotheses block reports the pre-registered checks (H1-H6) for the groups that were run.""")
code("""ENV["FINAL_EPOCHS"], ENV["FINAL_NTRAIN"] = str(EPOCHS), str(N_TRAIN)
p = subprocess.run([sys.executable, "completion/g_report.py"], env=ENV, capture_output=True, text=True)
print(p.stdout[-12000:] if p.returncode == 0 else p.stderr[-3000:])""")

md("""## Figures

1. **Skill ladder** - every rung from "copy the last frame" to the ConvLSTM on the never-seen region (the original CNN-LSTM row appears if you also ran `--arch v1`).
2. **What the trained models use** - skill when one input is removed at test time: the depth history is what they use.
3. **Rain settings** - causal / hybrid / forced.
4. **Training curves** of the first run.
5. **Example forecasts** on never-seen windows: last observed depth, the TRUE depth 2.5 h later, the model's forecast, and the forecast error next to the error of "copy the last frame".
6. **Grad-CAM** of the depth forecast: where the model's evidence for the predicted change sits.""")
code("""import os, sys
os.environ["FINAL_EPOCHS"], os.environ["FINAL_NTRAIN"] = str(EPOCHS), str(N_TRAIN)
sys.path.insert(0, "completion")
import matplotlib.pyplot as plt
import visualize as V

def show(fn, *a, **k):
    try:
        fn(*a, **k)
        plt.show()
    except Exception as ex:                     # a figure needs its data: say what is missing instead of failing the notebook
        print(f"[{fn.__name__} skipped: {type(ex).__name__}: {ex}]")

_arch = "clstm" if "clstm" in MODELS else MODELS[0]
show(V.plot_ladder, tuple(EXPERIMENTS))
show(V.plot_ablations, tuple(EXPERIMENTS), tuple(MODELS))
show(V.plot_regimes, tuple(EXPERIMENTS), tuple(MODELS))
show(V.plot_curves, EXPERIMENTS[0], _arch, REGIMES[0], FOLDS[0])
show(V.plot_forecast, EXPERIMENTS[0], _arch, REGIMES[0], FOLDS[0], EPOCHS)
show(V.plot_gradcam, EXPERIMENTS[0], _arch, REGIMES[0], FOLDS[0], EPOCHS)""")

md("""## Optional: severity, symbolic rules and Grad-CAM (Australia at 30 m)

The severity label is only informative for Australia at 30 m. This trains the joint model (a small ConvLSTM encoder shared by the depth head and a severity head)
with either the **capsule** head or a plain **dense** head, and reports accuracy against the two threshold baselines, the symbolic rule layer's effect
(original vs re-derived thresholds, and whether its stated reasons are true), and Grad-CAM faithfulness. Needs `RUN_SEVERITY = True` and `"E30"` data.""")
code("""if RUN_SEVERITY:
    for r in ("hybrid",):
        for h in ("capsule", "dense"):
            for f in FOLDS:
                print(f"--- severity {r} {h} fold {f}")
                run(["completion/severity_study.py", r, h, str(f), "--epochs", str(EPOCHS)], tail=8)""")

code("""import shutil
shutil.make_archive("study_results", "zip", "completion/results")
print("wrote study_results.zip")
try:
    from google.colab import files
    files.download("study_results.zip")
except Exception:
    pass""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
with open("flood_forecasting_study_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote flood_forecasting_study_colab.ipynb with {len(cells)} cells")
