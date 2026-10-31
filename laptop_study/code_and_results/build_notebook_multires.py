"""Builds flood_forecasting_multires_colab.ipynb: the two coarse-resolution transfer experiments (Pakistan -> Mozambique at 480 m,
Australia -> UK at 60 m) that complete the dataset coverage. Regenerate with:  python build_notebook_multires.py
The main notebook (30 m Australia -> UK, with severity) is build_notebook.py and is unchanged in purpose."""
import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.splitlines(keepends=True)})


md("""> **SUPERSEDED (September 2026) - kept as the record of the ORIGINAL model (v1).** A later audit found three problems with this notebook's setup:
> (1) the rainfall input carried up to 30 minutes of rain that had **not yet fallen** (rainfall file `j` is the rain falling *during* `[1800 j, 1800 (j+1))` s, but each input frame
> received the file of the interval *containing* it); (2) a trivial per-pixel linear model beats this CNN-LSTM, and a tiny per-pixel/local network scores far higher;
> (3) this model's dense bottleneck could not use the depth history, so its "rainfall drives the forecast" finding is an artefact of that weakness.
> The current, corrected study is **`flood_forecasting_study_colab.ipynb`** (see `PROJECT_LOG.md`, "Completion, round 2"). Numbers printed here describe v1 only.

# FloodCastBench - the coarse-resolution regions (Colab)

The main notebook trains on Australia and tests on the UK at **30 m**. That uses about half of FloodCastBench. This notebook
covers the rest, following the paper's own two transfer protocols:

| Experiment | Train | Test | Resolution |
|---|---|---|---|
| `pakistan_mozambique` | Pakistan (2022) | Mozambique (2019) | 480 m (low fidelity) |
| `australia60_uk60` | Australia (2022) | UK (2015) | 60 m (high fidelity) |

**Why these regions were missing before, and what changed.** Their depth rasters are the 30 m simulation resampled to 60 m / 480 m
and cropped, but the terrain, rainfall and land-use files stay on the 30 m grid. The first attempt to align them failed because it
correlated depth with terrain without removing the large-scale trend; a probe that does remove it was validated on the known cases
(Australia 30 m -> offset (0,0), UK 30 m -> (5,5)) and then recovered each coarse region's geometry, cross-checked by independent
signals (see `LAYER_GEOMETRY` in `real_data_loader.py` and `verification_outputs/audit/`).

**Depth only.** At these scales a 128-px patch spans 7.7 km (60 m) or 61 km (480 m), so nearly every patch is "severe" and the
severity label carries no information; severity and the symbolic rules are not evaluated here.

**Not comparable to the paper's tables.** The paper forecasts 19 steps from the initial state over the whole domain; this
notebook forecasts 5 steps (30-minute spacing, 2.5 h) from 5 observed frames on 128-px patches. Everything is judged against
persistence ("copy the last frame"): skill = 1 - MSE_model / MSE_persistence.

The last cells report, for each trained model, skill by lead time and what the model uses (test-time removal of rainfall, terrain
and depth history), because the 30 m audit found the depth skill was driven by rainfall alone.""")

md("## Setup")
code("""!pip install -q huggingface_hub rasterio scipy
import warnings
import tensorflow as tf
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
print("GPU available:", tf.config.list_physical_devices('GPU'))
print("TensorFlow version:", tf.__version__)""")

md("""## Configuration

`EXPERIMENTS` picks what to run. Defaults reproduce the local runs (seed 42, 400 training windows, 40 epochs).""")
code("""# CONFIG (overridable)
EPOCHS = 40
N_TRAIN = 400
N_VAL = 150
N_STARTS = 80
SEED = 42
EXPERIMENTS = ["pakistan_mozambique", "australia60_uk60"]""")

md("""## Pull the data from Hugging Face

Only what the chosen experiments need. **Large:** Pakistan alone is about 9.5 GB (5.8 GB depth + 3.6 GB rainfall); Australia 60 m
about 3.3 GB. Colab's disk is ephemeral, so this re-downloads in a fresh session.""")
code("""from huggingface_hub import snapshot_download

PATTERNS = ["Relevant data/Georeferenced files/*"]
if "pakistan_mozambique" in EXPERIMENTS:
    PATTERNS += ["Low-fidelity flood forecasting/480m/*",
                 "Relevant data/DEM/Pakistan_DEM.*", "Relevant data/DEM/Mozambique_DEM.*",
                 "Relevant data/Land use and land cover/Pakistan.*", "Relevant data/Land use and land cover/Mozambique.*",
                 "Relevant data/Rainfall/Pakistan flood/*", "Relevant data/Rainfall/Mozambique flood/*"]
if "australia60_uk60" in EXPERIMENTS:
    PATTERNS += ["High-fidelity flood forecasting/60m/*",
                 "Relevant data/DEM/Australia_DEM.*", "Relevant data/DEM/UK_DEM.*",
                 "Relevant data/Land use and land cover/Australia.*", "Relevant data/Land use and land cover/UK.*",
                 "Relevant data/Rainfall/Australia flood/*", "Relevant data/Rainfall/UK flood/*"]
DATA_ROOT = snapshot_download(repo_id="mahajananhad/floodcastbench", repo_type="dataset", allow_patterns=PATTERNS, max_workers=16)
print("Downloaded to:", DATA_ROOT)""")

md("## Core modules (verified locally)")
for fname in ["real_data_loader.py", "capsule_layers.py", "model.py", "windows.py", "depth_experiment.py"]:
    code(f"%%writefile {fname}\n" + open(fname, encoding="utf-8").read())

md("""## Check the geometry loads

Prints, for each region and resolution used, the depth grid, the registered scale (30 m base pixels per depth cell) and offset, and
how many 128-px tiles are usable. A region without a registered geometry is refused, never guessed.""")
code("""import sys
sys.path.insert(0, ".")
import depth_experiment as de
from real_data_loader import RegionData
from windows import valid_positions, all_positions

de.ROOT = DATA_ROOT
NEEDED = []
if "pakistan_mozambique" in EXPERIMENTS:
    NEEDED += [("Pakistan", 480), ("Mozambique", 480)]
if "australia60_uk60" in EXPERIMENTS:
    NEEDED += [("Australia", 60), ("UK", 60)]
for region, res in NEEDED:
    rd = RegionData(root_dir=DATA_ROOT, region=region, fidelity_dir=de.FID[res])
    print(f"{region:10s} {res:3d} m  depth {rd.grid_height}x{rd.grid_width}  scale {rd.layer_scale}  offset {rd.layer_offset}  "
          f"{len(rd.depth_files)} frames  tiles: {len(all_positions(rd, 128))} -> valid {len(valid_positions(rd, 128, 1.0))}")""")

md("""## Run the experiments

Each one: split the training region's tiles into train / validation (no shared tile), train the persistence-anchored CNN-LSTM on
depth only (loss = MSE / persistence MSE, so 1.0 means exactly copying the last frame), then score the held-out tiles and the test
region. A test domain smaller than one patch (the UK at 60 m is 85x137) is padded and scored on its real pixels only.""")
code("""results = {}
for name in EXPERIMENTS:
    results[name] = de.experiment(name, epochs=EPOCHS, n_train=N_TRAIN, n_val=N_VAL, n_starts=N_STARTS, seed=SEED)""")

md("""## Summary

Skill above 0 beats persistence. The paper's numbers are shown for orientation only (a different task: 19 steps from the initial state,
whole domain); the persistence RMSE on OUR windows is printed so the scale of the problem is visible.""")
code("""import json
print(f"{'experiment':22s} {'held-out skill':>15s} {'test skill':>11s} {'test RMSE model / persistence (m)':>36s}")
for name, r in results.items():
    a = r["test"]["model"]["CSI@0.001"]["RMSE"]; b = r["test"]["persistence"]["CSI@0.001"]["RMSE"]
    print(f"{name:22s} {r['validation']['skill']:+15.4f} {r['test']['skill']:+11.4f} {a:15.4f} / {b:.4f}")
print()
print("Paper (Xu et al. 2025, Table 5, different protocol): Pakistan->Mozambique 480 m FNO+ RMSE 0.0786 m, NSE 0.955 (U-Net RMSE 1.60, NSE -72);"
      " Australia->UK 60 m FNO+ RMSE 0.0248 m, NSE 0.9989 (U-Net RMSE 0.525).")
json.dump(results, open("multires_results.json", "w"), indent=1, default=float)
print("saved multires_results.json")""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
with open("flood_forecasting_multires_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote flood_forecasting_multires_colab.ipynb with {len(cells)} cells")
