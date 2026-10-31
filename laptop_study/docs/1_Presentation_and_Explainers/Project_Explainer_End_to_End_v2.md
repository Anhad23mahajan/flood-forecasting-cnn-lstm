# CNN-LSTM Based Spatiotemporal Flood Evolution Forecasting

**With capsule-based severity classification and a post-hoc NeuroSymbolic reasoning layer**

Anhad Mahajan (PRN: 24070126031) & Aman Sumeet Saxena (PRN: 24070126030) — DL Lab Project

This document explains the project end to end: the problem, the dataset, the model as finally built, the pipeline, the tech stack, **what was actually measured**, and what is still not
established. It reflects the state after a second, deeper audit (20 September 2026) that corrected several earlier results, and it matches `CNN-LSTM_Based_Spatiotemporal_Flood_Evolution_Forecasting_v2.pptx`.
The numbers come from `PROJECT_LOG.md` and the result files in the code handoff (`flood_forecasting_handoff.zip`, folder `completion/`), which are the source of truth; if this file and the log disagree, the log wins.

## What changed since the previous version of this explainer (September 19)

| The previous version said | Now (measured) |
|---|---|
| "No future information enters any input." | **Wrong for rainfall.** Rainfall file *j* is the rain falling *during* the 30 minutes after time 1800·*j* (mm/h intensities; checked in all four regions: correlation 1.00 at lag 0, depth gain = 0.50 × the file value). The original loader gave the last observed frame up to 30 minutes of rain that had not yet fallen. Three input regimes are now defined and tested: *causal*, *hybrid* (the original), *forced*. |
| "The forecast is last frame + current rainfall; the LSTM's history is not used." | **An artefact of the weak original model.** A per-pixel linear model that sees only the depth history beat the original CNN-LSTM by ~2×; the original could not read the history (dense bottleneck, depth scaled by 5 m). The final models use the depth history (freezing or scrambling it destroys their skill) and rainfall matters little (removing it costs 0.01-0.02). |
| "The model is the original CNN-LSTM; skill about +0.19 (10% lower error)." | The original model is kept as the record (v1). The final CNN-LSTM is a **small full-resolution ConvLSTM** (4,741 parameters). Skill on never-seen regions is about **+0.8 to +0.87** (0 = copying the last frame). |
| "FNO+ RMSE 0.0786 m / 0.0248 m" | The paper's "RMSE" is a **relative error, not metres**. On the paper's own protocol and metrics, **copying the initial frame** (persistence) scores as well as or better than its best model on most measures; the paper reports no such baseline. |
| "Georeferencing valid for Australia and the UK only." | Also checked against real maps for Mozambique and Pakistan (480 m) and the UK at 60 m. |
| One cross-regional test. | Two independent train → test pairs (each never-seen region is ONE flood event); results now carry block-bootstrap intervals. |

The earlier changes (still true) were:

| Earlier explainer said | Then |
|---|---|
| Cross-regional test: train on Pakistan/Australia, test on Mozambique/UK | The paper's three same-resolution transfer tests: Australia → UK at 30 m and 60 m, Pakistan → Mozambique at 480 m (depth only at 60 m and 480 m); resolutions are never mixed in one model. Pakistan and Mozambique were first dropped as "not alignable" — that was wrong (Section 3). |
| Severity head: 4-way softmax | One capsule length per class with the CapsNet margin loss (softmax over bounded lengths collapsed). |
| CNN decoder: Conv2DTranspose | Persistence-anchored: output = last observed depth + learned change (zero-initialised final layer). |
| 5-minute steps | 30-minute steps (rainfall's native step; at 5 minutes copying the last frame is right to 2 mm). |
| Tech stack: Pandas, scikit-learn, Plotly, Mapbox | Not used. |
| "Physics-informed constraints" | There is no physics loss; the symbolic layer is post-hoc. |

---

## 1. The Problem

**Goal:** given recent flood depth, rainfall, and land-shape (terrain) data, predict what the flood will look like a few time steps into the future — across the *whole* spatial area being modelled, not just at one point.

This is explicitly **not**:
- A water-level sensor. Knowing the water height at one river gauge doesn't tell you which streets or fields nearby will actually go underwater.
- A single-photo flood/no-flood classifier. One satellite image labelled "flooded" or "not flooded" doesn't show how the flood is spreading over time.

The model outputs a **spatiotemporal** prediction: a sequence of depth maps over the forecast horizon, plus a severity class and a stated reason.

## 2. Why It Matters

- A full physics-based hydrodynamic simulation is accurate but slow: per the dataset's own paper, generating the FloodCastBench Pakistan 2022 event (14 simulated days) took **about two weeks of GPU compute** (NVIDIA A6000). The other events were quicker: about two days each for Australia and Mozambique, about four hours for the UK.
- A trained surrogate gives a forecast in seconds. Real-world precedent: a CNN-LSTM surrogate for Norfolk, Virginia cut per-event runtime from **4–6 hours** (physics-based TUFLOW) to **3.2 minutes** (Wang et al., 2026). We did not measure our own inference time against a simulator.
- The gap we address: the FloodCastBench paper reports only U-Net, FNO and FNO+ baselines. We add a **patch-based CNN-LSTM result**, with the paper's three transfer settings (Australia → UK at 30 m and 60 m; Pakistan → Mozambique at 480 m).

## 3. The Dataset — FloodCastBench

- **Citation:** Xu, Q., Shi, Y., Zhao, J. et al. "FloodCastBench: A Large-Scale Dataset and Foundation Models for Flood Modeling and Forecasting." *Scientific Data* 12, 431 (2025). DOI 10.1038/s41597-025-04725-2. CC BY 4.0. Zenodo DOI 10.5281/zenodo.14017092.
- **Our mirror:** the extracted dataset (26.9 GB, 14,868 files) is on Hugging Face at `huggingface.co/datasets/mahajananhad/floodcastbench`, so Colab can pull single regions with `allow_patterns` instead of a 21.6 GB download.
- **Four simulated flood events** (physics simulations, not photographs): Pakistan (2022), UK (2015), Australia (2022), Mozambique (2019). **High-fidelity** (30 m / 60 m): Australia and UK. **Low-fidelity** (480 m): Pakistan and Mozambique.
- **Inputs per event:** depth-map time series (TIFFs), DEM (terrain), rainfall time series, land use/land cover.
- **Filenames:** depth files are unpadded integer seconds since the start (`100200.tif`, `1001700.tif`) and must be sorted **numerically**, not alphabetically. Rainfall files are calendar-stamped (`20220220-S190000.tif`, every 30 minutes).
- **Rainfall ↔ depth alignment (confirmed physically).** Rainfall file *j* is the rain that **falls during** `[1800 j, 1800 (j+1))` s of the simulation: the domain-mean depth gain over each 30-minute interval correlates 1.00 with file *j* (lag 0) in all four regions, with slope 0.494–0.502 (the files are mm/h intensities, so 0.5 h × the value), and the files' dates and durations match the paper's events. An independent re-computation from the raw files reproduced this. The original loader used the file of the interval that *contains* each frame's time — so the last observed frame carried up to 30 minutes of future rain (the *hybrid* regime).

### What we found about the four regions (measured)

| Region / resolution | Depth grid | Terrain / rain / land-use grid | Recovered geometry (30 m pixels per cell, crop offset) | Used as |
|---|---|---|---|---|
| Australia 30 m | 1073×1073, 2,881 frames (5 min) | 1073×1073 — identical | 1, (0, 0) | train (severity + depth) |
| UK 30 m | 170×275, 865 frames | 180×285 — depth is a crop | 1, (5, 5) | test (depth) |
| Australia 60 m | 536×536, 2,881 frames | 1073×1073 | 2, (0, 0) | train (depth) |
| UK 60 m | 85×137, 865 frames | 180×285 | 2, (5, 5) | test (depth; smaller than one patch, padded + masked) |
| Pakistan 480 m | 810×441, 4,033 frames | 13035×7298 (30 m) | 16, (42, 178) | train (depth) |
| Mozambique 480 m | 151×138, 1,729 frames | 2624×2321 (30 m) | 16, (150, 50) | test (depth) |

- **UK offset, measured two ways:** depth-vs-terrain correlation is −0.274 at offset (5,5) against a maximum of +0.031 over all 121 candidates; on a real 128 px patch, aligned correlation is −0.39 versus ≈0 unaligned. Without the offset the model would have been paired with unrelated terrain.
- **Severity classes are not in the dataset.** We derive them from depth: max depth in the patch over the forecast horizon, with fixed thresholds — none < 0.3 m, minor 0.3–1 m, moderate 1–2 m, severe > 2 m. This label is **not scale-invariant**: Australia is about 12% none / 9% minor / 8% moderate / 70% severe; the UK is 100% severe; Pakistan ~100%; Mozambique 84%. So severity is trained and scored on Australia at 30 m only; the 60 m and 480 m experiments are depth only.
- **How the coarse grids were recovered.** The 60 m and 480 m depth rasters are the 30 m simulation resampled (bilinear, per the paper) to 2×2 / 16×16-pixel cells and **cropped**, while terrain, rainfall and land use stay on the 30 m grid. The first attempt found nothing because it correlated depth with terrain without removing the large-scale terrain trend. A probe that high-passes both maps was **validated on the known cases first** (Australia 30 m → (0,0), r = −0.28; UK 30 m → (5,5), r = −0.41) and then recovered every coarse geometry: Mozambique (150, 50) at r = −0.27, 5.2 robust-sd (the main river channel visibly lines up, `multires/alignment_evidence.png`); Pakistan (42, 178), where two independent signals agree but the evidence is **weaker** (r = −0.09; the flood covers 69% of the domain); Australia 60 m (0, 0) and UK 60 m (5, 5). Precision is about ±2 pixels (60 m), under 6% of a 480 m cell, and cannot matter because rainfall's native resolution is ~10 km and the depth branch does not use terrain.
- **Whole-dataset audit:** every one of the 14,868 files (26.87 GB) was read: all depth frames are intact (no NaN, no negative depth, exactly even 300 s spacing); the only NoData is the padding ring in the UK, Pakistan and Mozambique terrain and rainfall rasters; rainfall start dates match the paper's event dates for all four events (hour-level start still inferred). The pipeline had previously read 13.47 GB, 50%.
- **NoData ring (found in the audit):** the UK terrain and rainfall rasters carry the float32 NoData sentinel (−3.4×10³⁸) in a padding ring *outside* the aligned crop (1,452 DEM pixels). No window we read touches it (checked over every file), so results are unaffected; the loader now refuses any window that does. The ring lying exactly outside the crop is a third independent confirmation of the (5, 5) offset.
- **Georeferencing:** the depth TIFFs carry no CRS. The archive's `Spatial_References.txt` (UTM zone per region, e.g. Australia = UTM 56S) and one `.tfw` world file per region supply it. The world files describe the 30 m grid; for the coarse grids the recovered scale and offset are applied on top. **Checked against real maps** (OpenStreetMap; Esri imagery returned HTTP 403 that day): Mozambique (simulated river, estuary and coast), Pakistan (two lakes, the desert and forest edges, the Indus channel) and the UK at 60 m (the River Eden meanders through Carlisle) all fall on the right ground at kilometre / tens-of-metres precision (`completion/results/georef_*.png`).

- **Negative result and a caveat on the alignment.** A physical water-surface probe (terrain + depth should be smooth where water is ponded) **failed its own validation** on Mozambique and is not used as evidence. The independent verifier could not confirm the crop offsets by its own probes either (inconclusive). They rest on the earlier terrain probe, the NoData-ring constraint and the map overlays; the final models are insensitive to them at that precision (removing the rainfall altogether costs only 0.01–0.02 skill).

## 4. Literature Basis

| Paper | Year | Contribution | Relevance |
|---|---|---|---|
| Shi et al. (ConvLSTM), arXiv:1506.04214 | 2015 | Convolutional gates fuse CNN + LSTM in one cell | Closest architectural ancestor; we use a lighter CNN-then-LSTM variant |
| Ronneberger et al. (U-Net), doi:10.1007/978-3-319-24574-4_28 | 2015 | Encoder–decoder CNN with skip connections | FloodCastBench's own spatial-only baseline; our decoder also has a skip connection |
| Malik, Feng, Shao & Abduljabbar, doi:10.1007/s12145-024-01354-y | 2024 | Time-distributed CNN-LSTM (TD-CNN-LSTM) | The design pattern we adopt for the depth branch |
| Wang et al. (Norfolk, VA), doi:10.1016/j.ejrh.2026.103234 | 2026 | CNN and LSTM fused at a bottleneck | Closest published real-world analogue; source of the "3.2 minutes" figure |
| Xu et al. (FloodCastBench), doi:10.1038/s41597-025-04725-2 | 2025 | The dataset, plus U-Net/FNO/FNO+ benchmark | Our dataset; the paper reports no CNN-LSTM baseline |
| Sabour, Frosst & Hinton (Capsule Networks), arXiv:1710.09829 | 2017 | Vector capsules, routing-by-agreement, margin loss | Basis for the severity head, including its loss |

## 5. The Research Gap, and What Is Explicitly Out of Scope

**Gaps we address:**
- No CNN-LSTM/ConvLSTM baseline is reported for FloodCastBench (only U-Net and FNO/FNO+); we add one. **More importantly, the paper reports no trivial baseline; we add a full baseline ladder** (copy the last frame, linear extrapolation, a two-feature lookup table, a per-pixel MLP) and find that on this task most of the skill sits in the simplest rungs (Section 10).
- To our knowledge FloodCastBench's cross-regional transfer has not been tried with CNN-LSTM; we run the paper's three same-resolution settings.

**Scope limits (deliberate boundaries):**
- No new hydrodynamic simulations; we rely on FloodCastBench's simulated sequences.
- **Patch-based** (128×128) forecasts, 5 observed frames → 5 forecast frames at 30-minute steps (2.5 hours ahead). No whole-scene training, no claim of operational early-warning accuracy.
- **All four regions and three resolutions are used, never mixed in one model.** Severity is scored only for Australia at 30 m (the label is degenerate elsewhere). Our task differs from the paper's (19 steps at 300 s from the initial state over the whole domain), so our numbers are **not comparable** to its tables.
- Each never-seen region is **one flood event**, and E60/E30 share the same Australia and UK events: the evidence amounts to **two independent train → test pairs**.
- No new Indian dataset. Capsules have no precedent for dense regression, so they are scoped to severity classification; the motivation (routing-by-agreement may preserve spatial relationships) is a **hypothesis we tested, not an assumption** (Section 10).

## 6. Architecture (as built)

**The final model — a small full-resolution CNN-LSTM (4,741 parameters).** Input: 5 observed frames (30-minute steps, 128×128) of depth, terrain and rainfall, plus the depth change since the previous frame (in centimetres).
A **ConvLSTM2D** (8 channels, 3×3) runs over the frames at the *native resolution* — the temporal state is a map, not a vector, so per-pixel change stays visible. A 3×3 convolution and a zero-initialised 1×1 convolution turn its final hidden map
into 5 change maps; **forecast = last observed depth + change** (persistence-anchored: an untrained model equals "copy the last frame" exactly). See `completion/results/architecture_v3.png` (and slide 7).

**Controls and comparators (same recipe, same folds).** `local`: a 4,101-parameter CNN with no recurrence (the temporal information enters only through the frame-difference channels); `local1` / `local11`: the same with 1×1 or 11×11 kernels (kernel 1 = a per-pixel network, no spatial context);
`v1`: the original CNN-LSTM (each frame flattened through one 128-number dense layer; kept as the record); heavier ConvLSTM variants for one fold. The **baseline ladder** (no deep learning): S1 linear on depth history, S3 linear with rain and terrain, **S5 a lookup table of the mean future change per bin of (current depth × last one-step change)**, S4 a per-pixel MLP.

**Why the original failed.** It flattened the 16×16×64 CNN features of each frame into a single 128-number dense layer before the LSTM and scaled depth by 5 m, so millimetre-scale frame-to-frame changes were ~10⁻⁴ in network units and no per-pixel temporal difference could survive the bottleneck; its decoder saw only the last frame's features. A per-pixel linear model of the depth history therefore beat it by ~2×.

**Severity branch and explanations.** The joint model shares the ConvLSTM encoder: the depth head above and a severity head (three stride-2 convolutions to 16×16×64, then either the **capsule** head — PrimaryCaps → routing-by-agreement → class capsules, margin loss — or a plain **dense** head as the control). The severity label is derived from the maximum depth in the patch over the forecast horizon (none < 0.3 m, minor < 1 m, moderate < 2 m, severe > 2 m); it is only informative for Australia at 30 m.
**Grad-CAM** explains the depth forecast (and, separately, the severity head's predicted class) at the shared encoder's hidden map; the **symbolic rule layer** is post hoc — deterministic rules over predicted depth rate, rainfall, slope and drainage proximity adjust the severity class by ±1 and attach a reason; **neither changes the forward pass.**

## 7. End-to-End Pipeline

| Step | What happens | Output |
|---|---|---|
| 1. Data | Load depth, DEM, rainfall for any region and resolution (30 / 60 / 480 m) via a recovered scale + crop offset per combination; refuse NoData windows and unregistered geometries; build the input for one of three **rainfall regimes** (causal / hybrid / forced) | Aligned rasters, cached per region |
| 2. Windows | 128×128 patches, 5 observed + 5 forecast frames at 30-minute steps; every wet window kept; **3 spatial tile folds** (each tile held out once); optional strict hold-out of a time block with an embargo | Training / held-out / never-seen windows |
| 3. Baselines | Persistence, linear, lookup table, per-pixel MLP on exactly the same windows | The ladder |
| 4. ConvLSTM encoder + head | Full-resolution ConvLSTM over the 5 frames; change maps added to the last observed depth | **Depth maps** |
| *(parallel)* Severity head | Capsule (or dense control) on the shared hidden map | **Severity class** |
| 5. Symbolic Rule Layer | Post-hoc rules over predicted depth rate, rainfall, slope, drainage proximity | Reasoned severity + justification |
| 6. Evaluate | Skill vs persistence (per lead), RMSE, CSI; block-bootstrap intervals; test-time and training-time ablations; hypotheses H1–H6 | Results tables |
| 7. Visualize | Georeferenced overlay on a real basemap | Map |

## 8. Tech Stack (what was actually used)

| Tool | Role |
|---|---|
| Python (3.11 locally; Colab's own version) | Data loading, model code |
| TensorFlow / Keras (2.21 local, 2.20 on Colab) | CNN, LSTM, custom Capsule layers, margin loss, the whole model in one graph |
| NumPy | Patch extraction, metrics |
| SciPy | Distance transform for the drainage-proximity rule |
| rasterio | Read/write GeoTIFF; apply per-region CRS/transform |
| Matplotlib | Training curves, Grad-CAM heatmaps, evaluation plots |
| folium | Interactive overlay on Esri World Imagery (OpenStreetMap tiles are blocked inside Colab) |
| Hugging Face Hub | Hosts the dataset mirror |
| Google Colab (T4 GPU) | Full-size training: dataset pull ~5–8 min, training ~8 min |

Not used, despite appearing in the earlier deck: Pandas, scikit-learn, Plotly, Mapbox.

## 9. Visualization Pipeline

`GeoTIFF (predicted depth) → rasterio (CRS + transform) → folium (overlay on satellite imagery)`

The per-region UTM zone and pixel transform come from FloodCastBench's own world files and spatial-reference table (which describe the 30 m grid; for the coarse grids the recovered scale and offset are applied on top). The transform includes the half-pixel shift for the pixel-centre origin.
Verified against real imagery/maps: Australia 30 m (Esri imagery), and — in the second audit — Mozambique and Pakistan at 480 m and the UK at 60 m against OpenStreetMap (rivers, lakes, coast, meanders line up). The interactive map itself (folium) was built and executed for Australia.

## 10. Evaluation and Results

**How to read the numbers.** Skill = 1 − (model squared error ÷ persistence squared error): **0 = no better than copying the last observed frame, 1 = perfect, negative = worse.** "Held-out tiles" are map squares of the *training* region never used for training (3 spatial folds; every tile is held out once); "never-seen region" is a different flood in a different place (the UK, Mozambique) — the stronger test. Intervals are 95% **block bootstraps** over time blocks (and tiles): neighbouring windows overlap in time and each region is one event, so they are wider than a naive bootstrap.
All models use one frozen recipe, written down before the final runs (`completion/PREREGISTRATION.md`); design choices for the new models used only held-out tiles of the training region.

**The main result.** On the task as defined (5 observed frames → 5 forecast frames, 30-minute steps), a forecast that merely **continues each pixel's recent depth trajectory** already reaches most of the achievable skill:
a lookup table keyed on just two numbers per pixel (current depth, last one-step change) scores about +0.79 on the never-seen UK (60 m) and +0.74 on Mozambique; a per-pixel MLP (no convolution, no recurrence) about +0.85 and +0.83; the small local CNN +0.84 and +0.81; the small ConvLSTM **+0.87 and +0.81** — against +0.32 / +0.15 for a straight-line (linear) extrapolation and roughly +0.05 to +0.2 for the original CNN-LSTM. Spatial and recurrent structure adds nothing measurable: under the frozen 30-epoch recipe the ConvLSTM is ahead of the local CNN only on the UK at 60 m (+0.025 [+0.004, +0.043], hybrid regime) and equivalent on Mozambique, but that edge is a training-length effect of the plain CNN: with both models trained for 90 epochs the difference is +0.004 [−0.021, +0.027] (UK 60 m) and −0.014 [−0.029, +0.006] (Mozambique), i.e. equivalent; and at 600 training windows the plain CNN is level with the per-pixel MLP. The original CNN-LSTM scores +0.19 on the UK and +0.06 on Mozambique on the same folds (it uses only the rain and cannot read the depth history).


### Result tables (generated from the result files)


#### E60: Australia -> UK, 60 m  (skill vs persistence; mean over 3 tile folds; [95% CI]: bootstrap over start times / tiles)

| model | causal: never-seen test | hybrid: never-seen test | forced: never-seen test | causal: held-out tiles (neural: pooled; baselines: mean of folds) | hybrid: held-out tiles (neural: pooled; baselines: mean of folds) | forced: held-out tiles (neural: pooled; baselines: mean of folds) |
|---|---|---|---|---|---|---|
| persistence (copy the last frame) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) |
| linear: depth trend only | +0.323 | +0.323 | +0.323 | +0.303 | +0.303 | +0.303 |
| linear: rain + state + trend (>=0) | +0.334 | +0.345 | +0.365 | +0.357 | +0.376 | +0.430 |
| 2-feature lookup table (depth x last change) | +0.786 | +0.786 | +0.786 | +0.702 | +0.702 | +0.702 |
| per-pixel MLP | +0.839 | +0.846 | +0.872 | +0.791 | +0.798 | +0.847 |
| **original CNN-LSTM (v1) †** | - | +0.187 [+0.109, +0.240] | - | - | +0.212 [+0.066, +0.314] | - |
| **CNN kernel 1 (per pixel)** | - | +0.848 [+0.808, +0.898] | - | - | +0.760 [+0.651, +0.828] | - |
| **local CNN 5x5, 4.1k params** | +0.822 [+0.780, +0.878] | +0.841 [+0.798, +0.892] | +0.866 [+0.814, +0.914] | +0.744 [+0.641, +0.808] | +0.756 [+0.660, +0.814] | +0.817 [+0.715, +0.883] |
| **CNN kernel 11** | - | +0.841 [+0.796, +0.889] | - | - | +0.749 [+0.652, +0.811] | - |
| **small ConvLSTM (CNN-LSTM), 4.7k params** | +0.853 [+0.796, +0.905] | +0.866 [+0.808, +0.915] | +0.854 [+0.785, +0.911] | +0.764 [+0.670, +0.822] | +0.782 [+0.686, +0.840] | +0.808 [+0.708, +0.873] |
| **ConvLSTM, future rain as side channels (forced only) †** | - | - | +0.880 [+0.819, +0.930] | - | - | +0.823 [+0.728, +0.882] |
| **heavy ConvLSTM + local path** | - | +0.840 [+0.790, +0.893] | - | - | +0.770 [+0.667, +0.829] | - |

† contains runs that were finished with the dead-network check disabled after it aborted them falsely (same seed and recipe; see the table 'Runs aborted by the dead-network guard').

#### E30: Australia -> UK, 30 m  (skill vs persistence; mean over 3 tile folds; [95% CI]: bootstrap over start times / tiles)

| model | causal: never-seen test | hybrid: never-seen test | forced: never-seen test | causal: held-out tiles (neural: pooled; baselines: mean of folds) | hybrid: held-out tiles (neural: pooled; baselines: mean of folds) | forced: held-out tiles (neural: pooled; baselines: mean of folds) |
|---|---|---|---|---|---|---|
| persistence (copy the last frame) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) |
| linear: depth trend only | +0.345 | +0.345 | +0.345 | +0.336 | +0.336 | +0.336 |
| linear: rain + state + trend (>=0) | +0.347 | +0.359 | +0.379 | +0.357 | +0.370 | +0.441 |
| 2-feature lookup table (depth x last change) | +0.763 | +0.763 | +0.763 | +0.676 | +0.676 | +0.676 |
| per-pixel MLP | +0.841 | +0.854 | +0.869 | +0.766 | +0.775 | +0.817 |
| **original CNN-LSTM (v1) †** | - | +0.188 [+0.102, +0.246] | - | - | +0.193 [+0.069, +0.281] | - |
| **CNN kernel 1 (per pixel)** | - | +0.835 [+0.783, +0.879] | - | - | +0.739 [+0.654, +0.806] | - |
| **local CNN 5x5, 4.1k params** | +0.805 [+0.750, +0.854] | +0.830 [+0.777, +0.876] | +0.851 [+0.790, +0.904] | +0.723 [+0.621, +0.794] | +0.730 [+0.649, +0.803] | +0.799 [+0.712, +0.863] |
| **CNN kernel 11** | - | +0.831 [+0.775, +0.880] | - | - | +0.733 [+0.647, +0.803] | - |
| **small ConvLSTM (CNN-LSTM), 4.7k params** | - | +0.840 [+0.764, +0.904] | - | - | +0.768 [+0.687, +0.827] | - |

† contains runs that were finished with the dead-network check disabled after it aborted them falsely (same seed and recipe; see the table 'Runs aborted by the dead-network guard').

#### E480: Pakistan -> Mozambique, 480 m  (skill vs persistence; mean over 3 tile folds; [95% CI]: bootstrap over start times / tiles)

| model | causal: never-seen test | hybrid: never-seen test | forced: never-seen test | causal: held-out tiles (neural: pooled; baselines: mean of folds) | hybrid: held-out tiles (neural: pooled; baselines: mean of folds) | forced: held-out tiles (neural: pooled; baselines: mean of folds) |
|---|---|---|---|---|---|---|
| persistence (copy the last frame) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) |
| linear: depth trend only | +0.151 | +0.151 | +0.151 | +0.146 | +0.146 | +0.146 |
| linear: rain + state + trend (>=0) | +0.170 | +0.182 | +0.202 | +0.195 | +0.219 | +0.261 |
| 2-feature lookup table (depth x last change) | +0.737 | +0.737 | +0.737 | +0.721 | +0.721 | +0.721 |
| per-pixel MLP | +0.835 | +0.829 | +0.855 | +0.874 | +0.868 | +0.916 |
| **original CNN-LSTM (v1)** | - | +0.061 [+0.030, +0.092] | - | - | +0.096 [+0.044, +0.204] | - |
| **CNN kernel 1 (per pixel)** | - | +0.818 [+0.774, +0.884] | - | - | +0.240 [-0.523, +0.906] | - |
| **local CNN 5x5, 4.1k params** | +0.794 [+0.757, +0.860] | +0.807 [+0.766, +0.870] | +0.834 [+0.790, +0.895] | +0.263 [-0.440, +0.885] | +0.430 [-0.113, +0.895] | +0.463 [-0.078, +0.948] |
| **CNN kernel 11** | - | +0.793 [+0.749, +0.859] | - | - | -0.657 [-2.590, +0.888] | - |
| **small ConvLSTM (CNN-LSTM), 4.7k params** | +0.810 [+0.763, +0.874] | +0.811 [+0.761, +0.874] | +0.765 [+0.699, +0.844] | +0.840 [+0.794, +0.898] | +0.845 [+0.806, +0.901] | +0.815 [+0.743, +0.921] |
| **ConvLSTM, future rain as side channels (forced only)** | - | - | +0.822 [+0.765, +0.887] | - | - | +0.880 [+0.836, +0.946] |
| **heavy ConvLSTM + local path** | - | +0.802 [+0.758, +0.865] | - | - | +0.582 [+0.242, +0.903] | - |

#### Pre-registered hypotheses

* **H1** (local and clstm beat persistence with both intervals above 0, every regime/experiment): 13/16 cells pass; not passing: local/E480/causal (test CI +0.757, held-out CI -0.440); local/E480/hybrid (test CI +0.766, held-out CI -0.113); local/E480/forced (test CI +0.790, held-out CI -0.078).
* **H2** (beat the best linear baseline S3c on the never-seen region): 16/16 cells pass.
* **H2b** (exploratory, added after seeing the per-pixel baseline: do the CNN/ConvLSTM beat the per-pixel MLP S4c on the never-seen region?): 2/16 cells; mean difference -0.020 (range -0.090 .. +0.019).
* **H3** (depth history matters; local trained nodepthhist vs full, never-seen region): full - nodepthhist, mean over folds (fold sd of the full model): E60/hybrid +0.669 (0.003); E30/hybrid +0.676 (0.013); E480/hybrid +0.721 (0.008). Rule (difference > fold sd) holds in 3/3 cells.
* **H3** (depth history matters; local trained nohist vs full, never-seen region): full - nohist, mean over folds (fold sd of the full model): E60/hybrid +0.682 (0.003); E30/hybrid +0.677 (0.013); E480/hybrid +0.722 (0.008). Rule (difference > fold sd) holds in 3/3 cells.
* **H4** (recurrence matters: clstm - local, never-seen region, paired block bootstrap; pre-registered rule): E60/causal +0.031 [+0.008, +0.052] -> clstm larger; E60/hybrid +0.025 [+0.004, +0.043] -> clstm larger; E60/forced -0.012 [-0.031, +0.002] -> local larger (by the rule, but the interval includes 0); E30/hybrid +0.009 [-0.014, +0.031] -> clstm larger (by the rule, but the interval includes 0); E480/causal +0.016 [-0.004, +0.034] -> clstm larger (by the rule, but the interval includes 0); E480/hybrid +0.004 [-0.011, +0.019] -> equivalent; E480/forced -0.069 [-0.095, -0.038] -> local larger.
* **H5** (causal <= hybrid <= forced on the never-seen region, means): 3/5 (arch, experiment) cases hold. Caution (code review F4): the recurrent model reads the forced rain as five extra steps AFTER the observed history, which handicaps it; a failure for clstm is an input-layout effect, not evidence about rain physics.
* **H6** (strict space+time hold-out within 0.05 of the tile-only hold-out): E60/local: +0.756 -> +0.716; E60/clstm: +0.782 -> +0.645; E30/local: +0.730 -> +0.686; E480/local: +0.430 -> -6.564; E480/clstm: +0.845 -> +0.742. Pass in 2/5.

#### What the trained models use (test-time ablations, never-seen region, hybrid regime, mean over folds)

| experiment | model | none | rain_removed | terrain_removed | depth_hist_scrambled | depth_hist_frozen | rain_hist_frozen | rain_shifted_earlier |
|---|---|---|---|---|---|---|---|---|
| E60 | original CNN-LSTM (v1) | +0.187 | +0.003 | +0.181 | +0.187 | +0.187 | +0.186 | +0.186 |
| E60 | CNN kernel 1 (per pixel) | +0.848 | +0.838 | +0.848 | +0.163 | +0.037 | +0.846 | +0.843 |
| E60 | local CNN 5x5, 4.1k params | +0.841 | +0.833 | +0.841 | -3.037 | -0.006 | +0.834 | +0.833 |
| E60 | CNN kernel 11 | +0.841 | +0.825 | +0.841 | -5.952 | +0.012 | +0.831 | +0.834 |
| E60 | small ConvLSTM (CNN-LSTM), 4.7k params | +0.866 | +0.857 | +0.865 | -2.649 | +0.020 | +0.863 | +0.863 |
| E60 | heavy ConvLSTM + local path | +0.840 | +0.832 | +0.840 | -1.654 | +0.006 | +0.832 | +0.834 |
| E30 | original CNN-LSTM (v1) | +0.188 | +0.003 | +0.183 | +0.188 | +0.188 | +0.187 | +0.188 |
| E30 | CNN kernel 1 (per pixel) | +0.835 | +0.824 | +0.835 | +0.121 | +0.020 | +0.830 | +0.829 |
| E30 | local CNN 5x5, 4.1k params | +0.830 | +0.810 | +0.830 | -1.393 | +0.010 | +0.825 | +0.819 |
| E30 | CNN kernel 11 | +0.831 | +0.792 | +0.831 | -6.173 | +0.013 | +0.825 | +0.819 |
| E30 | small ConvLSTM (CNN-LSTM), 4.7k params | +0.840 | +0.827 | +0.839 | -1.141 | +0.012 | +0.837 | +0.832 |
| E480 | original CNN-LSTM (v1) | +0.061 | +0.002 | +0.060 | +0.061 | +0.061 | +0.061 | +0.053 |
| E480 | CNN kernel 1 (per pixel) | +0.818 | +0.812 | +0.818 | -2.856 | -0.018 | +0.812 | +0.801 |
| E480 | local CNN 5x5, 4.1k params | +0.807 | +0.790 | +0.807 | -3.666 | -0.008 | +0.798 | +0.784 |
| E480 | CNN kernel 11 | +0.793 | +0.778 | +0.793 | -3.501 | -0.010 | +0.784 | +0.777 |
| E480 | small ConvLSTM (CNN-LSTM), 4.7k params | +0.811 | +0.791 | +0.811 | -2.929 | -0.007 | +0.797 | +0.791 |
| E480 | heavy ConvLSTM + local path | +0.802 | +0.783 | +0.802 | -7.831 | -0.015 | +0.789 | +0.778 |

#### Training-time ablations (models trained without one ingredient; never-seen region skill, mean over folds)

| experiment | model | regime | full | nodepthhist | norainhist | nohist | norain | noterrain |
|---|---|---|---|---|---|---|---|---|
| E60 | local CNN | hybrid | +0.841 | +0.172 | +0.839 | +0.159 | +0.835 | +0.842 |
| E30 | local CNN | hybrid | +0.830 | +0.155 | +0.817 | +0.154 | +0.815 | +0.822 |
| E480 | local CNN | hybrid | +0.807 | +0.086 | +0.802 | +0.085 | +0.798 | +0.812 |

#### What the models do with KNOWN future rain (test-time ablations, forced regime, never-seen region, mean over folds)

| experiment | model | none | rain_removed | future_rain_removed | depth_hist_frozen |
|---|---|---|---|---|---|
| E60 | local CNN 5x5, 4.1k params | +0.866 | +0.834 | +0.735 | +0.024 |
| E60 | small ConvLSTM (CNN-LSTM), 4.7k params | +0.854 | +0.825 | +0.753 | +0.031 |
| E60 | ConvLSTM, future rain as side channels (forced only) | +0.880 | +0.855 | +0.803 | +0.015 |
| E30 | local CNN 5x5, 4.1k params | +0.851 | +0.810 | +0.739 | +0.035 |
| E480 | local CNN 5x5, 4.1k params | +0.834 | +0.801 | +0.775 | +0.005 |
| E480 | small ConvLSTM (CNN-LSTM), 4.7k params | +0.765 | +0.736 | +0.702 | +0.007 |
| E480 | ConvLSTM, future rain as side channels (forced only) | +0.822 | +0.784 | +0.764 | +0.014 |

#### Exploratory: ConvLSTM with the future rain as side channels (clstmf, forced regime), per fold (frozen recipe unless marked)

| experiment | model | fold | seed | never-seen skill | held-out tiles skill | note |
|---|---|---|---|---|---|---|
| E60 | clstm | 0 | 1003 | +0.860 | +0.804 |  |
| E60 | clstm | 1 | 1013 | +0.855 | +0.835 |  |
| E60 | clstm | 2 | 1023 | +0.847 | +0.791 |  |
| E60 | clstmf | 0 | 1003 | +0.876 | +0.808 |  |
| E60 | clstmf | 1 | 1013 | +0.885 | +0.843 | guard disabled after a false abort (post hoc; identical recipe and seed) |
| E60 | clstmf | 1 | 1015 | +0.865 | +0.841 | RETRY after the frozen-seed run was aborted by the guard (post hoc): seed 1015, learning rate 0.0005 |
| E60 | clstmf | 2 | 1023 | +0.880 | +0.819 |  |
| E480 | clstm | 0 | 1003 | +0.763 | +0.773 |  |
| E480 | clstm | 1 | 1013 | +0.780 | +0.855 |  |
| E480 | clstm | 2 | 1023 | +0.752 | +0.836 |  |
| E480 | clstmf | 0 | 1003 | +0.822 | +0.852 |  |
| E480 | clstmf | 1 | 1013 | +0.827 | +0.909 |  |
| E480 | clstmf | 2 | 1023 | +0.816 | +0.891 |  |

#### The original model (v1) against the final models, same folds, hybrid regime, per fold (never-seen / held-out tiles)

| experiment | fold | original v1 | local CNN | small ConvLSTM |
|---|---|---|---|---|
| E60 | 0 | +0.184 / +0.209 (guard disabled: the old check looked at the earliest, near-static windows and aborted it falsely) | +0.838 / +0.727 | +0.863 / +0.765 |
| E60 | 1 | +0.195 / +0.259 (guard disabled: the old check looked at the earliest, near-static windows and aborted it falsely) | +0.843 / +0.784 | +0.868 / +0.804 |
| E60 | 2 | +0.181 / +0.178 (guard disabled: the old check looked at the earliest, near-static windows and aborted it falsely) | +0.842 / +0.753 | +0.866 / +0.776 |
| E30 | 0 | +0.181 / +0.174 (guard disabled: the old check looked at the earliest, near-static windows and aborted it falsely) | +0.837 / +0.714 | +0.847 / +0.776 |
| E30 | 1 | +0.193 / +0.220 (guard disabled: the old check looked at the earliest, near-static windows and aborted it falsely) | +0.839 / +0.742 | +0.843 / +0.777 |
| E30 | 2 | +0.189 / +0.184 (guard disabled: the old check looked at the earliest, near-static windows and aborted it falsely) | +0.815 / +0.733 | +0.830 / +0.753 |
| E480 | 0 | +0.060 / +0.099 | +0.801 / +0.333 | +0.811 / +0.833 |
| E480 | 1 | +0.063 / +0.167 | +0.815 / +0.126 | +0.814 / +0.832 |
| E480 | 2 | +0.062 / +0.056 | +0.805 / +0.692 | +0.806 / +0.863 |

#### Exploratory: optimisation robustness (batch 8, learning rate 3e-3, hybrid; a deviation from the frozen recipe)

| experiment | model | frozen recipe: never-seen / held-out | batch 8, lr 3e-3: never-seen / held-out | folds |
|---|---|---|---|---|
| E60 | local | +0.841 / +0.756 | +0.829 / +0.751 | 3 |
| E60 | clstm | +0.866 / +0.782 | +0.853 / +0.764 | 3 |
| E30 | local | +0.830 / +0.730 | +0.822 / +0.738 | 3 |
| E480 | local | +0.807 / +0.430 | +0.798 / +0.771 | 3 |
| E480 | clstm | +0.811 / +0.845 | +0.785 / +0.837 | 3 |

#### Exploratory: seed repeats of the local CNN (hybrid; the frozen-seed run plus seeds 11-14)

| experiment | fold | never-seen skill per seed (frozen seed first) | sd over seeds | held-out tiles skill per seed (seed AND data draw change) | sd over seeds |
|---|---|---|---|---|---|
| E60 | 0 | +0.838 +0.843 +0.844 +0.857 +0.839 | 0.008 | +0.727 +0.734 +0.764 +0.773 +0.750 | 0.020 |
| E60 | 1 | +0.843 +0.843 +0.839 +0.856 +0.846 | 0.006 | +0.784 +0.799 +0.796 +0.802 +0.778 | 0.010 |
| E60 | 2 | +0.842 +0.837 +0.839 +0.857 +0.838 | 0.008 | +0.753 +0.757 +0.771 +0.788 +0.758 | 0.014 |
| E30 | 0 | +0.837 +0.823 +0.831 +0.843 +0.830 | 0.007 | +0.714 +0.703 +0.653 +0.702 +0.701 | 0.024 |
| E30 | 1 | +0.839 +0.843 +0.825 +0.854 +0.827 | 0.012 | +0.742 +0.775 +0.759 +0.742 +0.688 | 0.033 |
| E30 | 2 | +0.815 +0.835 +0.831 +0.842 +0.841 | 0.011 | +0.733 +0.754 +0.736 +0.792 +0.756 | 0.023 |
| E480 | 0 | +0.801 +0.812 +0.807 +0.806 +0.816 | 0.006 | +0.333 +0.658 +0.333 -0.372 +0.675 | 0.424 |
| E480 | 1 | +0.815 +0.818 +0.809 +0.811 +0.822 | 0.005 | +0.126 +0.679 +0.603 -0.073 +0.709 | 0.358 |
| E480 | 2 | +0.805 +0.801 +0.790 +0.795 +0.802 | 0.006 | +0.692 +0.764 +0.660 +0.289 +0.773 | 0.200 |

**E60:** sd over seeds within a fold (mean of the 3 folds): never-seen 0.007, held-out tiles 0.015; sd of the seed-averaged skill over the 3 folds: never-seen 0.001, held-out tiles 0.021.

**E30:** sd over seeds within a fold (mean of the 3 folds): never-seen 0.010, held-out tiles 0.027; sd of the seed-averaged skill over the 3 folds: never-seen 0.003, held-out tiles 0.031.

**E480:** sd over seeds within a fold (mean of the 3 folds): never-seen 0.006, held-out tiles 0.327; sd of the seed-averaged skill over the 3 folds: never-seen 0.008, held-out tiles 0.161.

#### Exploratory: the strict space+time split on Pakistan (E480, local CNN, fold 0): other seeds and the batch-8 / lr 3e-3 recipe

| recipe | seed | held-out (tiles AND a time block hidden) | never-seen (Mozambique) |
|---|---|---|---|
| frozen (batch 1, lr 1e-3) | 1003 | -8.099 | +0.780 |
| frozen (batch 1, lr 1e-3) | 11 | -6.933 | +0.784 |
| frozen (batch 1, lr 1e-3) | 12 | -9.082 | +0.777 |
| frozen (batch 1, lr 1e-3) | 13 | -8.140 | +0.774 |
| frozen (batch 1, lr 1e-3) | 14 | -8.854 | +0.785 |
| batch 8, lr 0.003 | 1003 | -5.730 | +0.763 |

#### Exploratory: learning curve (never-seen skill vs number of training windows; hybrid; mean over 3 folds; the frozen recipe uses 300; FIXED 30 epochs, so the number of optimiser steps grows with n - see the equal-steps table below)

| experiment | model | n = 25 | n = 50 | n = 100 | n = 300 | n = 600 |
|---|---|---|---|---|---|---|
| E60 | local CNN | +0.760 | +0.794 | +0.823 | +0.841 | +0.861 |
| E60 | linear trend | +0.365 | +0.348 | +0.349 | +0.323 | +0.327 |
| E60 | lookup table | +0.710 | +0.716 | +0.753 | +0.786 | +0.798 |
| E60 | per-pixel MLP | +0.829 | +0.857 | +0.859 | +0.846 | +0.868 |
| E30 | local CNN | +0.666 | +0.762 | +0.787 | +0.830 | +0.850 |
| E30 | linear trend | +0.373 | +0.362 | +0.359 | +0.345 | +0.325 |
| E30 | lookup table | +0.623 | +0.696 | +0.722 | +0.763 | +0.764 |
| E30 | per-pixel MLP | +0.803 | +0.836 | +0.837 | +0.854 | +0.836 |
| E480 | local CNN | +0.643 | +0.760 | +0.766 | +0.807 | +0.833 |
| E480 | linear trend | +0.230 | +0.168 | +0.163 | +0.151 | +0.159 |
| E480 | lookup table | +0.489 | +0.585 | +0.647 | +0.737 | +0.758 |
| E480 | per-pixel MLP | +0.809 | +0.807 | +0.831 | +0.829 | +0.844 |

#### Exploratory: learning curve with EQUAL optimiser steps (local CNN, hybrid, never-seen, mean of 3 folds; steps = epochs x windows)

| experiment | n=300, 30 epochs (9,000 steps) | n=600, 15 epochs (9,000 steps) | n=300, 60 epochs (18,000 steps) | n=600, 30 epochs (18,000 steps) |
|---|---|---|---|---|
| E60 | +0.841 | +0.847 | +0.863 | +0.861 |
| E30 | +0.830 | +0.838 | +0.846 | +0.850 |
| E480 | +0.807 | +0.811 | +0.835 | +0.833 |

#### Exploratory: training length (hybrid; 30 epochs = frozen recipe versus 90 epochs; mean over 3 folds)

| experiment | model | never-seen, 30 epochs | never-seen, 90 epochs | held-out tiles (mean of fold skills), 30 | held-out tiles, 90 |
|---|---|---|---|---|---|
| E60 | local CNN | +0.841 | +0.862 | +0.755 | +0.770 |
| E60 | small ConvLSTM | +0.866 | +0.866 | +0.782 | +0.794 |
| E30 | local CNN | +0.830 | +0.838 | +0.730 | +0.749 |
| E480 | local CNN | +0.807 | +0.831 | +0.383 | +0.678 |
| E480 | small ConvLSTM | +0.811 | +0.818 | +0.843 | +0.861 |

* Recurrence with EQUAL training length (90 epochs both), E60: ConvLSTM minus local CNN = +0.004 [-0.021, +0.027] (paired block bootstrap over time blocks)
* Recurrence with EQUAL training length (90 epochs both), E480: ConvLSTM minus local CNN = -0.014 [-0.029, +0.006] (paired block bootstrap over time blocks)

#### Exploratory: reverse-direction transfers (train on the small region, test on the big one; 3 TIME-block folds, no tile folds - see completion/reverse_transfer.py)

Never-seen skill (mean over the 3 folds, [95% block-bootstrap interval]); 'time-held-out' = windows of the training region from a hidden third of the start times (pooled). Not comparable with the forward experiments cell by cell.

| experiment | model | causal: never-seen | hybrid: never-seen | forced: never-seen | causal: time-held-out | hybrid: time-held-out | forced: time-held-out |
|---|---|---|---|---|---|---|---|
| R480 (Mozambique -> Pakistan, 480 m) | linear: depth trend only | -0.675 | -0.675 | -0.675 | -0.055 | -0.055 | -0.055 |
| R480 (Mozambique -> Pakistan, 480 m) | linear: rain + state + trend (>=0) | -0.490 | -0.489 | -0.465 | +0.002 | +0.021 | +0.052 |
| R480 (Mozambique -> Pakistan, 480 m) | 2-feature lookup table (depth x last change) | +0.549 | +0.549 | +0.549 | +0.639 | +0.639 | +0.639 |
| R480 (Mozambique -> Pakistan, 480 m) | per-pixel MLP | -1.111 | -0.312 | -0.560 | +0.877 | +0.867 | +0.881 |
| R480 (Mozambique -> Pakistan, 480 m) | **local CNN 5x5** | -0.869 [-2.636, +0.855] | -0.884 [-2.665, +0.853] | -0.583 [-2.126, +0.914] | +0.816 | +0.825 | +0.847 |
| R480 (Mozambique -> Pakistan, 480 m) | **small ConvLSTM** | - | +0.444 [+0.277, +0.649] | - | - | +0.764 | - |
| R60 (UK -> Australia, 60 m) | linear: depth trend only | +0.025 | +0.025 | +0.025 | -1.187 | -1.187 | -1.187 |
| R60 (UK -> Australia, 60 m) | linear: rain + state + trend (>=0) | +0.082 | +0.144 | +0.075 | -0.822 | -0.739 | -0.658 |
| R60 (UK -> Australia, 60 m) | 2-feature lookup table (depth x last change) | +0.601 | +0.601 | +0.601 | +0.741 | +0.741 | +0.741 |
| R60 (UK -> Australia, 60 m) | per-pixel MLP | +0.701 | +0.659 | +0.745 | +0.865 | +0.851 | +0.880 |
| R60 (UK -> Australia, 60 m) | **local CNN 5x5** | +0.702 [+0.523, +0.769] | +0.738 [+0.571, +0.797] | +0.772 [+0.624, +0.824] | +0.799 | +0.808 | +0.834 |
| R60 (UK -> Australia, 60 m) | **small ConvLSTM** | - | +0.695 [+0.499, +0.757] | - | - | +0.854 | - |
| R30 (UK -> Australia, 30 m) | linear: depth trend only | +0.236 | +0.236 | +0.236 | -0.165 | -0.165 | -0.165 |
| R30 (UK -> Australia, 30 m) | linear: rain + state + trend (>=0) | +0.276 | +0.330 | +0.316 | -0.034 | +0.011 | +0.054 |
| R30 (UK -> Australia, 30 m) | 2-feature lookup table (depth x last change) | +0.606 | +0.606 | +0.606 | +0.737 | +0.737 | +0.737 |
| R30 (UK -> Australia, 30 m) | per-pixel MLP | +0.636 | +0.588 | +0.690 | +0.851 | +0.833 | +0.858 |
| R30 (UK -> Australia, 30 m) | **local CNN 5x5** | +0.637 [+0.375, +0.728] | +0.662 [+0.420, +0.746] | +0.718 [+0.524, +0.790] | +0.820 | +0.821 | +0.854 |
| R30 (UK -> Australia, 30 m) | **small ConvLSTM** | - | +0.584 [+0.399, +0.661] | - | - | +0.842 | - |

#### Where the failures are: the windows at the very START of the flood (start index 0)

Pooled over the 3 folds. 'start 0' = windows whose first observed frame is the first frame of the simulation (the flood is just beginning).

| what | windows (of which start 0) | skill, all windows | share of the model's squared error in the start-0 windows | skill without the start-0 windows |
|---|---|---|---|---|
| Mozambique -> Pakistan (R480), local CNN, never-seen | 3240 (54) | -0.884 | 93% | +0.834 |
| Mozambique -> Pakistan (R480), ConvLSTM, never-seen | 3240 (54) | +0.444 | 38% | +0.550 |
| E480 strict split, local CNN, held-out | 480 (6) | -6.564 | 98% | +0.842 |
| E480 strict split, ConvLSTM, held-out | 480 (6) | +0.742 | 63% | +0.866 |
| E480 tile-only hold-out, local CNN, held-out | 900 (12) | +0.430 | 82% | +0.865 |
| E480 tile-only hold-out, ConvLSTM, held-out | 900 (12) | +0.845 | 40% | +0.877 |
| E480 never-seen (Mozambique), local CNN | 714 (6) | +0.807 | 0% | +0.807 |
| UK -> Australia 60 m (R60), local CNN, never-seen | 2859 (45) | +0.738 | 0% | +0.738 |
| UK -> Australia 30 m (R30), local CNN, never-seen | 11010 (159) | +0.662 | 0% | +0.663 |

#### Runs aborted by the dead-network guard (the OLD check: per-pixel spread of the predicted change below 1e-5 m across the 12 EARLIEST held-out windows; an aborted run gives no result of its own)

| run | epoch of abort | per-pixel spread (m) | outcome of a re-run with the guard disabled (post hoc) |
|---|---|---|---|
| E30_v1_hybrid_full_f0of3_s1003_ep30 | 2 | 3.83e-06 | finished: never-seen +0.181, held-out +0.174, final train/persistence loss 0.70 |
| E30_v1_hybrid_full_f1of3_s1013_ep30 | 2 | 7.95e-06 | finished: never-seen +0.193, held-out +0.220, final train/persistence loss 0.65 |
| E30_v1_hybrid_full_f2of3_s1023_ep30 | 5 | 4.91e-06 | finished: never-seen +0.189, held-out +0.184, final train/persistence loss 0.70 |
| E60_clstmf_forced_full_f1of3_s1013_ep30 | 13 | 8.04e-06 | finished: never-seen +0.885, held-out +0.843, final train/persistence loss 0.19 |
| E60_clstmf_forced_retry_f1 | 6 | 3.95e-06 | not re-run |
| E60_v1_hybrid_full_f0of3_s1003_ep30 | 2 | 6.29e-06 | finished: never-seen +0.184, held-out +0.209, final train/persistence loss 0.60 |
| E60_v1_hybrid_full_f1of3_s1013_ep30 | 2 | 8.06e-07 | finished: never-seen +0.195, held-out +0.259, final train/persistence loss 0.61 |
| E60_v1_hybrid_full_f2of3_s1023_ep30 | 2 | 1.48e-06 | finished: never-seen +0.181, held-out +0.178, final train/persistence loss 0.64 |

The check is a safety net against a network that has died (constant output). All eight aborts listed here came from its first version, which looked only at the 12 EARLIEST held-out windows: the flood has barely started there, so even a healthy network predicts almost no change (and a slow-starting model such as the original one looks dead for its first epochs). Seven of the eight were re-run with the check disabled (same seed and recipe) and trained normally; the tables mark those runs with a dagger. The check now samples 12 windows across the whole held-out set (changed 21 September, about 00:00). The eighth (a retry with the next seed) was not re-run.

#### Data warnings: result groups that are incomplete or hold duplicate folds (their cells are shown as '-' above)

* E30 / clstm / causal / full: folds present [0] (each of 0..2 is needed exactly once)

### Reading the results

**10.1 Where the skill is.** Never-seen region, hybrid input, mean of 3 tile folds (0 = copying the last frame; intervals in the tables above). Australia -> UK at 60 m: straight-line extrapolation of the depth history +0.32, linear with rain/state/trend +0.34, **a two-feature lookup table (current depth x last one-step change) +0.79**, per-pixel MLP +0.85, local CNN +0.84 [+0.80, +0.89], **small ConvLSTM +0.87 [+0.81, +0.91]**. At 30 m: +0.34 / +0.36 / +0.76 / +0.85 / +0.83 / +0.84. Pakistan -> Mozambique at 480 m: +0.15 / +0.18 / +0.74 / +0.83 / +0.81 / +0.81. The original CNN-LSTM scored roughly +0.05 to +0.2 on the same tasks (see 10.9). The forecast is, to first order, **a nonlinear per-pixel function of the recent depth trajectory**: a table with a few hundred cells reaches about 90% of the best score; adding rain, terrain, neighbouring pixels or recurrence moves it by at most a few hundredths.

**10.2 Rainfall regimes (H5).** For the local CNN the ordering causal < hybrid < forced holds in all three experiments (UK 60 m +0.82 / +0.84 / +0.87; UK 30 m +0.81 / +0.83 / +0.85; Mozambique +0.79 / +0.81 / +0.83): the original input's partly-future rain was worth about +0.01 to +0.025, and known future rain about +0.02 to +0.03 more. The leak is real but small; it does not change any conclusion. For the ConvLSTM the ordering fails on Mozambique (causal +0.81, hybrid +0.81, forced +0.77) and is flat on the UK at 60 m (+0.85 / +0.87 / +0.85). The code review (F4) suspected the input layout (its own wording: "my hypothesis, unverified"): the forced rain is appended as five extra time steps after the observed history, which the recurrent state must carry; a variant that gives it the future rain as side channels is reported in 10.8.

**10.3 What the trained models use (test-time ablations, hybrid).** Normal / rain removed / terrain removed / depth history frozen / rain strictly past: ConvLSTM UK 60 m +0.87 / +0.86 / +0.87 / +0.02 / +0.86; local CNN UK 60 m +0.84 / +0.83 / +0.84 / -0.01 / +0.83; ConvLSTM Mozambique +0.81 / +0.79 / +0.81 / -0.01 / +0.79; local CNN Mozambique +0.81 / +0.79 / +0.81 / -0.01 / +0.78. Putting the depth history in the wrong order makes the skill strongly negative (-2.6 to -3.7). **The models use the depth history and almost nothing else** (H3 for the test-time version); rainfall is worth 0.01-0.02 on the never-seen region (0.02-0.07 on held-out tiles); the terrain nothing. (The `rain_shifted_cells` ablation is partly a no-op because of edge clipping and is not used.)

**10.4 Does spatial context matter?** Kernel 1 / 5 / 11 (a per-pixel network / 5x5 / 11x11), never-seen region: UK 60 m +0.85 / +0.84 / +0.84; UK 30 m +0.84 / +0.83 / +0.83; Mozambique +0.82 / +0.81 / +0.79 — **no benefit from looking at neighbouring pixels**; a per-pixel MLP (+0.85, +0.85, +0.83) does as well as any CNN. On the held-out Pakistan tiles the plain CNNs are unstable with the frozen batch-1 recipe (pooled skill +0.24 for kernel 1, +0.43 for kernel 5, -0.66 for kernel 11, with intervals that include 0), while the ConvLSTM (+0.84) and the per-pixel MLP (+0.87) are steady; whether that is a training-recipe effect is examined in 10.7.

**10.5 Does recurrence matter (H4, pre-registered rule: equivalent if |difference| < 0.02 and the interval lies inside +-0.03)?** ConvLSTM minus local CNN, never-seen region, paired block bootstrap: UK 60 m hybrid **+0.025 [+0.004, +0.043]** (ConvLSTM larger; causal +0.031), Mozambique hybrid +0.004 [-0.011, +0.019] (equivalent); Mozambique forced -0.069 (local larger; layout effect, see 10.2); UK 30 m hybrid +0.009 [-0.014, +0.031] and Mozambique causal +0.016 [-0.004, +0.034] favour the ConvLSTM by the rule's "larger wins" clause but their intervals include 0 (the report marks them "clstm larger (by the rule, but the interval includes 0)"). Per-pixel MLP versus ConvLSTM: the ConvLSTM is better only on the UK at 60 m (+0.02). **At the frozen 30-epoch recipe the ConvLSTM is ahead on one of the two independent pairs and equivalent on the other. But the plain CNN is slightly under-trained at 30 epochs (10.7). With BOTH models trained for 90 epochs (a fairness check added after seeing the CNN's 90-epoch gain) the ConvLSTM-minus-local difference is +0.004 [-0.021, +0.027] on the UK at 60 m and -0.014 [-0.029, +0.006] on Mozambique: equivalent by the pre-registered rule in both independent pairs. The recurrent CNN-LSTM is therefore NOT shown to beat a recurrence-free CNN once training length is equal; its skill is equivalent within the pre-registered margin (difference inside +-0.03).**

**10.6 Training-time ablations (H3, pre-registered rule: skill without depth history lower by more than the fold-to-fold sd).** Local CNN trained without one ingredient, never-seen region, hybrid (UK 60 m / Mozambique; full model +0.841 / +0.807): without the depth history **+0.172 / +0.086**; without depth AND rain history +0.159 / +0.085; without the rain history only +0.839 / +0.802; without any rain +0.835 / +0.798; without terrain +0.842 / +0.812. **H3 holds by a wide margin:** the depth history is worth +0.669 and +0.721 of skill (fold sd of the full model 0.003 and 0.008); rain is worth about 0.005-0.009 and terrain nothing. This is the same answer as the test-time ablations (10.3) and as the linear ladder (a linear model with rain and state but no depth trend reaches only +0.21 / +0.19 / +0.10 on the never-seen region for UK 60 m / UK 30 m / Mozambique).

**10.7 Strict space+time hold-out (H6), optimisation, training length: what limits the same-region hold-out score.** With a hidden time block as well as hidden tiles, the same-region held-out score falls: E60 local +0.756 -> +0.716 (within the pre-registered 0.05), E60 ConvLSTM +0.782 -> +0.645, E30 local +0.730 -> +0.686 (within 0.05), E480 local +0.430 -> **-6.56**, E480 ConvLSTM +0.845 -> +0.742. H6 passes in 2 of 5 cells. (The strict split for the ConvLSTM at 30 m was not run: 30 m is the same storm pair as 60 m.) The failure sits in the windows at the very START of the flood (start index 0, absent from training in the fold whose time block contains it): 6 of the 480 strict held-out windows carry 98% of the plain CNN's squared error (in that fold the six start-0 windows carry a squared error of 33,901 against 1,262 for persistence, out of 34,294 against 3,769 for the whole fold; the single worst window alone is 14,853 against 780), and without them its strict score is +0.842 (ConvLSTM +0.866); in the tile-only hold-out 12 of 900 windows carry 82% of its error (+0.865 without them). Never-seen-region skill is unaffected by the strict split (E480 local +0.807 -> +0.803; E60 +0.841 -> +0.837). A further check separates "the data lack the onset" from "one training run went wrong": the plain CNN retrained on the Pakistan fold-0 strict split with four more seeds fails in the same way (held-out skill -6.9, -9.1, -8.1, -8.9; the frozen seed -8.1) and so does the batch-8 / lr 3e-3 recipe (-5.7), while the never-seen skill stays +0.76 to +0.79 in all of them. **The failure of the plain CNN reproduces across five seeds and a second recipe on this fold (-5.7 to -9.1), so it is not one unlucky run; that the cause is the missing onset is an inference: the ConvLSTM on the same fold scores +0.722, the lookup table +0.509, and the plain CNN's other two folds +0.848 and +0.738.** **Optimisation (batch 8, learning rate 3e-3, a deviation from the frozen recipe):** never-seen skill is 0.008-0.026 *lower* than under the frozen recipe (E60 local +0.829 vs +0.841, E60 ConvLSTM +0.853 vs +0.866, E30 local +0.822 vs +0.830, E480 local +0.798 vs +0.807, E480 ConvLSTM +0.785 vs +0.811), so a larger batch and a higher learning rate do not improve the never-seen score (the frozen recipe is not limited by its step size), but the frozen 30-epoch recipe does under-train the plain CNN in number of steps (see Training length); but the plain CNN's held-out Pakistan-tile score goes from +0.430 to **+0.771**, which suggests (one seed) that its tile-level instability on Pakistan is optimisation noise, not missing data ; the ConvLSTM on Mozambique is also slightly worse under the batch-8 recipe (+0.785 vs +0.811; held-out tiles +0.837 vs +0.845). **Training length:** 90 instead of 30 epochs raises the local CNN's never-seen skill by +0.021 (E60, +0.862), +0.008 (E30, +0.838) and +0.024 (E480, +0.831) and its Pakistan held-out score from +0.430 to +0.699 pooled (+0.383 to +0.678 as the mean of the three fold skills): the frozen 30-epoch recipe under-trains the plain CNN slightly ; the ConvLSTM gains almost nothing from 90 epochs (UK 60 m +0.866 -> +0.866, Mozambique +0.811 -> +0.818; Mozambique held-out tiles +0.843 -> +0.861), so at equal length the two are level (10.5). **Reading:** the plain CNN's score on hidden Pakistan tiles is unstable across seeds (sd 0.33 over five seeds) while the ConvLSTM (+0.83 to +0.86) and the per-pixel MLP (+0.87) are steady; a model that has never seen a flood START cannot be trusted on one (under the strict split the lookup table stays at +0.66 and the ConvLSTM at +0.742; the per-pixel MLP (-2.65, mean of folds) and the plain CNN (-6.56) do not).

**10.8 Future rain as side channels (does the input layout explain the ConvLSTM's forced-rain deficit? code review F4).** Giving the ConvLSTM the five future rain maps as extra channels on its last hidden map instead of as five extra time steps lifts its forced-regime never-seen skill on Pakistan -> Mozambique from +0.765 (folds +0.763 / +0.780 / +0.752) to **+0.822** (+0.822 / +0.827 / +0.816) and on Australia -> UK at 60 m from +0.854 (+0.860 / +0.855 / +0.847) to **+0.880** (+0.876 / +0.885 / +0.880). The layout explanation is therefore supported: with the future rain side-loaded the ConvLSTM has the highest forced-regime skill on the UK (+0.880, against the per-pixel MLP +0.872 and the local CNN +0.866; differences of this size are within seed noise) and close to the local CNN on Mozambique (+0.834). Removing the known future rain at test time costs the side-channel model 0.06-0.08 (E60 +0.880 -> +0.803, E480 +0.822 -> +0.764) and the local CNN 0.06-0.13, so models trained with future rain do use it. Fold 1 of the UK experiment was aborted twice by the dead-network guard (see 10.15); finished with the guard disabled (same seed) it scored +0.885, and the two post-hoc retries (seed +1: aborted again; seed +2 with learning rate 5e-4: +0.865) are listed separately in the report. Not run at 30 m (same storm pair as 60 m).

**10.9 The original model (v1) under the same folds and recipe (hybrid input).** Never-seen skill per fold: UK 60 m +0.184 / +0.195 / +0.181 (mean +0.187), UK 30 m +0.181 / +0.193 / +0.189 (+0.188), Mozambique +0.060 / +0.063 / +0.062 (+0.061); held-out tiles +0.209 / +0.259 / +0.178, +0.174 / +0.220 / +0.184, +0.099 / +0.167 / +0.056. The final local CNN and ConvLSTM score +0.841 / +0.866, +0.830 / +0.840, +0.807 / +0.811 on the same tasks, i.e. **about 0.65 (UK) and 0.75 (Mozambique) higher**. The v1 figures agree, by a different code path (the study engine instead of the original notebooks), with the first study's typical UK skill of about +0.19 (its seven runs spanned +0.04 to +0.22, and identical configurations moved by up to 0.03 between machines). Test-time ablations of v1: removing the rain takes it to +0.003 / +0.003 / +0.002; freezing or scrambling the depth history changes nothing (+0.187 -> +0.187). So the earlier statement "the skill is rainfall-driven and the depth history is not used" is **correct for v1** and was wrongly generalised to "the CNN-LSTM": v1 cannot read the depth history (one 128-number bottleneck per frame, depth scaled by 5 m). The UK v1 runs were aborted by the dead-network guard at epoch 2-5 (a false alarm, 10.15) and finished with the guard disabled; the Mozambique runs ran with the guard on.

**10.10 How steady are the numbers? Seed repeats of the local CNN (five seeds per fold, hybrid).** The never-seen skill varies by only 0.006-0.010 between seeds within a fold (UK 60 m 0.007, UK 30 m 0.010, Mozambique 0.006) and by 0.001-0.008 between folds after averaging the seeds. **Differences smaller than about 0.02 in never-seen skill are therefore not resolvable**; the ConvLSTM's +0.025 edge over the local CNN on the UK at 60 m (30-epoch recipe) is only about 3 seed standard deviations and it disappears when the plain CNN is trained as long as the ConvLSTM (90 epochs: +0.004 [-0.021, +0.027]). The held-out-tile score is much noisier (sd 0.015 at 60 m, 0.027 at 30 m, **0.33 on Pakistan**: single seeds range from -0.37 to +0.77).

**10.11 Learning curve (does the number of training windows limit the models?).** Never-seen skill against 25 / 50 / 100 / 300 / 600 training windows (hybrid, mean of 3 folds). Local CNN: UK 60 m +0.760 / +0.794 / +0.823 / +0.841 / +0.861; UK 30 m +0.666 / +0.762 / +0.787 / +0.830 / +0.850; Mozambique +0.643 / +0.760 / +0.766 / +0.807 / +0.833 (still rising at 600). Per-pixel MLP: +0.829 / +0.857 / +0.859 / +0.846 / +0.868; +0.803 / +0.836 / +0.837 / +0.854 / +0.836; +0.809 / +0.807 / +0.831 / +0.829 / +0.844 (within about 0.04 of its 600-window score already with 25-50 windows). Lookup table: +0.710 -> +0.798, +0.623 -> +0.764, +0.489 -> +0.758. The straight-line trend does not improve (+0.365 -> +0.327 at 60 m). At 600 windows the plain CNN (+0.861 / +0.850 / +0.833) is level with the MLP (+0.868 / +0.836 / +0.844): **neighbouring pixels still add nothing**, but note that with the epochs fixed at 30 the number of optimiser steps grows with the number of windows, so this curve mixes 'more data' with 'more training' (separated by the equal-steps cells: with the number of optimiser steps held equal, doubling the data changes the never-seen skill by only -0.002 to +0.008 (UK 60 m +0.841 -> +0.847 at 9,000 steps and +0.863 -> +0.861 at 18,000; UK 30 m +0.830 -> +0.838 and +0.846 -> +0.850; Mozambique +0.807 -> +0.811 and +0.835 -> +0.833), i.e. inside seed noise: between 300 and 600 windows the plain CNN is limited by the amount of TRAINING, not by the number of windows (the equal-steps test was not made at 25-100 windows)); the per-pixel structure is learnable from a few dozen windows.

**10.12 Reverse-direction transfers (exploratory; train on the small region, test on the big one).** The small regions have no room for tile folds (Mozambique 1 valid 128-px tile, UK 30 m 2, UK 60 m none), so folds are hidden thirds of the START TIMES of the small region's corner tiles (four for Mozambique and for UK 30 m, two padded 85 x 85 squares for UK 60 m) (protocol in `completion/reverse_transfer.py`); the numbers are not comparable cell by cell with the forward experiments. Never-seen skill (hybrid; mean of 3 time folds): **UK -> Australia at 60 m** local CNN +0.738 [+0.571, +0.797], per-pixel MLP +0.659, lookup table +0.601; **at 30 m** local CNN +0.662 [+0.420, +0.746], MLP +0.588, table +0.606; both keep the ordering causal < hybrid < forced (60 m +0.702 / +0.738 / +0.772; 30 m +0.637 / +0.662 / +0.718). **Mozambique -> Pakistan at 480 m fails for the plain networks** (local CNN -0.884 [-2.665, +0.853], MLP -0.312), but only on the first windows of the Pakistan simulation: 54 of 3,240 window scores (start index 0) carry 93% of the local CNN's squared error, and without them it scores +0.834. In those windows the domain already holds 8-12 m of water that recedes to 2-7 m, a pattern absent from the small Mozambique flood, and the model's squared error in the ten worst windows of fold 0 is 4-11 times persistence's (over all 54 start-0 windows the ratio has median 7.7 and ranges from 0.3 to 27). The lookup table (+0.549, bounded bins) and the ConvLSTM (+0.444 [+0.277, +0.649]; +0.550 without the start windows, i.e. lower than the plain CNN on ordinary windows) are less exposed. On the training region's own hidden time thirds the local CNN scores +0.80 to +0.85 (pooled), the per-pixel MLP +0.83 to +0.88 and the ConvLSTM +0.76 to +0.85 (pooled); the lookup table (+0.64 to +0.74) and the linear baselines (negative to +0.05) are lower. Two caveats: the UK 60 m windows are padded squares (85 px of data in 128 px) and neither the training loss nor the baseline fits mask the padding, which moves the R60 baselines by 0.02-0.05 (the never-seen scoring itself is masked); and the ablations of the reverse runs use a 400-window subsample and are not tabulated. In this direction the ConvLSTM (hybrid) scores +0.695 [+0.499, +0.757] at 60 m and +0.584 [+0.399, +0.661] at 30 m, below the local CNN (by 0.04 at 60 m and 0.08 at 30 m, with overlapping intervals).

**10.13 Does capacity matter? The heavy comparator (v2lin: a ConvLSTM with a local path, 652,918 parameters, about 150 times the small models).** Same recipe, hybrid, 3 folds: never-seen skill UK 60 m **+0.840** [+0.790, +0.893] (small ConvLSTM +0.866, local CNN +0.841) and Mozambique **+0.802** [+0.758, +0.865] (+0.811 / +0.807); held-out tiles +0.770 and +0.582 [+0.242, +0.903] (the same Pakistan-tile instability as the plain CNN). Its test-time ablations match the small models' (depth history frozen +0.006 / -0.015, scrambled strongly negative). **A model 150 times larger does not do better: capacity is not what limits the skill.** (Run for UK 60 m and Mozambique only; 30 m is the same storm pair as 60 m.)

**10.14 Severity: augmentation, and the causal-rain regime.** The severity numbers of the first severity paragraphs (8b) were obtained with the original training (400 class-balanced windows per fold, drawn from the 41-42 tiles that are not held out, no augmentation). With the eight dihedral augmentations (identical for input and label) the capsule head's natural-mix accuracy rises from 0.678 to **0.803** and the dense head's from 0.602 to **0.798** (balanced accuracy 0.593 -> 0.697 and 0.479 -> 0.660). The capsule-minus-dense gap of +0.076 [+0.051, +0.101] shrinks to **+0.005 [-0.014, +0.022]**: the capsule design's earlier advantage disappears with augmentation, which is consistent with a small-sample regularisation effect (not tested directly) and gives no evidence that capsules suit the task. Both remain far below the trivial rules (last-frame threshold 0.974, forecast threshold 0.986; on the 47 class-changing windows the capsule is right in 22, the dense head in 27, the forecast threshold applied to their forecasts in 31 and 30). The rule layer still lowers accuracy after augmentation (capsule 0.803 -> 0.741, 8 fixed / 119 broken; dense 0.798 -> 0.751, 12 fixed / 96 broken; recalibrated thresholds 0.745 / 0.754); its stated depth-trend reasons stay true about 82-84% of the time. Grad-CAM for the depth forecast passes the deletion test in 84-92% of windows for all four models; the severity map remains weak for the capsule head (top beats both controls in 55-56%) and moderate for the dense head (73-75%). **Causal-rain regime (no look-ahead rain in the inputs; addresses the rain part of code-review finding F8 (the recalibration is still fitted on true rates and applied to predicted ones)):** without augmentation the capsule head scores 0.623 and the dense head 0.568 (hybrid: 0.678 / 0.602); the capsule-minus-dense gap is +0.056 [+0.029, +0.082]; the trivial thresholds are unchanged (0.974 last frame, 0.986-0.988 forecast); the rules again lower accuracy (capsule 0.623 -> 0.575, recalibrated 0.588; dense 0.568 -> 0.547 / 0.556) and their depth-trend reasons are true 84-85% of the time; Grad-CAM for the depth forecast beats both control masks in 93-97% of windows and the severity map in 57-77%. None of the severity conclusions depended on the partly-future rain in the hybrid input. (Severity under the forced regime was not run: there is no use case for a known-future-rain severity label.)

**10.15 The dead-network guard raised false alarms (a mistake found and fixed on 21 September).** The engine aborts a run when the predicted change map is identical across validation windows (per-pixel spread below 1e-5 m). It fired eight times: six original-model runs (UK 60 m and 30 m, epoch 2-5) and two side-channel runs (UK 60 m fold 1: the frozen seed at epoch 13 and a retry with the next seed at epoch 6). Seven of the eight were verified as **false alarms** by finishing them with the guard disabled and the same seed: the six original-model runs trained normally (+0.18 to +0.195) and the frozen-seed side-channel run scored +0.885; the eighth (the retry) was not re-run. Cause: the check used the 12 EARLIEST held-out windows (code review finding F13), in which the flood has barely started, so even a healthy network predicts nearly no change; slow-starting models (the original) look "dead" for the first epochs. Fix (21 September, 00:00; refined at about 05:50 to evenly spaced indices and no abort when fewer than two check windows exist): the check now samples 12 windows spread over the whole held-out period; it only changes the decision to abort (argued from the code; a bit-for-bit re-training comparison was not run), so runs that finished under the old check are unaffected by the change; whether they would also pass the new check was not tested (the 12 windows are taken at a stride and cover 84-92% of the held-out set). I first described the aborts as "collapses" in messages during the day; that was wrong and is corrected here and in the report table "Runs aborted by the dead-network guard".

### Severity, symbolic rules and Grad-CAM (hybrid regime; Australia 30 m; 3 folds pooled; details in the severity tables below)

**Capsule vs its control (the design hypothesis).** On the natural class mix (1,800 held-out windows) the capsule head is right 67.8% of the time and a plain dense head 60.2%: **capsule - dense = +0.076, 95% CI [+0.051, +0.101]**. The capsule design therefore does beat its own control on this task, by a modest margin. **Both lose badly to the trivial rule:** thresholding the last observed frame scores 97.4% and thresholding the model's own forecast 98.7%, because the label (maximum depth over the forecast in the patch) is almost fixed by the last frame. On the 47 windows where the class changes, the capsule is right in 22 (47%), the dense head in 24, the forecast-threshold in 34 (72%); the last-frame rule is wrong there by construction. The earlier statement "no evidence the head anticipates escalation" stands: it does not do better than thresholding the forecast.

**The symbolic rule layer now lowers accuracy.** With the original thresholds the rules change 161 of the capsule's answers: 25 fixed, 109 broken, accuracy 67.8% -> 63.1%; with thresholds re-derived on each fold's training windows 64.0% (51 fixed, 119 broken); for the dense head 60.2% -> 56.5% / 58.4%. (With the original weak model the net effect on accuracy was between -0.0 and +0.4 points in two full evaluations: 4 fixed / 4 broken and 2 fixed / 1 broken.) The rules' stated depth-trend reasons are true in about 84% of windows (the predicted trend has the same sign as the actual one; 'rising' claims 84%, 'receding' 91%). The reasons were not reviewed by a flood expert; the rule inputs use rainfall that is partly future in the hybrid regime and the recalibration is fitted on true rates but applied to predicted ones (code review F8).

**Grad-CAM, redesigned test.** For the depth forecast, deleting the top-10% heat pixels changes the predicted change maps more than **both** a wet/dry-matched random mask and the same mask moved elsewhere in 88% (capsule model) and 92% (dense model) of the windows in which the deletion had any effect (30-40 of the 40 fixed windows per fold; mean over the 3 folds); wet cells are hotter than dry ones in 79% / 76% of windows (median ratio 3.7 / 3.8). The heat maps of two models trained with different seeds agree moderately (median rank correlation 0.52 / 0.54): the hot spots are partly stable, partly not. For the severity class the map is weaker and less stable (top beats both controls in 55% of windows for the capsule head, 73% for the dense head; seed-to-seed rank correlation 0.17 for the capsule, 0.76 for the dense head). Grad-CAM for the depth forecast is a usable explanation; for the capsule's severity class it is not established.

### Severity, symbolic rules and Grad-CAM (re-evaluated on the new encoder; Australia 30 m)


#### Severity accuracy on held-out tiles (3 folds pooled; the natural set keeps the real class mix, the balanced set has 60 windows per class)

| regime | head | aug | folds | natural acc | balanced acc | last-frame threshold (nat / bal) | forecast threshold (nat) | class-changing windows: n, model right, forecast-threshold right |
|---|---|---|---|---|---|---|---|---|
| causal | capsule | n | 3 | 0.623 | 0.539 | 0.974 / 0.954 | 0.986 | 47, 24, 30 |
| causal | dense | n | 3 | 0.568 | 0.463 | 0.974 / 0.954 | 0.988 | 47, 20, 34 |
| hybrid | capsule | n | 3 | 0.678 | 0.593 | 0.974 / 0.954 | 0.987 | 47, 22, 34 |
| hybrid | capsule | y | 3 | 0.803 | 0.697 | 0.974 / 0.954 | 0.986 | 47, 22, 31 |
| hybrid | dense | n | 3 | 0.602 | 0.479 | 0.974 / 0.954 | 0.986 | 47, 24, 32 |
| hybrid | dense | y | 3 | 0.798 | 0.660 | 0.974 / 0.954 | 0.986 | 47, 27, 30 |

* capsule - dense accuracy (hybrid, natural windows, paired): +0.076 [95% CI +0.051, +0.101] over 1800 windows.
* capsule - dense accuracy (hybrid, augmented, natural windows, paired): +0.005 [95% CI -0.014, +0.022] over 1800 windows.
* capsule - dense accuracy (causal, natural windows, paired): +0.056 [95% CI +0.029, +0.082] over 1800 windows.

#### Symbolic rule layer (natural held-out windows, pooled over folds)

| regime | head | thresholds | accuracy before -> after | windows changed | fixed a wrong answer | broke a right one | rules triggered |
|---|---|---|---|---|---|---|---|
| causal | capsule | original | 0.623 -> 0.575 | 153 | 19 | 106 | 889 |
| causal | capsule | recalibrated | 0.623 -> 0.588 | 196 | 50 | 114 | 1170 |
| causal | dense | original | 0.568 -> 0.547 | 156 | 35 | 72 | 863 |
| causal | dense | recalibrated | 0.568 -> 0.556 | 185 | 55 | 77 | 1141 |
| hybrid | capsule | original | 0.678 -> 0.631 | 161 | 25 | 109 | 861 |
| hybrid | capsule | recalibrated | 0.678 -> 0.640 | 203 | 51 | 119 | 1147 |
| hybrid | capsule+aug | original | 0.803 -> 0.741 | 138 | 8 | 119 | 876 |
| hybrid | capsule+aug | recalibrated | 0.803 -> 0.745 | 184 | 32 | 136 | 1164 |
| hybrid | dense | original | 0.602 -> 0.565 | 179 | 41 | 107 | 849 |
| hybrid | dense | recalibrated | 0.602 -> 0.584 | 241 | 83 | 115 | 1135 |
| hybrid | dense+aug | original | 0.798 -> 0.751 | 116 | 12 | 96 | 866 |
| hybrid | dense+aug | recalibrated | 0.798 -> 0.754 | 142 | 27 | 106 | 1156 |

Are the rule reasons TRUE statements? (the stated depth trend is the model's predicted patch-mean rate; 'true' = same sign as the actual rate)

| regime | head | sign agreement predicted vs actual trend | 'rising' claims true | 'receding' claims true |
|---|---|---|---|---|
| causal | capsule | 0.847 | 0.837 | 0.944 |
| causal | dense | 0.839 | 0.835 | 0.950 |
| hybrid | capsule | 0.843 | 0.841 | 0.906 |
| hybrid | capsule+aug | 0.838 | 0.835 | 0.882 |
| hybrid | dense | 0.842 | 0.838 | 0.927 |
| hybrid | dense+aug | 0.822 | 0.819 | 0.939 |

#### Grad-CAM (40 fixed held-out windows per model, at the shared encoder's hidden map; hot pixels = the evidence for the target)

Two maps: 'depth' (target = predicted depth summed over all leads) and 'severity' (target = the predicted class score of the capsule / dense head). Deletion removes the depth + rain history at the top-10% (positive-heat) pixels and is compared with (a) random pixels of the same wet/dry mix and (b) the same mask moved 40 px.

| regime | head | map | wet hotter than dry (share of windows) | median wet/dry ratio | mean effect: top / random / moved | top beats random | top beats both controls |
|---|---|---|---|---|---|---|---|
| causal | capsule | depth | 0.77 | 4.64 | 116.505 / 77.382 / 32.232 | 0.93 | 0.93 |
| causal | capsule | severity | 0.74 | 3.72 | 0.215 / 0.057 / 0.028 | 0.67 | 0.57 |
| causal | dense | depth | 0.78 | 4.83 | 114.485 / 72.355 / 29.006 | 0.97 | 0.97 |
| causal | dense | severity | 0.74 | 4.98 | 16.538 / 5.086 / 2.600 | 0.80 | 0.77 |
| hybrid | capsule | depth | 0.79 | 3.66 | 121.714 / 82.260 / 35.255 | 0.89 | 0.88 |
| hybrid | capsule | severity | 0.61 | 2.66 | 0.188 / 0.049 / 0.015 | 0.64 | 0.55 |
| hybrid | capsule+aug | depth | 0.83 | 4.96 | 120.916 / 84.181 / 33.597 | 0.94 | 0.92 |
| hybrid | capsule+aug | severity | 0.64 | 2.61 | 0.182 / 0.065 / 0.021 | 0.65 | 0.56 |
| hybrid | dense | depth | 0.76 | 3.77 | 112.219 / 80.595 / 32.682 | 0.92 | 0.92 |
| hybrid | dense | severity | 0.75 | 4.81 | 30.250 / 10.374 / 4.443 | 0.76 | 0.73 |
| hybrid | dense+aug | depth | 0.74 | 3.32 | 113.444 / 82.950 / 33.975 | 0.85 | 0.84 |
| hybrid | dense+aug | severity | 0.77 | 3.61 | 15.409 / 7.557 / 2.624 | 0.80 | 0.75 |

* Grad-CAM stability across two training seeds (capsule head, fold 0, same 40 windows, depth map): median Spearman correlation 0.52 (25-75%: 0.34 .. 0.70).
* Grad-CAM stability across two training seeds (capsule head, fold 0, same 40 windows, severity map): median Spearman correlation 0.17 (25-75%: -0.20 .. 0.82).
* Grad-CAM stability across two training seeds (dense head, fold 0, same 40 windows, depth map): median Spearman correlation 0.54 (25-75%: 0.23 .. 0.72).
* Grad-CAM stability across two training seeds (dense head, fold 0, same 40 windows, severity map): median Spearman correlation 0.76 (25-75%: 0.51 .. 0.85).

## 11. Course Outcome (CO) Mapping

CO1–CO5: **Dense (CO1)** · **Classification head (CO2)** · **CNN enc/dec (CO3)** · **Dropout/LR (CO4)** · **LSTM enc-dec (CO5)**

## 12. Decided, Built, Not Established

**Established (measured; the numbers are in Section 10 and in `PROJECT_LOG.md`)**

*Data and alignment.* Every one of the 14,868 dataset files reads cleanly; the geometry of the 60 m and 480 m grids for all four regions (Mozambique strongly, Pakistan more weakly); georeferencing lines up with real maps (Australia 30 m satellite check; Mozambique, Pakistan and the UK on OpenStreetMap); **the rainfall timing** - file `j` is the rain falling during `[1800 j, 1800 (j+1))` s, measured in all four regions (correlation 1.00 at lag 0, slope 0.494-0.502) and independently re-derived from the raw files.

*The depth forecast.* Under a frozen, pre-registered recipe (3 tile folds, block-bootstrap intervals), a 4,101-parameter local CNN and a 4,741-parameter ConvLSTM ("the CNN-LSTM") beat persistence on regions they were never trained on: skill **+0.81 to +0.87** (UK 60 m +0.84 / +0.87, UK 30 m +0.83 / +0.84, Mozambique +0.81 / +0.81), in all three rainfall regimes for the local CNN (+0.79 to +0.87) and for the ConvLSTM on UK 60 m and Mozambique (+0.77 to +0.87; the 30 m ConvLSTM was run in the hybrid regime only), with a seed-to-seed noise of about 0.01. Never-seen-region skill is unchanged by a strict space-and-time hold-out (E60 +0.841 -> +0.837; E480 +0.807 -> +0.803). The original model (v1) scores +0.18 to +0.19 on the UK and +0.06 on Mozambique on the same folds (consistent with the first study's typical +0.19).

*What the skill is made of.* A two-feature lookup table (current depth x last one-step change) reaches +0.74 to +0.79 and a per-pixel MLP +0.83 to +0.85. Removing the depth history at training time drops the local CNN from +0.84 / +0.83 / +0.81 to +0.17 / +0.15 / +0.09 (H3 holds by 0.67-0.72 against a fold sd of 0.003-0.013); removing rain costs 0.005-0.015 and terrain nothing; at test time freezing the history takes the skill to about zero and scrambling it makes it strongly negative (-1.1 to -7.8; the exceptions are the per-pixel CNN on the UK, +0.12 to +0.16, and the original model, which ignores the history). The forecast is, to first order, **a nonlinear per-pixel extrapolation of the recent depth trajectory**. Neighbouring pixels add nothing (kernel 1 / 5 / 11; once the plain CNN is trained long enough it is level with the per-pixel MLP). **Recurrence adds nothing measurable at equal training length**: the ConvLSTM's +0.025 edge on the UK at 60 m under the 30-epoch recipe becomes +0.004 [-0.021, +0.027] when both models are trained for 90 epochs (Mozambique -0.014 [-0.029, +0.006]).

*Rainfall.* The original input contained up to 30 minutes of future rain in the last frame; under strictly past rain the local CNN loses +0.01 to +0.025, and knowing the future rain gains +0.02 to +0.03 (H5 holds for the local CNN). The ConvLSTM's forced-rain deficit is supported as an input-layout effect: with the future rain as side channels it gains +0.06 (Mozambique, +0.765 -> +0.822) and +0.026 (UK 60 m, +0.854 -> +0.880).

*The original model.* v1 uses the rain and cannot read the depth history (rain removed: +0.003; history frozen or scrambled: no change). The earlier finding "rainfall drives the skill and the depth history is not used" is true of v1 and was wrongly generalised.

*Robustness checks.* A larger batch and learning rate (batch 8 / lr 3e-3) do not help the never-seen score (-0.008 to -0.026), but the plain CNN is under-trained at 30 epochs: 90 epochs give +0.008 to +0.024. The plain CNN's score on hidden Pakistan tiles is optimisation-noisy (seed sd 0.33); its failure under the strict space-and-time split on Pakistan (fold 0) reproduces across seeds and recipes (all five seeds and the alternative recipe fail: -5.7 to -9.1) and sits in the windows at the very start of the flood, which are absent from training (6 of 480 windows carry 98% of the error; without them the strict score is +0.84 for the plain CNN and +0.87 for the ConvLSTM). The learning curve shows the per-pixel model is within about 0.04 of its 600-window score with 25-50 training windows; the plain CNN keeps improving with more windows mainly because it then trains for more steps (between 300 and 600 windows, with the steps held equal, doubling the data changes its skill by -0.002 to +0.008). A 150-fold larger ConvLSTM scores no better (UK 60 m +0.840, Mozambique +0.802), so capacity is not the limit either.

*Reverse-direction transfers (exploratory).* Training on the small UK flood and testing on Australia works for the plain CNN (+0.74 at 60 m, +0.66 at 30 m; ConvLSTM +0.70 / +0.58); training on Mozambique and testing on Pakistan fails for plain networks (-0.88) but only on the first windows of the Pakistan simulation (93% of the error sit in 54 of 3,240 window scores; +0.83 without them); the ConvLSTM (+0.44) and the lookup table (+0.55) are less exposed.

*Persistence vs the paper.* On the paper's own whole-domain protocol persistence scores NSE 0.9995 / CSI(1 mm) 0.983 on Mozambique (FNO+ 0.955 / 0.934) and 0.9993 / 0.980 on the UK at 60 m (0.9989 / 0.860); the paper reports no persistence baseline. Our task differs from the paper's; numbers are not comparable.

*Severity, rules and Grad-CAM (Australia 30 m, hybrid, 3 folds pooled).* The capsule head beats its dense control by +0.076 [+0.051, +0.101] accuracy on the natural mix without augmentation (67.8% vs 60.2%); **with augmentation both reach about 80% and the gap is +0.005 [-0.014, +0.022]**, so the capsule's advantage is consistent with a small-sample effect. Both are far below the trivial rules (97.4% last-frame threshold, 98.6-98.7% forecast threshold). The symbolic rule layer lowers accuracy with and without augmentation (0.678 -> 0.631; 0.803 -> 0.741); its stated depth-trend reasons are true about 84% of the time. Grad-CAM for the depth forecast points at cells that matter more than random or displaced masks in 84-92% of windows; for the severity class it is weak. The same conclusions hold when the inputs contain no look-ahead rain (causal regime: capsule 0.623, dense 0.568, gap +0.056 [+0.029, +0.082]; rules 0.623 -> 0.575).

*Research gap.* The paper's benchmark has no LSTM/CNN-LSTM model (checked against its text).

**Not established, or not supported**

- **Skill on a storm the models have not seen from a region they trained on.** The dataset has exactly four events; every region is ONE flood. E60 and E30 share the UK event and E480 is a separate pair, so there are **two independent train -> test pairs**, not sixteen results. Intervals are block bootstraps and are wider than naive ones for this reason.
- **Skill when a flood's onset was never in the training data.** With a strict space-and-time split the same-region held-out score falls (E60 local +0.756 -> +0.716; ConvLSTM +0.782 -> +0.645; Pakistan local +0.430 -> -6.56, ConvLSTM +0.845 -> +0.742; the lookup table stays +0.66). H6 passes in 2 of 5 cells. The Pakistan failure of the plain CNN reproduces across seeds and recipes on fold 0.
- **That the CNN-LSTM's recurrence is needed or helps.** At equal training length the ConvLSTM and the recurrence-free CNN are equivalent on both independent pairs.
- **Whether the capsule severity head suits the task.** Without a small-sample advantage it equals a plain dense head, and both lose to thresholding the last observed frame; on the 47 windows where the class changes they are right in 22-27 against 30-34 for thresholding the model's own forecast.
- **That the symbolic rules help.** They lower accuracy; their reasons were not reviewed by a flood expert; the recalibrated thresholds are fitted on true rates and applied to predicted ones.
- **That the depth model is physical.** It is a statistical surrogate of the simulator's output for one flood per region: it extrapolates the simulator's trajectory; it has not been tested against observed inundation, and nothing here says it would behave correctly for a different rainfall pattern.
- **Crop offsets to sub-cell precision** (the independent verifier could not confirm them; supported by the terrain probe, the NoData ring and the map overlays); Pakistan's alignment beyond about half a cell.
- **A cross-hardware check.** The whole study ran on one CPU laptop; the Colab notebook is a smaller demonstration of the same code. Numbers from another machine will differ in the last decimals (a same-seed rerun on this machine is documented in `completion/results/repro_check.json`).
- **Cells dropped after analysis (redundant; reasons in the round-2 section):** training-time ablations of the ConvLSTM, and the 30 m ConvLSTM strict-split, side-channel and optimisation cells (30 m is the same storm pair as 60 m), and severity under the forced-rain regime (not run: there is no use case for a known-future-rain severity label).
- **Deviations from the pre-registration (disclosed):** baselines were re-run with the per-pixel MLP, the lookup table and clipped variants after the first ladder; the kernel batch, optimisation batch, side-channel batch, strict-split ConvLSTM batch, learning curve, 90-epoch batches, strict-split seed reruns, reverse transfers and guard-disabled reruns were added after seeing results (exploratory / post hoc; also the equal-steps learning-curve cells and the two post-hoc retries of the side-channel fold 1); the dead-network guard's window sampling was changed after it raised false alarms.
- **Independence of the design phase.** Four non-blind design runs (the v2 family) printed their UK 60 m score during design; the two chosen architectures (local CNN, ConvLSTM) have only blind design runs, and the UK ranking of the heavy runs matched their held-out ranking, so no decision plausibly changed. The E60 fold-0 held-out windows are the windows used to choose the design (same seed), so that one held-out score is not independent of it; the never-seen scores are (code review F10).
- **Independence of the hold-outs.** In the reverse-direction transfers the training region's hold-out is a hidden time block of the SAME tiles (no spatial hold-out) and the embargo between training and hidden windows is one window length (the flood state is continuous across the boundary); in the forward experiments the tile hold-out is spatial and the strict split hides a time block as well. All skills are computed on windows that are wet at the last observed or the last forecast frame (windows dry in both are dropped, for model and persistence alike); two Australia 30 m tiles carrying at most 0.43 m of water are excluded by the tile filter.
- **Run provenance.** The result files carry six different engine hashes (136 / 82 / 30 / 27 / 23 / 12 runs; the dead-network check was changed twice, the newest 27 runs were written after `engine.py` was edited on 21 September at 05:48, and the batch-size, learning-rate and guard switches were added while the queue ran). The training mathematics of the default path is unchanged (seed repeats made with the current code scatter around the frozen runs about as much as the seeds scatter among themselves: the mean of the four repeats minus the frozen run is between -0.008 and +0.010 in 8 of 9 folds and +0.022 in UK 30 m fold 2; single repeats differ by up to 2.6 seed standard deviations), but the headline UK 60 m runs used a version of `model_v2.py` that no longer exists on disk and could not be diffed. A same-seed re-run of one frozen headline run (UK 60 m, local CNN, fold 0) with the CURRENT code, threads and machine state reproduces the stored result to 9e-6 (never-seen skill +0.83794 vs +0.83793) and 5e-5 (held-out), so the default path is unchanged (`completion/results/repro_check.json`).
- **A leftover run.** One E30 ConvLSTM causal fold-0 run from an early attempt is in `completion/results/runs`; it is part of no table (the report's data-warning section lists it as incomplete (fold 0 only)).

## 13. Where Everything Is

- **This folder:** the updated deck (`..._v2.pptx`), this explainer, and `architecture_diagram/` (the diagram of the *original* model, kept as the record; the final architecture figure is in the code handoff at `completion/results/architecture_v3.png`). The original deck and explainer are unchanged.
- **Code handoff** (`flood_forecasting_handoff.zip`, and `C:\Dev\flood-forecast-demo` on Anhad's laptop): `PROJECT_LOG.md` (full record with corrections), `HANDOFF_README.md`, **`flood_forecasting_study_colab.ipynb` (the current study: run on Colab, Runtime → GPU → Run all)**, `flood_forecasting_colab.ipynb` and `flood_forecasting_multires_colab.ipynb` (the ORIGINAL model, marked superseded), the study engine and analysis scripts in `completion/`, the pre-registration `completion/PREREGISTRATION.md`, the two independent reviews in `completion/review/`, and every result file and log.

## 14. Independent Audits and Corrections (September 2026)

**Round 1 (19 September) — re-checking the original pipeline from the raw files.** Passed: raw rasters clean inside the aligned crop; the train/validation split shares no tile; 24 random windows match independent reads exactly; the headline skill recomputes identically; the metric code returns known answers; an untrained model equals persistence exactly. Found and fixed: a NoData padding ring in the UK layers (never read by any window; the loader now refuses it); the claim "the LSTM captures temporal context" was untested; "Pakistan and Mozambique cannot be aligned" was wrong and the pipeline had used only half of the dataset (13.5 of 26.9 GB) — both regions and the 60 m data were then recovered and used.

**Round 2 (20 September) — questioning the results themselves.** Doing the remaining "not established" items exposed larger problems:
1. **Rainfall timing.** Physical test in all four regions: file *j* = rain falling *during* `[1800 j, 1800 (j+1))` s (depth gain = 0.494–0.502 × the file value; correlation 1.00 at lag 0). The original input therefore carried up to 30 minutes of not-yet-fallen rain in the last frame. The earlier claim "no future information" was wrong for rainfall. Effect on skill turned out small (causal vs hybrid: about 0.01–0.03), but three regimes (causal / hybrid / forced) are now tested everywhere.
2. **The original model was weak, and I had never compared it with a trend baseline.** A per-pixel linear model of the depth history scored about twice the original CNN-LSTM; a lookup table on (current depth, last change) and a per-pixel MLP score far higher still. The original's inability to use the depth history (flatten + 128-number bottleneck, depth scaled by 5 m) had produced the earlier finding "the forecast is last frame + rainfall" — an **artefact**, now corrected.
3. **The paper's metric.** Its "RMSE" is a relative error, not metres; and copying the initial frame matches or beats its best model on most of its own measures (the paper has no such baseline).
4. **Physical alignment check.** A water-surface probe failed its own validation (reported as a negative result); the map overlays confirm the coordinate frame for Mozambique, Pakistan and the UK; the crop offsets themselves are supported but could not be independently re-derived by the verifier.

**How the new results were protected against mistakes.** (a) Protocol and hypotheses written down before the final runs (`completion/PREREGISTRATION.md`, checksum recorded); model design used only held-out tiles of the training region. (b) **Independent verification by a separate agent that used no project code:** rain lag/slope, persistence errors on the never-seen windows (Mozambique 238 windows, 17.0757 mm; UK at 60 m 120 windows, 15.7004 mm; agreement ~10⁻⁸), bit-exact equality of 100 random cached windows per pool with the raw frames, and the paper-protocol persistence numbers all **matched**. (c) **Adversarial code review by a second agent:** no critical finding (no future or target depth reaches an input, no train/held-out tile or time-block overlap, padded pixels masked, metrics and bootstrap statistics correct). Five major findings were fixed: the aggregation would have silently merged non-frozen runs into headline cells; the confidence intervals treated overlapping windows as independent (now block bootstraps, wider); two hypothesis verdicts were coded looser than pre-registered; a Grad-CAM deletion test measured a different quantity than its map and never involved the severity head (redesigned); and the forced-regime input layout handicaps the recurrent model (disclosed; a variant with the future rain as side channels was added). Minor findings are disclosed in `PROJECT_LOG.md` (partial no-op of one ablation; 8 m terrain shift from padding the small UK domain, with no measurable effect; each region is a single event).

**Mistakes made during the project, all corrected** (full list in `PROJECT_LOG.md`: the audit sections and "Completion, round 2", section 10): asserting layer alignment after checking one region; blaming data volume for an architecture and loss problem; a rule-engine "success" fed by noise; an invalid first notebook; an unchecked map-tile fix; quoting single-seed numbers as stable; a one-window Grad-CAM claim; adopting augmentation on one seed; describing the model as temporal without testing it; **dropping two regions and calling the project complete while half the dataset was unused (and auditing only the half in use)**; leaving a latent training-collapse mode in the recipe; **telling you no future information entered any input (wrong for rainfall)**; **never comparing the original model with a trend baseline, so its weakness went unnoticed**; **concluding "rainfall drives the skill / the history is not used" from a model that could not read the history**; **writing the paper's relative error as metres**; **a dead-network guard that falsely aborted eight runs because it looked only at the 12 earliest held-out windows, and my calling those aborts "collapses" (Section 10.15)**; **a learning curve that confounded more data with more training steps**; and, caught by smoke tests before use, a Grad-CAM deletion measure that counted the removed water itself and an earlier version of the guard that mistook sparse changes for a collapsed model.

---

*If the architecture or the results change again, regenerate this file with `completion/build_explainer.py` rather than hand-editing, to avoid drift.*
