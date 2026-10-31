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
a lookup table keyed on just two numbers per pixel (current depth, last one-step change) scores about +0.79 on the never-seen UK (60 m) and +0.74 on Mozambique; a per-pixel MLP (no convolution, no recurrence) about +0.85 and +0.83; the small local CNN +0.84 and +0.81; the small ConvLSTM **+0.87 and +0.81** — against +0.32 / +0.15 for a straight-line (linear) extrapolation and roughly +0.05 to +0.2 for the original CNN-LSTM. Spatial and recurrent structure adds little: the ConvLSTM is significantly better than the local CNN only on the UK at 60 m (+0.025 [+0.004, +0.043] in the hybrid regime), equivalent on Mozambique, and *no* map-aware model beats the per-pixel MLP except the ConvLSTM on the UK at 60 m (+0.02).


### Result tables (generated from the result files)


#### E60: Australia -> UK, 60 m  (skill vs persistence; mean over 3 tile folds; [95% CI]: bootstrap over start times / tiles)

| model | causal: never-seen test | hybrid: never-seen test | forced: never-seen test | causal: held-out tiles (pooled) | hybrid: held-out tiles (pooled) | forced: held-out tiles (pooled) |
|---|---|---|---|---|---|---|
| persistence (copy the last frame) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) |
| linear: depth trend only | +0.323 | +0.323 | +0.323 | +0.303 | +0.303 | +0.303 |
| linear: rain + state + trend (>=0) | +0.334 | +0.345 | +0.365 | +0.357 | +0.376 | +0.430 |
| 2-feature lookup table (depth x last change) | +0.786 | +0.786 | +0.786 | +0.702 | +0.702 | +0.702 |
| per-pixel MLP | +0.839 | +0.846 | +0.872 | +0.791 | +0.798 | +0.847 |
| **local CNN 5x5, 4.1k params** | +0.822 [+0.780, +0.878] | +0.841 [+0.798, +0.892] | +0.866 [+0.814, +0.914] | +0.744 [+0.641, +0.808] | +0.756 [+0.660, +0.814] | +0.817 [+0.715, +0.883] |
| **small ConvLSTM (CNN-LSTM), 4.7k params** | +0.853 [+0.796, +0.905] | +0.866 [+0.808, +0.915] | +0.854 [+0.785, +0.911] | +0.764 [+0.670, +0.822] | +0.782 [+0.686, +0.840] | +0.808 [+0.708, +0.873] |

#### E30: Australia -> UK, 30 m  (skill vs persistence; mean over 3 tile folds; [95% CI]: bootstrap over start times / tiles)

| model | causal: never-seen test | hybrid: never-seen test | forced: never-seen test | causal: held-out tiles (pooled) | hybrid: held-out tiles (pooled) | forced: held-out tiles (pooled) |
|---|---|---|---|---|---|---|
| persistence (copy the last frame) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) |
| linear: depth trend only | +0.345 | +0.345 | +0.345 | +0.336 | +0.336 | +0.336 |
| linear: rain + state + trend (>=0) | +0.347 | +0.359 | +0.379 | +0.357 | +0.370 | +0.441 |
| 2-feature lookup table (depth x last change) | +0.763 | +0.763 | +0.763 | +0.676 | +0.676 | +0.676 |
| per-pixel MLP | +0.841 | +0.854 | +0.869 | +0.766 | +0.775 | +0.817 |
| **local CNN 5x5, 4.1k params** | +0.805 [+0.750, +0.854] | +0.830 [+0.777, +0.876] | +0.851 [+0.790, +0.904] | +0.723 [+0.621, +0.794] | +0.730 [+0.649, +0.803] | +0.799 [+0.712, +0.863] |
| **small ConvLSTM (CNN-LSTM), 4.7k params** | - | +0.840 [+0.764, +0.904] | - | - | +0.768 [+0.687, +0.827] | - |

#### E480: Pakistan -> Mozambique, 480 m  (skill vs persistence; mean over 3 tile folds; [95% CI]: bootstrap over start times / tiles)

| model | causal: never-seen test | hybrid: never-seen test | forced: never-seen test | causal: held-out tiles (pooled) | hybrid: held-out tiles (pooled) | forced: held-out tiles (pooled) |
|---|---|---|---|---|---|---|
| persistence (copy the last frame) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) | 0 (by definition) |
| linear: depth trend only | +0.151 | +0.151 | +0.151 | +0.146 | +0.146 | +0.146 |
| linear: rain + state + trend (>=0) | +0.170 | +0.182 | +0.202 | +0.195 | +0.219 | +0.261 |
| 2-feature lookup table (depth x last change) | +0.737 | +0.737 | +0.737 | +0.721 | +0.721 | +0.721 |
| per-pixel MLP | +0.835 | +0.829 | +0.855 | +0.874 | +0.868 | +0.916 |
| **CNN kernel 1 (per pixel)** | - | +0.818 [+0.774, +0.884] | - | - | +0.240 [-0.523, +0.906] | - |
| **local CNN 5x5, 4.1k params** | +0.794 [+0.757, +0.860] | +0.807 [+0.766, +0.870] | +0.834 [+0.790, +0.895] | +0.263 [-0.440, +0.885] | +0.430 [-0.113, +0.895] | +0.463 [-0.078, +0.948] |
| **small ConvLSTM (CNN-LSTM), 4.7k params** | +0.810 [+0.763, +0.874] | +0.811 [+0.761, +0.874] | +0.765 [+0.699, +0.844] | +0.840 [+0.794, +0.898] | +0.845 [+0.806, +0.901] | +0.815 [+0.743, +0.921] |

#### Pre-registered hypotheses

* **H1** (local and clstm beat persistence with both intervals above 0, every regime/experiment): 13/16 cells pass; not passing: local/E480/causal (test CI +0.757, held-out CI -0.440); local/E480/hybrid (test CI +0.766, held-out CI -0.113); local/E480/forced (test CI +0.790, held-out CI -0.078).
* **H2** (beat the best linear baseline S3c on the never-seen region): 16/16 cells pass.
* **H2b** (exploratory, added after seeing the per-pixel baseline: do the CNN/ConvLSTM beat the per-pixel MLP S4c on the never-seen region?): 2/16 cells; mean difference -0.020 (range -0.090 .. +0.019).
* **H4** (recurrence matters: clstm - local, never-seen region, paired block bootstrap; pre-registered rule): E60/causal +0.031 [+0.008, +0.052] -> clstm larger; E60/hybrid +0.025 [+0.004, +0.043] -> clstm larger; E60/forced -0.012 [-0.031, +0.002] -> local larger; E30/hybrid +0.009 [-0.014, +0.031] -> clstm larger; E480/causal +0.016 [-0.004, +0.034] -> clstm larger; E480/hybrid +0.004 [-0.011, +0.019] -> equivalent; E480/forced -0.069 [-0.095, -0.038] -> local larger.
* **H5** (causal <= hybrid <= forced on the never-seen region, means): 3/5 (arch, experiment) cases hold. Caution (code review F4): the recurrent model reads the forced rain as five extra steps AFTER the observed history, which handicaps it; a failure for clstm is an input-layout effect, not evidence about rain physics.

#### What the trained models use (test-time ablations, never-seen region, hybrid regime, mean over folds)

| experiment | model | none | rain_removed | terrain_removed | depth_hist_scrambled | depth_hist_frozen | rain_hist_frozen | rain_shifted_cells | rain_shifted_earlier |
|---|---|---|---|---|---|---|---|---|---|
| E60 | local CNN 5x5, 4.1k params | +0.841 | +0.833 | +0.841 | -3.037 | -0.006 | +0.834 | +0.841 | +0.833 |
| E60 | small ConvLSTM (CNN-LSTM), 4.7k params | +0.866 | +0.857 | +0.865 | -2.649 | +0.020 | +0.863 | +0.866 | +0.863 |
| E30 | local CNN 5x5, 4.1k params | +0.830 | +0.810 | +0.830 | -1.393 | +0.010 | +0.825 | +0.830 | +0.819 |
| E30 | small ConvLSTM (CNN-LSTM), 4.7k params | +0.840 | +0.827 | +0.839 | -1.141 | +0.012 | +0.837 | +0.840 | +0.832 |
| E480 | CNN kernel 1 (per pixel) | +0.818 | +0.812 | +0.818 | -2.856 | -0.018 | +0.812 | +0.818 | +0.801 |
| E480 | local CNN 5x5, 4.1k params | +0.807 | +0.790 | +0.807 | -3.666 | -0.008 | +0.798 | +0.807 | +0.784 |
| E480 | small ConvLSTM (CNN-LSTM), 4.7k params | +0.811 | +0.791 | +0.811 | -2.929 | -0.007 | +0.797 | +0.811 | +0.791 |

#### Training-time ablations (local model; never-seen region skill, mean over folds)

| experiment | regime | full | nodepthhist | norainhist | nohist | norain | noterrain |
|---|---|---|---|---|---|---|---|

[PENDING: completion/results/explainer_section10_extra.md not generated yet]
### Severity, symbolic rules and Grad-CAM (re-evaluated on the new encoder; Australia 30 m)

[PENDING: completion/results/SEVERITY_REPORT.md not generated yet]

## 11. Course Outcome (CO) Mapping

CO1–CO5: **Dense (CO1)** · **Classification head (CO2)** · **CNN enc/dec (CO3)** · **Dropout/LR (CO4)** · **LSTM enc-dec (CO5)**

[PENDING: completion/results/explainer_section12.md not generated yet]

## 13. Where Everything Is

- **This folder:** the updated deck (`..._v2.pptx`), this explainer, and `architecture_diagram/` (the diagram of the *original* model, kept as the record; the final architecture figure is in the code handoff at `completion/results/architecture_v3.png`). The original deck and explainer are unchanged.
- **Code handoff** (`flood_forecasting_handoff.zip`, and `C:\Dev\flood-forecast-demo` on Anhad's laptop): `PROJECT_LOG.md` (full record with corrections), `HANDOFF_README.md`, **`flood_forecasting_study_colab.ipynb` (the current study: run on Colab, Runtime → GPU → Run all)**, `flood_forecasting_colab.ipynb` and `flood_forecasting_multires_colab.ipynb` (the ORIGINAL model, marked superseded), the study engine and analysis scripts in `completion/`, the pre-registration `completion/PREREGISTRATION.md`, the two independent reviews in `completion/review/`, and every result file and log.

[PENDING: completion/results/explainer_section14.md not generated yet]

---

*If the architecture or the results change again, regenerate this file with `completion/build_explainer.py` rather than hand-editing, to avoid drift.*
