# CNN-LSTM Based Spatiotemporal Flood Evolution Forecasting

**With capsule-based severity classification and a post-hoc NeuroSymbolic reasoning layer**

Anhad Mahajan (PRN: 24070126031) & Aman Sumeet Saxena (PRN: 24070126030) — DL Lab Project

This document explains the project end to end: the problem, the dataset, the exact architecture as built, the pipeline, the tech stack, **what was actually measured**, and what is still not established. It reflects the verified state after the full-size runs (September 2026) and matches `CNN-LSTM_Based_Spatiotemporal_Flood_Evolution_Forecasting_v2.pptx`. The numbers come from `PROJECT_LOG.md` in the code handoff (`flood_forecasting_handoff.zip`), which is the source of truth; if the two ever disagree, the log wins.

## What changed since the earlier explainer

The earlier version described the plan before anything was built. Building and measuring changed these points:

| Earlier explainer said | Now (measured) |
|---|---|
| Cross-regional test: train on Pakistan/Australia, test on Mozambique/UK | **The paper's three same-resolution transfer tests:** Australia → UK at 30 m and at 60 m, Pakistan → Mozambique at 480 m (depth only at 60 m and 480 m). Mixing resolutions in one model is not attempted (a 128-px patch spans 3.8 km, 7.7 km or 61 km). Pakistan and Mozambique were first dropped as "not alignable" — **that was wrong**: their geometry was recovered with a validated probe and cross-checked (Section 3). |
| Severity head: "4-way softmax" | The capsule head outputs **one capsule length per class, trained with the CapsNet margin loss**; the prediction is the longest capsule. Softmax over bounded lengths caps at ~0.475 and collapsed to one class. |
| CNN Decoder: Conv2DTranspose | **Persistence-anchored decoder:** output = last observed depth frame + a learned change (zero-initialised, so an untrained model equals "copy the last frame"), with a skip connection from the encoder. |
| Evaluate against U-Net / FNO / FNO+ | **Every result is reported against persistence** (depth) and **threshold-the-last-frame** (severity). Comparison with U-Net/FNO under the paper's whole-scene protocol was not done. |
| 5-minute steps | **30-minute steps** (rainfall's native resolution). At 5 minutes the rainfall channel is constant and copying the last frame is right to 2 mm. |
| Tech stack: Pandas, scikit-learn, Plotly, Mapbox | Not used. The stack is Python, TensorFlow/Keras, NumPy, SciPy, rasterio, Matplotlib, folium, Hugging Face Hub, Colab. |
| "Physics-informed constraints" (deck subtitle) | There is no physics loss. The symbolic layer is post-hoc, rule-based and has no gradients. |
| "We are the first to add a CNN-LSTM result" | Reworded to "to our knowledge"; our result is patch-based and **not directly comparable** to the paper's whole-scene baselines. |
| Pending: capsule not built, rules not defined, cross-regional test not run | All built and run. Results below. |

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
- **Rainfall ↔ depth alignment is an inference.** Nothing in the archive says how they line up. For Australia, rainfall's 481 files × 30 min span exactly the 10 days depth spans, and the paper states the simulation period as 20 February to 2 March 2022. The *hour* is not independently confirmed.

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
- **Georeferencing:** the depth TIFFs carry no CRS. The archive's `Spatial_References.txt` (UTM zone per region, e.g. Australia = UTM 56S) and one `.tfw` world file per region supply it. **The world files describe the 30 m grid for all four regions, so they apply directly only to the 30 m depth grids (Australia and the UK); for the coarse grids the recovered scale and offset would have to be applied on top (not done).** The world-file origin is the *centre* of the upper-left pixel, so the affine transform needs a half-pixel shift. We checked this against real satellite imagery: the raster's river channel and flood edge line up with the real river (visual check, tens of metres; `verification_outputs/georeferencing/` in the code handoff).

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
- No CNN-LSTM/ConvLSTM baseline is reported for FloodCastBench (only U-Net and FNO/FNO+).
- To our knowledge FloodCastBench's cross-regional transfer has not been tried with CNN-LSTM. We run the paper's three same-resolution transfer settings (Australia → UK at 30 m and 60 m, Pakistan → Mozambique at 480 m; Section 10).

**Scope limits (deliberate boundaries):**
- No new hydrodynamic simulations; we rely entirely on FloodCastBench's simulated sequences.
- **Patch-based** (128×128) forecasts at 30-minute steps, 2.5 hours ahead. No whole-scene training, and no claim of operational early-warning accuracy.
- **All four regions and three resolutions are used, but never mixed in one model.** The paper's own transfer tests are Australia → UK (60 m, plus 30 m zero-shot) and Pakistan → Mozambique (480 m); we run those same-resolution pairs. Severity is scored only for Australia at 30 m (the label is degenerate elsewhere). Our task differs from the paper's (5 observed frames → 5 forecast frames on 128-px patches, versus 19 steps from the initial state over the whole domain), so our numbers are **not comparable** to its U-Net / FNO / FNO+ tables.
- No new Indian dataset; no public source fits this exact task without out-of-scope data engineering.
- Capsules have no precedent for dense regression, so they are scoped to severity classification. The motivation — routing-by-agreement may preserve spatial relationships that max-pooling discards — is a **hypothesis we tested, not an assumption** (Section 10).

## 6. Architecture (as built)

Two branches share one encoder. Each has its own explainability mechanism, and a post-hoc rule layer turns the two outputs into a justified severity call.

```
Input: 5 past frames x (depth, DEM, rainfall), 128x128, 30-min steps
            |
            v   fixed normalization inside the model
   CNN Encoder (TimeDistributed Conv2D 16-32-64, stride 2)  <..... Grad-CAM (post-hoc)
            |  -> 16x16x64 feature map per step
      +-----+------------------------------+
      v                                    v
 Dense(128) -> LSTM(128) encoder      last-step features (16x16x64)
      |                                    |
      v                                    v
 LSTM decoder, one step per          PrimaryCaps (8 x 8-D, conv 3x3/2)
 forecast frame                            |  512 capsules
      |                                    v
      v                              Routing-by-agreement (3 iterations)
 Anchored decoder                          |
 last observed frame                       v
   + learned change  <-- skip        4 class capsules (16-D)
   (from encoder features)                 |  length of each = class score
      |                                    v
      v                              Severity class (longest capsule)
 Depth maps, 5 steps ahead                 |
      |                                    |
      +-------------------+----------------+
                          v   (post-hoc, no gradients)
                 Symbolic Rule Layer
                 depth-rate, rainfall, slope, drainage-proximity
                          |
                          v
              Reasoned severity + stated reason
```

**Component by component:**

1. **Input:** a window of 5 observed frames with three channels (depth, DEM, rainfall). Depth is divided by 5 m, DEM is made patch-relative and divided by 50 m, rainfall is divided by 5 mm — done *inside* the model so it is applied identically everywhere.
2. **Shared CNN encoder:** three `TimeDistributed` `Conv2D` layers (16, 32, 64 filters, 3×3, stride 2) turn each 128×128 frame into a 16×16×64 feature map. Feeds both branches.
3. **Depth branch:** the feature maps go through a dropout + `Dense(128)` bottleneck and an `LSTM(128)` encoder. A one-step LSTM decoder runs once per forecast frame, and a dense bottleneck reshapes its state to 16×16×64. That is concatenated with the encoder's last-step features (**skip connection**) and upsampled by three `Conv2DTranspose` layers to 128×128, ending in a 1×1 convolution, **zero-initialised**, scaled by 5 cm. **The output is the last observed depth frame plus this learned change.** An untrained model therefore reproduces the last frame exactly (verified: max difference 0.0), so training can only improve on it.
4. **Severity branch:** a Capsule Network built as custom `tf.keras.layers.Layer`s in the same TensorFlow graph as the encoder, so severity gradients update the shared encoder (verified: severity loss alone gives real gradients on all encoder convolution variables; one combined step updates all 29 trainable variables). It reads the encoder's **last-step** features: `PrimaryCaps` (8 capsule types × 8 dimensions, 512 capsules) → routing-by-agreement (3 iterations) → 4 class capsules of 16 dimensions. The **length** of each class capsule is that class's score; the prediction is the longest.
5. **Grad-CAM (post-hoc):** explains the depth branch at the encoder's last convolution. Because the model predicts a *signed change*, the target is signed by the predicted direction of change (otherwise the ReLU blanks the map whenever the model predicts drying). Not part of the forward pass or any loss.
6. **Symbolic Rule Layer (NeuroSymbolic):** a deterministic function with no weights and no gradients. It starts from the capsule's class and applies explicit rules:
   - **Escalate one level** if predicted depth-rate > 0.0030 m/hr **and** more than 1000 m from baseline water (a floodplain-inundation pattern).
   - **De-escalate one level** if depth-rate < −0.0001 m/hr (receding).
   - **Informational only** (add a reason, no class change): rainfall > 3.0 mm with rising depth; slope > 7% with rising depth.
   - Depth-rate is the model's predicted change from the last *observed* frame per hour of real time. Distance to water comes from the land-use "water" class (checked against the Esri/Impact Observatory legend). The thresholds were derived from **40 real Australian windows (10 per class)** at 30 m, so **their calibration is valid only there, and it is loose**: over all Australian windows the true depth-rate separates the classes only weakly (median 0.000 / 0.000 / 0.0006 / 0.0004 m/hr for none / minor / moderate / severe; only ~22% of moderate and severe windows exceed the 0.0030 escalation threshold). The per-hour normalisation does hold across time steps (the 5-minute-step and 30-minute-step rates agree).

**Caveat on the "LSTM":** the depth branch contains an encoder LSTM and a decoder LSTM, but in the trained model the encoder's depth history is not exploited (Section 10); the forecast is driven by the last frame and the current rainfall.

**Losses and training:** depth loss = MSE divided by the persistence MSE (1.0 = exactly copying the last frame; below 1.0 = better); severity loss = CapsNet margin loss; the two are summed and backpropagated through the shared encoder. Adam, peak learning rate 1e-3 with cosine decay to 2%, gradient clipping at 1.0, dropout 0.3, batch size 1, 40 epochs.

**Why two separate explainability mechanisms:** Grad-CAM (visual) addresses the depth branch — *where* the encoder looked. The Symbolic Rule Layer (logic-based) attaches a stated reason to the severity output. Both are post-hoc; **neither changes the forward pass.**

## 7. End-to-End Pipeline

| Step | What happens | Output |
|---|---|---|
| 1. Data | Load depth, DEM, rainfall for any region and resolution (30 / 60 / 480 m) via a recovered scale + crop offset per combination; align rainfall to depth; refuse NoData windows and unregistered geometries | Aligned rasters |
| 2. Windows | 128×128 patches; 5 past + 5 future frames at 30-min steps; class-balanced sampling; **split by spatial position** (validation patches are never seen in training) | 400 training / ~240 validation windows |
| 3. CNN Encoder | TimeDistributed Conv2D | Shared feature maps for both branches |
| 4. LSTM encoder–decoder | Encodes the 5-frame history; a decoder LSTM unrolls one step per forecast frame (audit: the encoded depth history is not what drives the skill, see Section 10) | Per-step state passed to the decoder |
| 5. Anchored decoder | Last frame + learned change, with skip connection | **Depth maps** |
| *(parallel)* Capsule network | PrimaryCaps → routing → class capsules | **Severity class** |
| 6. Symbolic Rule Layer | Post-hoc rules over depth-rate, rainfall, slope, drainage proximity | **Reasoned severity** with justification |
| 7. Evaluate | Skill vs persistence, RMSE, NSE, r, CSI; severity vs threshold baseline | Results table |
| 8. Visualize | Attach real CRS/transform; overlay on satellite imagery | Georeferenced map |

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

`GeoTIFF (predicted depth) → rasterio (CRS + transform) → folium (overlay on Esri satellite imagery)`

The per-region UTM zone and pixel transform come from FloodCastBench's own world files and spatial-reference table. They describe the 30 m grid, so we georeferenced and checked **Australia and the UK only**; the 60 m / 480 m depth grids would need the recovered scale and offset applied on top, which was not done or checked. The transform includes the half-pixel shift for the pixel-centre origin.

## 10. Evaluation and Results

**30 m setup (Australia → UK; the rest of the dataset follows in "Results on the rest of the dataset"):** train on Australia (400 class-balanced windows from 46 patches); validate on 16 held-out Australian patches (~240 windows); test on the UK, which the model never sees (136 windows from 2 patches).

**Baselines, and why they matter.** Persistence ("the future equals the last observed frame") is off by only **2 mm at 5-minute steps** and **13 mm at 30 minutes**; thresholding the last frame gives severity right 99.9% (96.7% at 30 min) of the time. So a small error alone proves nothing. Depth is reported as **skill = 1 − (model RMSE ÷ persistence RMSE)²**: 0 = no better than copying, above 0 = better. NSE and r cannot separate the model from persistence at all (both ≈ 0.999).

**Seven full-size runs** (Colab T4 and laptop CPU; identical settings differ by about ±0.03 in Australian skill from floating-point non-determinism alone):

| Run | Depth skill, Australia (balanced) | Australia (uniform) | **UK (never seen)** | Severity accuracy (threshold baseline) |
|---|---|---|---|---|
| Colab 1, seed 42 (100 val windows) | +0.190 | +0.225 | +0.189 | 70.0% (90.0%) |
| CPU, seed 42 | +0.169 | +0.163 | +0.204 | 62.9% (92.9%) |
| Colab 2, seed 42 | +0.191 | +0.172 | +0.201 | 62.5% (92.9%) |
| CPU, seed 42 (audit run) | +0.217 | +0.215 | +0.217 | 65.4% (92.9%) |
| CPU, seed 7 (different windows and held-out patches) | +0.140 | +0.077 | +0.206 | 54.6% (92.4%) |
| CPU, seed 42 + augmentation | +0.207 | +0.217 | +0.202 | 81.2% (92.9%) |
| CPU, seed 7 + augmentation | +0.051 | +0.071 | +0.044 | 68.1% (92.4%) |

**What the results support**
- **Depth:** the model beats "copy the last frame" in **all seven runs, in both Australia and the never-seen UK**. Skill +0.19 is about **10% lower RMSE** (Colab 1: 10.0 vs 11.2 mm in Australia, 17.5 vs 19.4 mm in the UK). The gain is millimetres because the simulated flood barely changes over 2.5 h. The size of the win varies with seed and validation patches (Australia +0.05 to +0.23, UK +0.04 to +0.22), so **quote a range, not a single number**. The most stable figure is UK skill in the five default-setting runs: +0.19 to +0.22. The gain is a roughly constant ~20% reduction in squared error at every lead time (30 min to 2.5 h).
- **What drives the depth forecast (audit, test-time ablations on one trained model):** removing **rainfall** takes Australian skill from +0.217 to **+0.013** and UK skill from +0.217 to **+0.004**; removing terrain (+0.216 / +0.213), scrambling the order of the depth history (+0.217 / +0.217) or freezing it (+0.209 / +0.213) changes skill by less than 0.01. The forecast is effectively **"last frame + current rainfall → change"**. So the deck's claim that the LSTM "captures temporal context" is **not supported**: the decoder LSTM unrolls the per-step change, but the encoder's depth history is not exploited, and the UK transfer is transfer of a rainfall-to-change relationship. (Caveat: test-time removal measures reliance, not what a differently trained model could learn.)
- **Severity:** the capsule head scores **55–70%** (chance 25%) but **stays far below the 92–93% obtained by simply thresholding the last frame**. Severe floods are under-called (recall 55%; 26 of 60 called "moderate"). On windows where the class changes over the forecast it is no better than elsewhere (59% vs 63%), so there is **no evidence it anticipates escalation**. The head overfits its 46 training patches (training margin loss 0.026 while validation stays flat). So the "capsules preserve spatial relationships useful for severity" hypothesis is **not supported** by this data and label.
- **Augmentation** (random flips/rotations of training windows): raised severity on both seeds (+18 and +13.5 points) but destabilised depth training on seed 7 (UK skill +0.21 → +0.04), so it is **off** by default. With two seeds a fluke cannot be separated from a systematic effect.
- **Symbolic rule layer:** over all 240 validation windows, capsule alone **62.5%** → capsule + rules **62.5%**. The rules changed the class in 8 windows (3%): **4 fixed a wrong answer, 4 broke a right one**. Net effect on accuracy: zero (a second full evaluation gave 65.4% → 65.8%, with 3 changes: 2 fixed, 1 broke). It is a **justification layer, not an accuracy booster**; its reasons are computed post hoc from window features, not read out of the network, and were not checked by a flood expert.
- **Grad-CAM:** over 27–39 validation windows per model, heat is higher on wet than on dry pixels in 69–93% (chance = 50%), median ratio 1.1–1.5×. A mild preference for water at 16×16 resolution; the hot spots move between training runs even with the same seed. Not a full explanation.

**Caveats to state when presenting:** these are development-set numbers (design choices were made while watching them, and the UK windows never changed); few seeds; one storm (validation patches are unseen *places*, not unseen *storms*); a 2-patch UK test at 30 m; no comparison with U-Net/FNO under the paper's whole-scene protocol.

## Results on the rest of the dataset (Pakistan, Mozambique, 60 m)

The 30 m experiments above use half of FloodCastBench. The other half was first left out (wrongly) and is now used: the paper's other two transfer settings, **depth only** (Pakistan → Mozambique at 480 m; Australia → UK at 60 m), same model, persistence-anchored, 400 training windows, 40 epochs, two seeds. Skill = 1 − MSE_model / MSE_persistence, as before.

| Evaluation | Skill, seed 42 | Skill, seed 7 | Skill with rainfall removed (42 / 7) |
|---|---|---|---|
| Pakistan held-out tiles (480 m) | **+0.09** | **+0.10** | +0.00 / +0.02 |
| **Mozambique, never seen** (480 m) | **+0.05** | **+0.07** | +0.00 / +0.01 |
| Australia held-out tiles (60 m) | **+0.41** | **+0.23** | +0.02 / -0.01 |
| **UK, never seen** (60 m) | **+0.18** | **+0.18** | +0.00 / +0.01 |

- **Depth beats "copy the last frame" everywhere it was tested, on both seeds, including the two never-seen regions:** Mozambique +0.05 to +0.07, UK at 60 m +0.18 to +0.18. Both of the paper's transfer directions now have a result. The gains are small in absolute terms (about 1–2 mm of RMSE), like at 30 m, because the simulated floods change little over 2.5 hours.
- **The held-out Australian tiles at 60 m vary a lot between seeds (+0.23 to +0.41):** each seed holds out different tiles. The unseen-region numbers are the stable ones. Quote ranges.
- **The 30 m finding replicates at 60 m and 480 m:** removing rainfall takes skill to about zero in all eight evaluations; terrain and the depth history change it by less than 0.02. The forecast is "last frame + current rainfall" at every resolution, and the encoded depth history is not what gives the skill.
- **A weakness, stated plainly:** at the paper's 1 mm wet threshold the model is *worse* than persistence (it leaves millimetre-scale noise in dry cells); at 0.1 m it is equal or slightly better.
- **A failed run, diagnosed:** the first 60 m run collapsed to a constant offset (skill +0.045, every ablation identical). Extreme windows hit Adam at full learning rate; a 200-step warm-up fixed it and training now raises an error if a model becomes input-blind. The 30 m runs did not collapse (their ablations show input-dependent outputs) but were not re-run with the warm-up.
- **Not comparable to the paper's tables** (whole-domain forecasts from the initial state); one storm per unseen region (Mozambique: 4 heavily overlapping tiles; UK at 60 m: 2 padded windows); development-set numbers.

## 11. Course Outcome (CO) Mapping

CO1–CO5: **Dense (CO1)** · **Classification head (CO2)** · **CNN enc/dec (CO3)** · **Dropout/LR (CO4)** · **LSTM enc-dec (CO5)**

## 12. Decided, Built, Not Established

**Built and verified**
- Two-branch architecture in a single TensorFlow/Keras graph; the Capsule Network is a custom `tf.keras.layers.Layer`, so severity gradients reach the shared encoder (29/29 variables updated).
- Persistence-anchored decoder with skip connection (untrained model = persistence, exactly).
- Symbolic Rule Layer with thresholds derived from real Australian data; direction-aware Grad-CAM.
- Data pipeline with measured alignment (UK offset), refusal of misaligned regions, class-balanced sampling and a spatial train/validation split.
- The Colab notebook, executed end to end at full size (twice) and locally at tiny and full size.
- Georeferenced output that lines up with real satellite imagery (Australia).
- An independent audit (September 2026, Section 14): labels, splits, window indexing, loader output and the headline metric were re-derived from raw files; all passed.
- The whole dataset: all 14,868 files read and checked; geometry recovered for all six region/resolution combinations; the loader and a depth-only experiment script (`depth_experiment.py`) run Pakistan → Mozambique at 480 m and Australia → UK at 60 m, with the ablations and a dead-network guard.

**Not established**
- That the model learns temporal dynamics (depth history contributes < 0.01; the skill is rainfall-driven).
- A clean final test (design choices were made while watching validation and UK numbers).
- Any stable single depth-skill number (it moves with seed and validation patches, and by ~±0.03 between identical runs).
- That the capsule head adds anything over thresholding depth, or that the symbolic layer adds accuracy.
- That Grad-CAM points at the same places from run to run.
- Rainfall-to-depth alignment beyond the matching date ranges (the hour is inferred); the UK offset beyond the measurement; that the symbolic thresholds transfer beyond Australia at 30 m.
- Cross-regional transfer beyond **one storm per unseen region** (UK 30 m: 2 patches; UK 60 m: 2 padded windows; Mozambique: 4 heavily overlapping tiles), and any result at 60 m / 480 m beyond the seeds reported in Section 10.
- The 30 m results under the learning-rate warm-up recipe (they were produced without it; their ablations show they did not collapse).
- Pakistan's alignment beyond about half a cell (two independent signals agree, but the evidence is weaker than for the other regions; it cannot affect the depth results, see Section 3).

**Next steps if time allows:** more seeds and storms; compare with U-Net/FNO under the paper's whole-scene, initial-state protocol; re-run the 30 m experiments with the learning-rate warm-up.

## 13. Where Everything Is

- **This folder:** the updated deck (`..._v2.pptx`), this explainer, and `architecture_diagram/` (TikZ source, PDF and PNG of the redrawn diagram). The original deck and explainer are unchanged.
- **Code handoff** (`flood_forecasting_handoff.zip`, and `C:\Dev\flood-forecast-demo` on Anhad's laptop): `PROJECT_LOG.md` (full record with corrections), `HANDOFF_README.md`, `flood_forecasting_colab.ipynb` (30 m, severity + rules + Grad-CAM; run on Colab: Runtime → GPU → Run all), `flood_forecasting_multires_colab.ipynb` (the 60 m and 480 m depth experiments), `depth_experiment.py` (the same as a script) and the verification logs.

## 14. Independent Audit (September 2026)

A full re-check from raw files, written to find mistakes (scripts and logs: `verification_outputs/audit/` in the code handoff).

**Passed:** raw rasters are clean inside the aligned crop; the train/validation split shares no tile; no future information enters any input; 24 random windows match independent reads exactly (inputs, targets, rainfall, terrain, both labels); 186 scanned windows match an independent census; the UK offset is again the strongest of 121 candidates; the headline skill recomputes identically; the metric code returns known answers; an untrained model equals persistence exactly; predicted depth never goes below zero.

**Found and fixed or corrected**
- **NoData ring in the UK layers** (float32 −3.4×10³⁸, outside the aligned crop). No window we read touches it, so results are unaffected; the loader had no guard and now refuses such windows.
- **The "LSTM captures temporal context" claim was untested and is not supported** (Section 10). Wording corrected in the deck and here.
- **"Pakistan and Mozambique cannot be aligned" was wrong, and the scope was incomplete**: the pipeline had read only 50% of the dataset. The first search failed for a method reason (no trend removal); a validated probe recovered both geometries, the loader was generalised, and both regions plus the 60 m data are now used (Section 3, Section 10).
- **"Two weeks of simulation"** is the Pakistan event only (others: about two days; UK about four hours). The paper's own transfer test is Australia → UK, our direction.
- **Rule thresholds** are on the right scale across time steps but came from 40 windows and separate the classes only loosely.
- **Development-set hygiene**: the numbers are not a clean final test.

**Confirmed against sources:** the FloodCastBench paper benchmarks exactly U-Net, FNO and FNO+ (no LSTM/CNN-LSTM), so the research-gap claim stands; the Norfolk CNN-LSTM paper reports "four to six hours using TUFLOW to 3.2 min per event".

**Mistakes made during the project, all corrected** (full list in `PROJECT_LOG.md`, section 8): declaring the project technically complete while half the dataset was unused (Pakistan, Mozambique and the 60 m data were dropped after one failed search and never revisited; the "audit" covered only the half in use); leaving a latent training-collapse mode in the recipe (found on the 60 m run and fixed with a warm-up and a guard); asserting layer alignment after checking one region; blaming data volume for what was an architecture and loss problem; a rule-engine "success" fed by noise; an invalid first notebook; an unchecked map-tile fix; quoting single-seed numbers as stable; a one-window Grad-CAM claim; adopting augmentation on one seed; and describing the model as temporal without testing it.

---

*If the architecture changes again, regenerate this file from the current deck and log rather than hand-editing, to avoid drift.*
