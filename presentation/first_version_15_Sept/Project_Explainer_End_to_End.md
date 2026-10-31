# CNN-LSTM Based Spatiotemporal Flood Evolution Forecasting

**With capsule-based severity classification and a post-hoc NeuroSymbolic reasoning layer**

Anhad Mahajan (PRN: 24070126031) & Aman Sumeet Saxena (PRN: 24070126030) — DL Lab Project

This document explains the project end to end: the problem, the dataset, the exact architecture, the full pipeline from raw data to final output, the tech stack, and what is decided versus still pending. It reflects the current, verified state of the project as of the latest fixed deck (`CNN-LSTM_Based_Spatiotemporal_Flood_Evolution_Forecasting.pptx`).

---

## 1. The Problem

**Goal:** given recent flood depth, rainfall, and land-shape (terrain) data, predict what the flood will look like a few time steps into the future — across the *whole* spatial area being modeled, not just at one point.

This is explicitly **not**:
- A water-level sensor. Knowing the water height at one river gauge doesn't tell you which streets or fields nearby will actually go underwater.
- A single-photo flood/no-flood classifier. Looking at one satellite image and labeling it "flooded" or "not flooded" doesn't show how the flood is spreading over time.

The model has to output a **spatiotemporal** prediction: a sequence of depth maps evolving over the forecast horizon, not a single number or a single label.

## 2. Why It Matters

- A full physics-based hydrodynamic simulation is accurate but slow: generating the FloodCastBench Pakistan 2022 event (14 simulated days) took **about two weeks of GPU compute** (NVIDIA A6000), per the dataset's own paper.
- A trained surrogate model gives a forecast in seconds. As a real-world precedent: a CNN-LSTM surrogate for Norfolk, Virginia cut per-event runtime from **4–6 hours** (physics-based TUFLOW model) to **3.2 minutes** (Wang et al., 2026).
- The gap we're filling: **no one has benchmarked a CNN-LSTM model on FloodCastBench.** The dataset's own paper only reports U-Net, FNO, and FNO+ as baselines. We are adding the first CNN-LSTM result on this exact benchmark.

## 3. The Dataset — FloodCastBench

- **Citation:** Xu, Q., Shi, Y., Zhao, J. et al. "FloodCastBench: A Large-Scale Dataset and Foundation Models for Flood Modeling and Forecasting." *Scientific Data* 12, 431 (2025). DOI: 10.1038/s41597-025-04725-2. CC BY 4.0. Hosted on Zenodo (DOI 10.5281/zenodo.14017092).
- **Four real flood events**, each a separate region: Pakistan (2022), UK (2015), Australia (2022), Mozambique (2019).
- **Two fidelity tiers:**
  - High-fidelity: 30m and 60m resolution — Australia and UK.
  - Low-fidelity: 480m resolution — Pakistan and Mozambique.
- **Inputs per event:** depth-map time series (TIFFs), DEM (digital elevation model), rainfall time series, land use/land cover.
- **Filenames are unpadded integers** — seconds since the start of the simulated event (e.g. `100200.tif`, `1001700.tif`). They must be **sorted numerically, not alphabetically**, or frame order scrambles (`"1001700.tif" < "100200.tif"` as strings, despite 1,001,700 > 100,200).
- **Georeferencing — verified directly against the real downloaded archive, not assumed:** the individual simulation-output TIFFs (the depth-map time series) do **not** carry an embedded CRS or affine transform — confirmed with `rasterio` on real files (2 regions, 2 fidelities, 2 timesteps: all return `CRS: None`, identity transform). Real georeferencing *does* exist in the archive, but separately: a `Spatial_References.txt` giving each region's UTM zone (e.g. Australia = WGS_1984_UTM_Zone_56S) and one `.tfw` world file per region (exact pixel size + origin, e.g. Australia = 30m pixels, origin at UTM 56S (517452.19, 6808598.86)). **Our pipeline is responsible for attaching this per-region CRS/transform to our own predicted output grids at export time** — it is not something we simply "read off" the input files.
- **Existing baselines reported by the dataset's own paper:** U-Net, FNO (Fourier Neural Operator), FNO+ (FNO plus terrain/rainfall inputs). No CNN-LSTM/ConvLSTM baseline exists.

## 4. Literature Basis

| Paper | Year | Contribution | Relevance |
|---|---|---|---|
| Shi et al. (ConvLSTM), arXiv:1506.04214 | 2015 | Convolutional gates fuse CNN + LSTM in one cell | Closest architectural ancestor; we use a lighter CNN-then-LSTM variant instead of fusing them into one cell |
| Ronneberger et al. (U-Net), doi:10.1007/978-3-319-24574-4_28 | 2015 | Encoder–decoder CNN with skip connections | FloodCastBench's own spatial-only baseline |
| Malik, Feng, Shao & Abduljabbar, doi:10.1007/s12145-024-01354-y | 2024 | Time-distributed CNN-LSTM (TD-CNN-LSTM) | The design pattern we adopt for the depth-forecasting path |
| Wang et al. (Norfolk, VA), doi:10.1016/j.ejrh.2026.103234 | 2026 | CNN and LSTM fused at a bottleneck | Closest published real-world analogue to our pipeline; source of the "3.2 minutes" figure above |
| Xu et al. (FloodCastBench), doi:10.1038/s41597-025-04725-2 | 2025 | The dataset itself, + U-Net/FNO/FNO+ benchmark | Our dataset; the CNN-LSTM gap we fill |
| Sabour, Frosst & Hinton (Capsule Networks), arXiv:1710.09829 | 2017 | Vector capsules with routing-by-agreement, preserving pose/spatial relationships lost to CNN max-pooling | Basis for the severity-classification head |

All six citations verified as real papers with correct DOIs during this project's technical review.

## 5. The Research Gap, and What's Explicitly Out of Scope

**Established gaps we're addressing:**
- No CNN-LSTM/ConvLSTM baseline exists for FloodCastBench (only U-Net and FNO/FNO+ reported).
- Most spatially-explicit CNN-LSTM flood work is single-region; FloodCastBench's cross-regional transfer test (train on some regions, test on others) has never been attempted with CNN-LSTM.

**Explicit scope limits (not gaps we're failing to address — deliberate boundaries):**
- No new hydrodynamic simulations. We rely entirely on FloodCastBench's already-simulated sequences.
- No full-resolution whole-scene training on all four events, and no claim of operational early-warning-grade accuracy.
- No new Indian dataset — no existing public source currently fits this exact task without out-of-scope data engineering.
- Capsule networks are scoped to **severity classification only**, not dense depth regression — there's no established precedent for capsules in dense regression. They're used here because routing-by-agreement preserves spatial relationships between inundated regions and terrain contours that max-pooling would discard — information relevant to judging *severity*, not just raw depth.

## 6. Architecture

Two branches share a common encoder, each branch has its own explainability mechanism, and a post-hoc reasoning layer combines both outputs into a final, justified severity call.

```
Input patches (Depth, DEM, Rainfall)
            │
            ▼
   CNN Encoder (TimeDistributed Conv2D)  ◄╌╌╌╌╌╌ Grad-CAM (post-hoc, visual)
            │
     ┌──────┴──────┐
     ▼             ▼
LSTM Encoder–   Capsule Network
Decoder         (PrimaryCaps →
(temporal       Routing-by-Agreement)
context)             │
     │                ▼
     ▼          Severity Class
CNN Decoder     (4-way softmax)
(Conv2D-             │
Transpose)           │
     │                │
     ▼                │
Depth Map             │
Prediction            │
     │                │
     └──────┬─────────┘
            ▼  (both, dashed/post-hoc)
   Symbolic Rule Layer (NeuroSymbolic)
   rules over: depth-rate, rainfall, slope, drainage-proximity
            │
            ▼  (solid)
   Reasoned Severity Output
   (with stated justification)
```

**Component by component:**

1. **Input patches** — spatial patches of depth, DEM (elevation), and rainfall, as a time sequence.
2. **CNN Encoder (TimeDistributed Conv2D)** — shared spatial feature extractor, applied per time step. Feeds *both* downstream branches.
3. **Depth-forecasting branch:** LSTM Encoder–Decoder (captures temporal context, produces the forecast horizon) → CNN Decoder (Conv2DTranspose, upsamples back to spatial resolution) → **Depth Map Prediction**.
4. **Severity-classification branch:** Capsule Network — PrimaryCaps followed by routing-by-agreement — → **Severity Class** (4-way softmax: a discrete severity level).
5. **Grad-CAM** — post-hoc, applied to the CNN encoder/decoder. Produces a visual heatmap explaining *which input regions* drove the depth prediction. Not part of the forward inference path or any loss — purely explanatory, run after the model produces its output.
6. **Symbolic Rule Layer (NeuroSymbolic)** — this is the part that makes the "NeuroSymbolic" label accurate: a genuine rule-based logic layer, not a differentiable loss term. It takes **both** the Depth Map Prediction and the Severity Class as input, alongside auxiliary signals (rainfall, slope, drainage-proximity, and the depth map's own rate of change over time), and applies explicit rules to produce a **Reasoned Severity Output with a stated justification** — e.g. "escalated to severe because depth-rate exceeds X near a low-drainage-proximity area," rather than a black-box softmax number. This is post-hoc reasoning over the network's outputs, not a training-time loss — no gradients flow through it.

**Why two separate explainability mechanisms, not one:** Grad-CAM (visual, post-hoc) explains the depth-forecasting path — *where* in the input the CNN encoder/decoder focused. The Symbolic Rule Layer (logic-based) explains the severity-classification path — *why* a given severity was assigned, in terms of stated rules. They cover different branches of the model and answer different kinds of "why" questions.

**A known implementation constraint that shapes the tech stack:** the CNN Encoder is *shared* between the TensorFlow/Keras-built depth path and the Capsule Network. For the whole model to train as one coherent graph — so that severity-classification gradients can actually reach and update the shared encoder — the Capsule Network must be implemented as a custom `tf.keras.layers.Layer`, not as a separate PyTorch model bolted on afterward. This is a **decided design choice** (see Section 13 — it is not yet implemented in code).

## 7. End-to-End Pipeline

| Step | What happens | Output |
|---|---|---|
| 1. Data | Load depth, DEM, rainfall for a chosen region/event from FloodCastBench | Raw rasters + time series |
| 2. Preprocess | Patch extraction, normalization | Fixed-size spatial patches, per time step |
| 3. CNN Encoder | TimeDistributed Conv2D over the patch sequence | Shared spatial feature maps, feeding both branches below |
| 4. LSTM Encoder–Decoder | Temporal modeling over the encoded sequence, projected to the forecast horizon | Temporally-contextualized features |
| 5. CNN Decoder | Conv2DTranspose, upsamples back to spatial resolution | **Depth Map Prediction** |
| *(parallel to 4–5)* Capsule Network | PrimaryCaps → routing-by-agreement, run on the same step-3 encoder output | **Severity Class** (4-way softmax) |
| 6. Symbolic Rule Layer | Post-hoc rules combining both branch outputs + auxiliary features | **Reasoned Severity Output**, with justification |
| 7. Evaluate | RMSE, NSE, r (correlation), CSI (Critical Success Index) against baselines (U-Net, FNO, FNO+) | Quantitative comparison table |
| 8. Visualize | Attach real per-region CRS/transform to the predicted depth grid; render as an interactive overlay | Georeferenced map, viewable over a real satellite/street basemap |

Grad-CAM runs alongside step 3 (post-hoc, on the trained encoder/decoder) — it isn't a numbered pipeline step since it doesn't feed forward into anything; it's diagnostic.

## 8. Tech Stack

| Tool | Role |
|---|---|
| Python | Data loading, model code, demo backend |
| TensorFlow / Keras | CNN, LSTM, Dense, Capsule Network, softmax layers — the entire model in one framework |
| NumPy | Patch extraction, normalization, metrics |
| Pandas | Train/val/test index, per-epoch logging |
| Matplotlib / Plotly | Training curves, heatmap rendering |
| scikit-learn | Splitting, scaling, standard metrics |
| Grad-CAM | Post-hoc visual explainability for the CNN encoder/decoder |
| Folium / Plotly (Mapbox) | Interactive georeferenced overlay on a real basemap |
| rasterio | Applies real per-region CRS/transform (from the archive's world files) to predicted output grids at export time |

## 9. Visualization Pipeline

`GeoTIFF (predicted flood depth) → rasterio (applies real per-region CRS + transform at export time) → Folium/Plotly (interactive overlay on a real satellite/street basemap)`

Per-region CRS and pixel transform come from FloodCastBench's own world files and spatial-reference table (see Section 3) — resolution and origin are fixed per region, so this is a one-time lookup per region, applied when writing out each predicted grid.

## 10. Evaluation

Metrics against the U-Net / FNO / FNO+ baselines reported in the FloodCastBench paper:
- **RMSE** — root mean squared error on predicted depth.
- **NSE** — Nash–Sutcliffe Efficiency (standard hydrology goodness-of-fit measure).
- **r** — correlation coefficient.
- **CSI** — Critical Success Index (flood extent classification skill).

The cross-regional generalization test (train on some regions, e.g. Pakistan/Australia, test on held-out regions, e.g. Mozambique/UK) is explicitly the kind of test FloodCastBench's own paper uses to evaluate transferability, and directly backs the "cross-regional gap" claim in Section 5 with a real result rather than just an assertion.

## 11. Course Outcome (CO) Mapping

CO1–CO5: **Dense (CO1)** · **Classification head (CO2)** · **CNN enc/dec (CO3)** · **Dropout/LR (CO4)** · **LSTM enc-dec (CO5)**

## 12. What's Decided vs. What's Still Pending

**Decided (reflected in the current deck):**
- Overall two-branch architecture (depth via LSTM/CNN-Decoder, severity via Capsule Network).
- Symbolic Rule Layer as genuine post-hoc rule-based reasoning (not a differentiable physics loss) — this is what makes "NeuroSymbolic" an accurate label.
- Grad-CAM for the depth path, Symbolic Rule Layer for the severity path — two separate, deliberately-scoped explainability mechanisms.
- Single-framework implementation: Capsule Network to be built in TensorFlow/Keras (not PyTorch), so the shared CNN Encoder trains as one coherent graph.
- Data access via Hugging Face + Colab rather than a local 21.6GB download, to avoid Colab's ephemeral-disk bottleneck.

**Still pending / not yet implemented:**
- **The Capsule Network still needs to actually be built as a custom `tf.keras.layers.Layer`.** The deck now correctly *describes* the single-framework design; the code doesn't exist yet.
- **The Symbolic Rule Layer's exact rules are not yet defined.** The deck names the factors (depth-rate, rainfall, slope, drainage-proximity) but not the actual if/then logic or thresholds — this needs to be written down and be ready to defend before presenting (e.g. "severity escalates one level if depth-rate exceeds X within Y meters of a drainage point").
- **Colab compute budget:** the standard Colab Pro tier (100 compute units/month) comfortably covers this model at its current small scale (patch-based, short sequences) — a T4 GPU gives roughly 51 hours from that budget, and a model this size trains in seconds-to-minutes per epoch even on CPU. The realistic constraint is data engineering (caching extracted patches to Drive per region) and team bandwidth, not GPU-hours.
- **Cross-regional generalization test** (Section 10) — planned, not yet run.

---

*This document reflects the project as of the latest corrected deck. If the architecture changes again, this file should be regenerated from the current PPT rather than hand-edited, to avoid drift between the two.*
