# Project Log — Building the Real Pipeline

The actual record of what was built, why, what broke, what the real numbers were — and where
earlier claims turned out to be wrong. Companion to the plan document (`effervescent-humming-tide.md`).

**How to read this.** Sections for Stages 0-5b describe what was done and reported at the time.
Where a later check showed a claim was wrong or incomplete, the section carries a **Correction**
block, and the full account of what changed is in **"Pre-flight before Colab"**, which is the most
important section here. Every number quoted was produced by code in this handoff.

**Round 2 (20 September 2026).** A second audit corrected several statements of the earlier sections (rain timing and 'no future information'; 'the skill is rainfall-driven / the history is not used'; 'RMSE in metres' for the paper's relative error; georeferencing scope; the headline depth skill; the severity/rules/Grad-CAM results). The old text is kept; each stale statement is followed by a **[Correction, round 2 ...]** note. The current state of the project is the section **'Completion, round 2'** (sections 1-10) and **'What is and is not established'** at the end. Severity, symbolic-rule and Grad-CAM numbers in the sections before 'Completion, round 2' belong to the original (v1) joint model; their re-evaluation is section 9 of the round-2 section.

---

## Context: why we started here

The deck (diagram, methodology, tech stack) was fixed and verified in an earlier phase. One decision
from that phase drives everything below: **the Capsule Network is built in TensorFlow/Keras, not
PyTorch**, so the shared CNN encoder receives real gradients from both the depth branch and the
severity branch.

An early prototype (CNN -> LSTM -> CNN-decoder, with a 2-class per-pixel head, tested only on random
data) was reused as the depth branch's starting point; the old head was removed.

**Decision: build and verify locally first; use Colab only when GPU-scale compute is actually needed.**
Local iteration is faster (no notebook overhead, no disconnects, no re-downloading), and the model is
small enough that CPU is fine for checking correctness. This held for everything through the pre-flight.

---

## Stage 0 — Dataset onto Hugging Face

FloodCastBench is **not** on Hugging Face (only Zenodo), so the team's "just pull it from Hugging Face"
plan rested on a false premise. Fix: make it true. Verified first that public HF dataset repos get 1TB
free and that FloodCastBench is CC BY 4.0 (redistribution permitted with attribution). Uploaded the
**extracted folder tree** (not the 21.6GB zip) so Colab can pull single regions with `allow_patterns`.

- 26GB, 14,868 real files (the zip's 14,892 "entries" included 24 directory entries).
- The upload was interrupted once by an app restart (977 files landed); `upload_large_folder` resumed
  from where it stopped.
- Verified by diffing full local vs remote file lists: 14,868 = 14,868, none missing or extra; confirmed
  the repo is public by loading the page unauthenticated.

**Result:** `https://huggingface.co/datasets/mahajananhad/floodcastbench`, 26.9GB.

---

## Stage 1 — Real data pipeline (`real_data_loader.py`)

Depth files (`0.tif`, `100200.tif`: unpadded seconds, every 300s) and rainfall files
(`20220220-S190000.tif`: calendar timestamps, every 1800s) use incompatible naming, and nothing in the
archive says how they line up. Evidence used to align them (Australia): rainfall's 481 files x 1800s
span exactly the 10 days depth spans, and the FloodCastBench paper states the simulation period is
"February 20 to March 02, 2022" - matching rainfall's own date range. The exact *hour* is not
independently confirmed, so the loader flags it as an assumption. **[Correction, round 2: later confirmed physically: rainfall file j is the rain falling during [1800 j, 1800 (j+1)) s (domain-mean depth gain = 0.494-0.502 x the file value; correlation 1.00 at lag 0 in all four regions). The calendar clock time of t = 0 is still not given by the archive; no experiment uses it.]**

Severity labels do not exist in FloodCastBench; they are derived from depth with fixed thresholds
(none <0.3m, minor 0.3-1m, moderate 1-2m, severe >2m) applied to the max depth in the patch over the
forecast frames.

Verified on real Australia patches: severity is monotonic in depth, and the flood water pools in exactly
the river channel the DEM shows as the low point - two independently-loaded layers agreeing.

> **Correction.** This section originally said *all layers share the identical grid*. That was checked
> **only on Australia**. It is false for the other three regions (see Pre-flight). The loader now
> validates alignment and refuses regions it cannot align.

---

## Stage 2 — Depth branch on real data

Trained on 25 real Australia windows for 6 epochs. Loss fell (0.201 -> 0.176); predictions were a faint
blur compared with ground truth.

> **Correction.** This was reported as "exactly what you'd expect from 25 examples over 6 epochs".
> That was only part of it. The decoder rebuilt the entire 128x128 depth field from a 128-number LSTM
> state with **no connection to the observed frame**, so it could never approach a near-exact copy of the
> last frame - and a copy is accurate to ~2 mm at a 5-minute step (measured). More data alone would not
> have fixed the blur. Fixed by a persistence-anchored decoder (see Pre-flight).

---

## Stage 3 — Capsule Network severity branch (`capsule_layers.py`)

`PrimaryCaps` and dynamic routing-by-agreement were built from scratch as `tf.keras.layers.Layer`s so
severity-loss gradients reach the shared encoder. `model.py`'s encoder was refactored to also expose its
raw per-timestep CNN feature maps; the capsule head reads the last observed step's features.

Bugs found and fixed: (1) a tile-rank crash in the first `RoutingCaps` -> rewritten with `tf.einsum`;
(2) a gradient check showing `None` on `PrimaryCaps` weights that was really my test computing the layer
*outside* the `GradientTape`; (3) a sampler that produced 48/56 "severe" windows, so the model just
predicted "severe" (an 83% "accuracy" that was majority-class collapse).

**The check that mattered, verified twice:** severity loss *alone* gives real gradients on all 6 shared
encoder conv variables; a full combined step updates all 29 trainable variables. This proves the
TensorFlow port solved the original framework-split problem.

> **Correction.** After class-balancing, severity accuracy was 14% on 7 windows and was attributed to
> "not enough data". Data volume was part of it; two other causes were found later: softmax over capsule
> lengths (bounded in [0,1)) can never exceed ~0.475 for one of 4 classes, and predictions collapsed to a
> single class (38/40 "moderate" in a 100-window run). Fixed with the CapsNet paper's margin loss. Severity
> is still far below a no-model baseline - see Pre-flight.

---

## Stage 4 — Grad-CAM (`grad_cam.py`)

Adapted from classification to dense regression by using the summed predicted depth as the target,
explaining `cnn_encoder_3` at the last observed step. On a 9-example model the map was a smooth diagonal
gradient unrelated to the river channel.

> **Correction / addition.** Later, with a better model, the map went **blank** on a real window. Verified
> cause: the model predicted a net *fall* (-0.11 m summed over the patch) so the raw map was 0% positive and
> Grad-CAM's ReLU zeroed it. The target is now signed by the predicted direction of change (decided outside
> the gradient path) and the function reports that direction. On the trained local model: falling window
> wet/dry mean 0.374/0.296, rising window 0.301/0.131.

---

## Stage 5 / 5b — Symbolic Rule Layer and end-to-end prediction

A plain deterministic function (no weights, no gradients), which is what makes "NeuroSymbolic" accurate.
Thresholds come from real Australia data across 40 windows spanning all classes:

| Factor | none | minor | moderate | severe |
|---|---|---|---|---|
| depth-rate (m/hr, patch-mean) | 0.0000 | 0.0002 | 0.0028 | 0.0035 |
| rainfall (mm/h; hybrid channel = the interval containing the last frame, partly future) | 0.000 | 0.189 | 2.055 | 4.056 |
| slope (% grade) | 4.85 | 4.53 | 5.79 | 8.22 |
| drainage-proximity (m to water) | 911.9 | 774.4 | 1030.1 | 1260.6 |

Two factors ran opposite to the initial assumption (severe cases are *farther* from baseline water and on
*steeper* ground - floodplain inundation, not channel pooling); the rules follow the data. The land-use
water class (`1`) was confirmed against the Esri/Impact Observatory product's published legend rather than
guessed. An automated check that flagged "a rule fired but severity didn't change" was a flaw in the check,
not the rules (two of four rules are deliberately informational).

> **Correction.** The reported behaviour "the escalation rule fired exactly when depth-rate crossed its
> threshold" was true of the *code*, but the depth-rate fed into it at inference was computed from the
> model's own forecast frames. With the original from-scratch decoder those frames were essentially noise,
> so the values that triggered escalation (0.0026-0.0036 m/hr) were very likely artifacts, not physics.
> (The *thresholds* are sound: derived from real ground truth.) Depth-rate is now the model's predicted
> change from the last *observed* frame per hour of real time. With the redesigned model it varies by window
> and tracks rainfall (see below). **[Correction, round 2: v1 tracking a rainfall channel that contained up to 30 minutes of rain that had not yet fallen.]**

---

## Pre-flight before Colab

Before spending Colab time I executed the notebook's actual code locally against the real data
(`dryrun_notebook.py` - every cell, with small sizes). The first draft of the notebook was **invalid**
in several independent ways. Everything below was measured, not reasoned.

### 1. The regions are not interchangeable

| Region | Depth grid | DEM / rainfall / land-use grid |
|---|---|---|
| Australia | 1073x1073 (30m) | 1073x1073 - matches |
| UK | 170x275 (30m) | **180x285** - 10px larger |
| Pakistan | 810x441 (**480m**) | **13035x7298** (30m base grid) |
| Mozambique | 151x138 (**480m**) | **2624x2321** (30m base grid) |

- **UK:** the depth raster is a crop. Two independent signals agree on a (5,5) pixel offset: depth-vs-DEM
  correlation is -0.274 at (5,5) against a maximum of +0.031 over all 121 candidate offsets, and wet pixels
  land on land-use "water" ~5x more than chance around (5-6, 5-6). On a real 128px patch, aligned
  depth-vs-DEM correlation is **-0.39** versus **+0.015** unaligned - the old loader would have paired the
  UK's floods with a DEM unrelated to them. The offset is measured, not documented.
- **Pakistan / Mozambique:** the loader would have read a 128px window of the 30m DEM at depth-pixel
  coordinates - entirely different geography. The size ratios (~16.1/16.6 and ~17.4/16.8) are not a clean
  16x, so no exact alignment is known. `RegionData` now **refuses** to load them.
- The world files (`.tfw`) say **30m for all four regions**, including the two 480m ones - they describe the
  30m base grid, not the low-fidelity depth rasters. Georeferencing with them is valid for Australia/UK only. **[Correction, round 2: the world files give the 30 m grid directly; the recovered scale and offset were applied to the 60 m and 480 m grids and checked against OpenStreetMap for Mozambique, Pakistan and the UK at 60 m.]**

### 2. Severity is not scale-invariant, so it can only be studied on Australia

Fraction of windows per class, using every valid 128px patch:

| Region | Patches | none | minor | moderate | severe |
|---|---|---|---|---|---|
| Australia | 62 | ~15% | ~10% | ~8% | ~67% |
| UK | 2 | 0 | 0 | 0 | **100%** |
| Pakistan | 18 | 0 | ~0 | ~0 | **~100%** (5 minor, 23 moderate of 6,048) |
| Mozambique | 1 | 8% | 6% | 3% | 84% |

A 128px patch at 480m covers ~61km x 61km, so its max depth is nearly always >2m; the UK domain is small
and always severe. Any "always severe" model would score ~83-100% outside Australia.

### 3. The first notebook's sampler and split were broken

Its "Australia + Pakistan" training set was **100% Australia** (the sampler filled every class from
Australia, then broke out before reaching Pakistan); its held-out set was 60 UK "severe" + 12 Mozambique;
and its split was random over windows drawn from the *same* patches (train/validation leakage). Now:
per-region random sampling, **split by spatial position** (validation patches never seen in training).

### 4. The task is nearly degenerate at the native time step

Persistence = "the future equals the last observed frame". Measured on Australia:

| Time stride | Horizon | Persistence RMSE | On wet pixels | Severity from thresholding the last frame | Class changes |
|---|---|---|---|---|---|
| 1 (5 min) | 0.4 h | **2.0 mm** | 4.1 mm | **99.9%** correct | 0.1% |
| 3 | 1.2 h | 6.2 mm | 12.4 mm | 98.7% | 1.3% |
| **6 (30 min)** | 2.5 h | **13.5 mm** | 26.0 mm | 96.7% | 3.3% |
| 12 | 5.0 h | 28.4 mm | 55.4 mm | 94.3% | 5.7% |

Also, at stride 1 the five history frames span 25 minutes, but rainfall is sampled every 30 minutes - so
the rainfall channel was effectively constant. **`TIME_STRIDE = 6`** (30 min, rainfall's native resolution)
gives each step its own rainfall and a non-trivial horizon. Two consequences that shape every result: [round 2: the interval that contains the frame's time]
- any depth model must be judged **against persistence**, not just by RMSE;
- the severity label is ~a threshold on the current depth, so a capsule network cannot be shown to add value
  over "threshold the last frame" (96.7-99.9%) except on the few percent of windows where the class changes.
  This is the circularity flagged earlier, now quantified.

### 5. The architecture needed three fixes

1. **Persistence-anchored decoder.** Output = last observed frame + learned change; the final conv is
   zero-initialised so an *untrained model reproduces the last frame exactly* (verified: max difference 0.0).
   Training can only improve on persistence.
2. **Decoder skip connection.** With only the residual, held-out skill plateaued at **exactly +0.0205 from
   epoch 2 through 25, and +0.0207 on the never-seen UK** - and `depth_rate` was 0.0007 for *every* window.
   That is a region-independent constant drift, not learning: the decoder had only a 128-number summary and
   no way to say *where* change happens. Feeding the encoder's last-step spatial features into the decoder fixed it.
3. **Margin loss for the capsule head** (CapsNet paper, eq. 4) instead of softmax cross-entropy on lengths.

Also added: fixed per-channel input normalization inside the model (DEM in hundreds of metres next to
rainfall in mm previously fed the CNN raw; DEM is made patch-relative so it can transfer), a loss normalised
by persistence error (1.0 = exactly persistence, so depth and severity losses are commensurate), and cosine
learning-rate decay + gradient clipping (batch-size-1 Adam at constant 1e-3 swung validation by several
points epoch to epoch).

### 6. What the local runs showed (100 training windows, 25 epochs - smoke tests, not results)

| Run | Change | Australia held-out depth skill | UK (unseen) skill | Severity acc. (baseline 97.5%, chance 25%) |
|---|---|---|---|---|
| 2 | residual only | **+0.0205, flat for 23 epochs** | +0.0207 | 25% (38/40 "moderate") |
| 3 | + skip, + margin loss | +0.02 -> +0.09, noisy | +0.054 | 25% -> 60% |
| 4 | + cosine LR, + clip | +0.03 -> **+0.24**, stable over last 6 epochs | **+0.16** | 50% (train margin loss 0.085: overfit) |

Run 3 also showed `depth_rate` tracking physics: 0.0004-0.0007 m/hr in dry (0 mm) windows, 0.0031-0.0033 m/hr
with 4-6 mm/h of (hybrid-channel) rainfall - the model responding to its rainfall input [Correction, round 2: for the last frame that input includes rain that had not yet fallen; this shows input sensitivity, not learned flood dynamics], the signal the rule thresholds were
derived from. Absolute scale: RMSE 10.5 mm vs 12.0 mm for persistence.

### 7. Decisions this forced

- **Train on Australia only**; severity not scored on UK (census shows it is uninformative there).
- **Cross-regional test = Australia -> UK** (same 30m resolution), one held-out region - *not* the
  "train on two, test on two" originally planned, which is not achievable with this data.
- **Pakistan and Mozambique excluded** (misaligned layers, 480m vs 30m, ~100% severe).
- Every result reported next to a persistence baseline (depth) and a threshold baseline (severity).

### 8. Mistakes made along the way (kept, because they are the useful part)

- Asserted layer alignment after checking one region.
- Attributed blur and severity collapse to data volume when the architecture and loss were also at fault.
- Reported a rule-engine "success" whose input feature was noise.
- Two of my own test bugs this session: an untrained-model check run *after* an optimizer step, and a
  gradient check that summed two losses *outside* the tape (the same class of mistake as in Stage 3). Both
  were caught because the check contradicted a result that had already passed elsewhere.

**Added by the later runs and the September 2026 audit** (each corrected where it appears above):
- Wrote a notebook draft that passed a syntax check but was invalid in seven independent ways (found only by executing it).
- Chose CartoDB tiles as the fix for the OpenStreetMap block without looking at one; viewing a tile showed "API KEY REQUIRED".
- Quoted "+0.19" and "70% severity" as if stable; they came from one seed and a 100-window validation set (later: ~+-0.03 of
  pure non-determinism, severity 55-70%).
- Presented Grad-CAM "3.3x on water" from a single window; measured over many windows it is 1.1-1.5x and moves between runs.
- Adopted flip/rotation augmentation on one seed; the second seed reversed it.
- Wrote "Pakistan and Mozambique cannot be aligned" from size ratios without running the search (then: no verifiable alignment).
- Left the "two weeks of simulation" figure unqualified (it is the Pakistan event only).
- **Described the model as a CNN-LSTM that "captures temporal context" without ever testing it.** The audit's ablations show the
  depth skill is rainfall-driven and the depth history adds < 0.01. **[Correction, round 2: this describes the ORIGINAL model (v1) only. v1 could not use the depth history (each frame is squeezed through one 128-number dense layer and depth is divided by 5 m) and its last-frame rain channel contained look-ahead rain; later baselines showed the depth history IS informative and carries the skill of the final models ('Completion, round 2', sections 2, 7 and 8).]** New mistakes found in round 2 are listed in section 10 of the round-2 section.
- Left a NoData sentinel latent in the UK layers with no loader guard (no effect on results; guard added).
- Damaged this log while editing it (dropped a heading); caught while adding the audit and repaired.
- **Dropped Pakistan and Mozambique and called the project technically complete while 50% of the dataset was unused;** described the exclusion as
  "the only version the data allowed" when it was the only version I got working; ran the alignment search once, on one region, with a method that could not
  succeed; never opened the 60 m data; and ran a "full audit" that covered only the half in use. Corrected in "Completing the dataset".
- **Left a latent training-collapse mode in the recipe** (constant-start learning rate with Adam); the 30 m runs survived, the first 60 m run did not, and only the
  ablation exposed it because its reported skill (+0.045) looked like a small positive result. Fixed with warm-up plus a fail-loud guard.
- Sorted a probe's candidates by the wrong sign (the physical relation is negative) and got a false "the probe fails on known cases"; caught because
  the known cases are exactly what the probe is validated against.

---

## The Colab notebook now

`flood_forecasting_colab.ipynb`: pulls Australia + UK from Hugging Face (only the files those two regions
need), prints the class census, splits Australia by spatial position, trains on ~400 class-balanced windows
for 40 epochs (`@tf.function`-compiled), and reports **depth skill vs persistence** and **severity vs the
threshold baseline** on held-out patches (with the capsule's accuracy split by whether the threshold baseline
is right); Grad-CAM (one window plus a many-window statistic); the symbolic layer; a georeferenced prediction
on a real map (Esri imagery + a static fallback); and the Australia -> UK test with an automatic summary.
Regenerate it with `python build_notebook.py`; run the whole thing locally at small scale with
`python dryrun_notebook.py`.

**Defaults changed after the first Colab run:** `PER_CLASS_VAL = 60` (~240 validation windows instead of 100 -
a pure measurement improvement). `AUGMENT` exists but is **off** (see the A/B below). **To reproduce the first
Colab run exactly:** `PER_CLASS_VAL = 25`, `SEED = 42`, `AUGMENT = False`.

## First real-scale run (Colab, T4 GPU, TensorFlow 2.20.0)

Full notebook run, no cell errors. 4,385 files / 13.5GB pulled from Hugging Face in 7 min 43 s; training took
~10.5 s per epoch (40 epochs, ~7 min). Config: 400 class-balanced training windows (46 Australian patches),
100 balanced validation windows (25 per class, from 16 held-out patches), `SEED = 42`, `TIME_STRIDE = 6`.

| Result | Model | Baseline | Verdict |
|---|---|---|---|
| Depth, Australia held-out (balanced windows) | RMSE 10.03 mm | persistence 11.15 mm | skill **+0.190** |
| Depth, Australia (uniform windows) | RMSE 12.47 mm | persistence 14.17 mm | skill **+0.225** |
| Depth, **UK (never trained on)**, 136 windows | RMSE 17.50 mm | persistence 19.43 mm | skill **+0.189** |
| Severity, Australia held-out (balanced) | capsule **70.0%** | threshold-the-last-frame **90.0%**, chance 25% | **below the baseline** |

Skill = 1 - (model RMSE / persistence RMSE)^2, so +0.19 is ~19% lower squared error, ~10% lower RMSE
(1.1 mm in Australia, 1.9 mm in the UK). The reported skills match the reported RMSEs to within 0.0004.
The untrained model scored exactly +0.0000 at epoch 0. NSE (0.9989 vs 0.9986) and r cannot separate the
model from persistence - persistence is already that accurate - so skill is the metric that carries meaning.

**What this supports.** A model trained only on Australia beats "copy the last frame" in a region it never
saw by the same margin (+0.189 vs +0.190). **What it does not:** it is one run with one seed (late-epoch **[Correction, round 2: true for v1, but the bar is low: a per-pixel linear model of the depth history alone scores +0.15 (480 m) to +0.35 (30 m); the final models score +0.81 to +0.87 on never-seen regions, and a two-feature lookup table +0.74 to +0.79 ('Completion, round 2', sections 2, 7 and 8).]**
validation skill wobbled 0.13-0.19 within the run, and a later seed gave lower values - quote a range, see the A/B section); "held-out" in Australia means unseen *places* of the
*same storm*; the UK test is 2 patches / 136 strongly correlated windows; train 0.69 vs validation 0.81
(persistence-normalised) is a moderate generalisation gap. The absolute gain is millimetres because the
simulated flood barely changes over 2.5 h - a property of the data and horizon, not a defect.

**Severity, honestly.** 70% vs a 90% no-model baseline. Confusion matrix (rows true none/minor/moderate/severe):
`[[21,4,0,0],[8,16,1,0],[0,6,19,0],[0,1,10,14]]` - errors are between adjacent classes; only 14 of 25 severe
windows were recognised and 10 of the 11 misses were called "moderate" (it never called a non-severe window
severe). Train margin loss fell to 0.026 while validation accuracy was flat from ~epoch 15 - the head memorises
the 46 training patches. With 100 windows the accuracy has a standard error of ~4.6 points, so the 76% peak
vs 70% final is noise. The likely reason it cannot beat the baseline: the label *is* a depth threshold on the
patch's maximum, which the baseline reads directly and the capsule must learn from 400 windows (the
"routing dilutes a single deep pixel" idea from the notebook text is a hypothesis, not tested).

**Symbolic layer, honestly.** In the 8 demo windows the rules changed 0 classes. This is by design, not a bug:
escalation requires depth-rate > 0.0030 m/hr **and** > 1000 m from baseline water (case 1 had 0.0055 m/hr but
687 m). The layer here is an explanation layer; nothing shows it improves accuracy.

**Grad-CAM.** On that one (wettest) window: mean heat 0.40 on above-average-depth pixels vs 0.12 elsewhere
(3.3x), hottest spot on the deepest pond, at 16x16 resolution (blocky). **One window is an anecdote** - measured
over many windows the preference for water is much weaker (median 1.1x; see the A/B section). Not proof that
the explanation is causal.

**Georeferencing sanity check.** The written GeoTIFF (CRS EPSG:32756) spans lat -28.95..-28.92, lon
153.38..153.42 - northern New South Wales, where the Feb-Mar 2022 floods occurred. This confirms the coarse
location only, not pixel accuracy.

### Defects found in this run, and their fixes

- **Map showed a wall of "403 Access blocked".** folium's default tiles are OpenStreetMap's, whose servers
  answer a request with no `Referer` header with a "403 Access blocked" *image served as if it were a tile*
  (reproduced directly: identical picture). Colab's output frame sends no referrer. The obvious replacement,
  CartoDB's free tiles, now stamps "API KEY REQUIRED" across every tile (seen when the tile was viewed). The
  notebook now uses Esri World Imagery (identical bytes with and without a `Referer`; the notebook's own map
  HTML rendered satellite tiles in a browser) and adds a static, network-free plot. **Not verified inside
  Colab itself** - the in-app test browser sends a referrer and so did not reproduce the original block.
- The notebook intro claimed the pipeline was "verified locally at real scale" - it had only been run at tiny
  size. Reworded.
- The download was stated as ~15-16GB; measured 13.5GB. Corrected.
- `requirements.txt` pinned `folium==0.17.0`, but the tested version was 0.20.0. Corrected. (Colab ran TF 2.20.0
  and local runs use 2.21.0; the code ran on both.)

## Attacking the severity overfit: augmentation A/B (local CPU, full Colab size, two seeds)

The Colab run showed the capsule head memorising its 46 training patches (train margin loss 0.026, validation
flat). Hypothesis: with so few distinct patches the shared encoder can identify a patch from its terrain and
recall its label. Test: random flips / 90-degree rotations of each training window (`AUGMENT`), applied
identically to input frames and depth targets (unit-tested: all 8 orientations reachable, input and target
pixels land in the same place). Severity is invariant to these transforms.

**Criteria fixed before seeing any result:** adopt `AUGMENT = True` only if (a) severity accuracy improves by
>= 5 points on the enlarged validation set AND (b) Australia and UK depth skill are not worse by more than 0.02.

**Protocol.** The full notebook code (`dryrun_notebook.py` with Colab-size settings) on the laptop's CPU; only
`AUGMENT` differs between the arms of a seed (same training windows, same validation windows). Validation
enlarged from 100 to ~240 balanced windows (60 per class; accuracy standard error ~3 points, optimistic because
windows from one patch are correlated). Seed 42 uses the same training windows as the Colab run (persistence
MSE on them 1.027e-04 m^2, identical to the Colab print). Seed 7 draws *different* training windows and holds
out *different* patches (persistence MSE 7.52e-05 m^2; only 58 'moderate' validation windows existed, so 238).

| | Seed 42 no aug | Seed 42 **aug** | Seed 7 no aug | Seed 7 **aug** |
|---|---|---|---|---|
| Severity accuracy | 62.9% | **81.2%** | 54.6% | **68.1%** |
| Threshold-baseline accuracy (same windows) | 92.9% | 92.9% | 92.4% | 92.4% |
| Severe-class recall | 45% | 88% | 52% | 92% |
| Depth skill, Australia (balanced) | +0.169 | +0.207 | +0.140 | **+0.051** |
| Depth skill, Australia (uniform) | +0.163 | +0.217 | +0.077 | +0.071 |
| Depth skill, UK (unseen) | +0.204 | +0.202 | +0.206 | **+0.044** |
| Capsule accuracy where the class changes (n=17/18) | 59% | 65% | 22% | 39% |
| Grad-CAM: windows with heat higher on wet pixels | - | - | 69% | 74% |

**Outcome.** (a) is met on both seeds (+18.3 and +13.5 points). (b) is met on seed 42 and **failed on seed 7**:
Australian depth skill fell by 0.089 and UK skill by 0.162. In the seed-7 augmented run the training depth loss
*rose* from 0.79 (epoch 30) to ~0.91 (epochs 36-40) while the learning rate was decaying - a late instability that
the last-epoch report captured (skill was +0.14 at epoch 25-30, then +0.03 / +0.05). Whether that is a fluke of
one seed or a systematic property of augmented training is unknown with two seeds. **Decision: `AUGMENT` is off
by default** (the adoption rule was not satisfied robustly); it stays available as a switch. The severity gain
is real in direction on 2 of 2 seeds, but severity remains far below the threshold baseline either way (54-81%
vs 92-93%), so it does not change the story.

**Other things this showed.**
- The capsule is less accurate on windows where the class changes over the forecast than on the rest, in all
  four runs (e.g. 22% vs 57%). There is **no evidence the capsule anticipates escalation** rather than reading
  the current depth imperfectly.
- Severity accuracy without augmentation varies with seed and validation set: 54.6%-70.0% across the three
  no-augmentation runs. The Colab run's 70% was the top of that range; quote "roughly 55-70%".
- Depth skill also varies with seed and validation set (Australia +0.05 to +0.23 across the five runs). It is
  positive in all five runs and both regions. The most stable single number is UK skill for the no-augmentation
  runs: +0.189, +0.204, +0.206 (identical UK windows every time, so this isolates model-to-model variation).
- **Grad-CAM, measured properly.** Over 39 validation windows containing water, heat was higher on wet than on dry
  pixels in 69% (no aug) / 74% (aug) of windows (50% is chance), but the **median wet/dry ratio is only 1.1x**.
  The single "wettest window" reads much stronger (4.6x, 7x here; 3.3x on Colab) because it is the extreme. So
  the honest claim is a mild, better-than-chance preference for wet pixels - not the "3.3x" from one window.
  These maps also explain very small predicted changes (0.1-0.3 mm in the local windows; Colab's window was 7 mm).

## Second Colab run: the default notebook (seed 42, 240 validation windows, `AUGMENT = False`)

Run after the fixes above. No cell errors; 4,385 files in 4 min 39 s; ~12 s per epoch on the T4. Training windows
identical to the first Colab run (persistence MSE 1.027e-04 m^2) and to the local no-augmentation run; validation
enlarged to 240 balanced windows.

| Seed 42, no augmentation | First Colab (100 val) | Local CPU (240 val) | **This Colab run (240 val)** |
|---|---|---|---|
| Depth skill, Australia (balanced) | +0.190 | +0.169 | **+0.191** |
| Depth skill, Australia (uniform) | +0.225 | +0.163 | **+0.172** |
| Depth skill, UK (unseen) | +0.189 | +0.204 | **+0.201** |
| Severity accuracy (baseline) | 70.0% (90.0%) | 62.9% (92.9%) | **62.5% (92.9%)** |
| Severe-class recall | 56% | 45% | 55% |
| Capsule accuracy where the class changes / where it doesn't | - | 59% / 63% | 59% / 63% |

- **The local CPU run is a faithful proxy of Colab**: same data, same windows, results within ~0.02 skill and 0.4
  points of severity accuracy across different hardware and TensorFlow versions (2.21 vs 2.20).
- **Same seed, same data, two Colab runs**: UK skill +0.189 vs +0.201 - GPU non-determinism alone moves the
  number by ~0.01. Do not over-read differences smaller than that.
- **The first run's 70% severity was the small-validation-set number**; with 240 windows the same configuration gives
  62.5%. Quote roughly 55-70% across runs, below the ~92-93% threshold baseline.
- **Grad-CAM**: heat higher on wet than on dry pixels in 74% of 35 windows, median wet/dry ratio 1.3x (chance 50% /
  1.0x). But the single displayed map differs from the first Colab run's even with the same seed: this time the heat
  sits on the broad shallow flood zone and the deep ponds are cold; last time the hottest spot was a deep pond. The
  explanation is a mild preference for water whose *location* is not stable across training runs.
- **Symbolic layer, demo windows**: the rules changed 1 of 8 classes - and it **broke a right answer**
  (true moderate, raw moderate, escalated to severe on depth-rate 0.0033 m/hr at 1916 m from baseline water). The
  other 7 were untouched. Eight windows cannot say whether the layer helps, so the notebook gained a cell that
  measures it over all validation windows. **Result (same trained model, all 240 balanced validation windows):
  capsule alone 62.5%, capsule + rules 62.5%. The rules changed the class in 8 of 240 windows (3%): 4 fixed a wrong
  answer, 4 broke a right one.** Net effect on accuracy: zero. With only 8 changes the 4/4 split is
  indistinguishable from a coin flip (sign test p = 1.0), so this shows "no evidence it helps", not "proven
  harmless". It also cannot touch the capsule's main error (26 of 60 severe windows called moderate) because the
  escalation rule needs both a rising depth-rate above 0.0030 m/hr and > 1000 m from baseline water, which is rare.
- **Map**: the screenshot shows real Esri imagery inside Colab (attribution visible), so the tile block is fixed
  there. The view shown was tropical islands, not northern NSW - most likely the map had been moved by scrolling
  or dragging over it (Leaflet zooms on the scroll wheel), but this was not confirmed. The map now frames the patch
  (`fit_bounds`), outlines it in red, disables scroll-wheel zoom, and sqrt-scales overlay opacity so shallow water is
  visible.

### Georeferencing checked against the real world

`verification_outputs/georeferencing/georef_check.py` overlays the *observed* flood depth of the mapped patch (row
256, col 640) on Esri satellite imagery, using the notebook's georeferencing (UTM 56S, world-file origin shifted by
half a pixel). Result (`georef_check.png`): the thin blue river channel in the depth raster lies on the river visible in
the imagery, and the edge of the broad flooded area follows the curved channel/embankment around the floodplain.
This is a visual check at the figure's resolution (features agree to within a few tens of metres), not a measured
pixel offset. It supports the world-file / half-pixel handling for Australia at 30 m; nothing here covers the
other regions. **[Correction, round 2: this check covers Australia at 30 m only; the coarse regions were checked separately against OpenStreetMap ('Completion, round 2', section 4).]**

## Independent audit (September 2026): re-checking everything from raw files

A full re-check, written to find mistakes rather than confirm the story. Scripts and logs are in
`verification_outputs/audit/`. Data-level checks (`audit_data_checks.py`, no model involved) all pass:

| Check | Result |
|---|---|
| Raw rasters: NaN / negative / sentinel values | Depth, DEM and rainfall are clean **inside the aligned crop**, in both regions (see NoData finding below). |
| Train/validation split (seeds 42 and 7) | Disjoint tiles; tiles are non-overlapping, so no shared pixels. |
| Window arithmetic | History strictly precedes the forecast; uniform 30-min spacing; one new rainfall file per step; a window running past the end is rejected. **No future DEPTH enters an input.** [Correction, round 2: rainfall did - each frame's rain channel is the interval that contains it, so the last observed frame carried up to 30 minutes of rain that had not yet fallen; this audit checked file indices, not what each rain file covers in time.] |
| Loader vs independent reads | 24 random windows (12 Australia, 12 UK): every input array, target array, rainfall channel, DEM channel, severity label and persistence-baseline label equals a direct raw read. |
| Window scan vs independent census | 186 windows: every severity and baseline label matches. |
| UK (5,5) offset | Strongest of all 121 candidate offsets again (r = −0.252 aligned vs +0.007 unaligned). |
| `verify_stage1`, `verify_stage3_capsule`, `verify_stage3_combined` | Pass with real exit codes; `verify_stage1` reproduces its original numbers (0.270 dry / 10.964 wet). |

**Findings, in order of importance**

1. **NoData ring in the UK layers (latent hazard, no effect on results).** The UK DEM and every rainfall file carry the
   float32 NoData sentinel (−3.4×10³⁸) in a padding ring *outside* the depth-aligned crop (1,452 DEM pixels; 145 of 145
   rainfall files). No window this project reads touches it (checked over every file), so the UK results are unaffected.
   But the loader had no guard, and the flat-terrain filter would have kept a NoData patch as "valid" (its DEM standard
   deviation is enormous). **Fixed:** `real_data_loader.has_nodata()`; `load_patch_sequence` now raises on any NoData/NaN
   window and `valid_positions` drops such patches. The used patches are unchanged (Australia 62, UK 2). The ring lying exactly
   outside the crop is a third independent confirmation of the UK offset.
2. **"Pakistan and Mozambique cannot be aligned" was overstated.** It was inferred from non-integer size ratios; the
   correlation search used for the UK was never tried. Tried now for Mozambique (block-average by 16 plus crop, every
   sub-block phase and offset, 5,985 combinations): **no distinguishable peak** (best r = −0.174 against a median of −0.147),
   because a large coastal floodplain is uniformly low and wet and a 480 m shift barely changes the correlation. So the honest
   statement is "we could not verify an alignment", not "impossible". (Pakistan was not tested.) Wording corrected in the
   deck and explainer. **SUPERSEDED - see "Completing the dataset" below:** this first search removed no terrain trend and was
   the wrong method; a validated high-pass probe later recovered the geometry of both regions (and the 60 m data), and both
   are now used.
3. **The research-gap claim was checked against the paper and holds.** The FloodCastBench paper (Sci Data 12, 431) benchmarks
   exactly **U-Net, FNO and FNO+**; no LSTM, ConvLSTM or CNN-LSTM. Events, years, resolutions, the Australia period (20 Feb –
   2 Mar 2022) and the rainfall/depth time steps also match. Two refinements: the "two weeks on an A6000" simulation time is
   for the **Pakistan** event only (Australia and Mozambique about two days, UK about four hours); and the paper's own
   high-fidelity transfer test is **Australia → UK**, i.e. our direction (its low-fidelity one is Pakistan → Mozambique). The
   Norfolk claim (Wang et al., J. Hydrol.: Reg. Stud. 64, 103234, 2026: "four to six hours using TUFLOW to 3.2 min per event
   using CNN-LSTM") was also confirmed.
4. **Rule thresholds: right scale, loose calibration** (`audit_rule_calibration.py`, data only). Ground-truth patch-mean
   depth-rates at the 5-minute step and at the 30-minute stride agree (moderate 0.0016 vs 0.0018 m/hr), so normalising "per
   hour" does make the rule comparable across strides. But the thresholds came from 40 windows (10 per class); over all
   Australian windows the true rate separates classes only weakly (medians none 0.0000 / minor 0.0000 / moderate 0.0006 /
   severe 0.0004 m/hr; only ~22% of moderate and of severe windows exceed the 0.0030 escalation threshold, ~6% of minor do), and
   15–17% of truly severe windows have a receding rate below the de-escalation threshold. The layer is a heuristic, which is
   consistent with the 4-fixed / 4-broke result.
5. **Development-set hygiene.** The architecture fixes (residual decoder, skip connection, margin loss, cosine LR) were chosen
   while watching validation and UK skill in local runs, and the same 16 validation patches and the same 136 UK windows were
   used throughout. The augmentation decision *was* pre-registered and did not favour adoption, and seed 7 drew different
   training windows and held-out patches, but the UK set never changed. So the reported numbers are **development-set numbers,
   not a clean final test**, and the UK skill in particular should be read that way.

### Behavioural audit of a trained model (`audit_after_training.py`, full size, CPU, seed 42, default settings)

One more full-size run with the audit attached (log: `full_run_with_audit.log`). Headline numbers, in line with the earlier
runs: Australia skill **+0.217** (uniform windows +0.215), UK **+0.217**, severity **65.4%** vs the 92.9% threshold baseline;
symbolic layer 65.4% -> 65.8% (3 of 240 changed: 2 fixed, 1 broke); Grad-CAM heat higher on wet pixels in 93% of 27 windows,
median 1.5x. **Noise floor:** the same seed and the same 240 validation windows gave Australia skill +0.169 (CPU, 6 threads),
+0.191 (Colab) and +0.217 (CPU, 12 threads), so identical configurations differ by about +-0.03 through floating-point
non-determinism alone.

| Check | Result |
|---|---|
| [1] Headline skill recomputed independently from raw predictions in numpy | +0.2173 vs +0.2173 (difference 0.00000) |
| [2] Metric code on known answers | truth -> skill +1.000; copy vs copy -> 0.000; RMSE(truth) = 0 |
| [3] Fresh untrained model vs last observed frame, 10 real windows | worst difference 0.0 m |
| [6] Predicted depth below zero | 0.00% of pixels, minimum -0.000 m: no clipping needed |

**[4] Skill by lead time** (30-minute steps, 30 min ... 2.5 h): Australia +0.121, +0.198, +0.211, +0.221, +0.225; UK +0.214,
+0.220, +0.223, +0.220, +0.213 (persistence RMSE grows 3.1 -> 14.2 mm in Australia, 6.0 -> 29.1 mm in the UK). The gain is a
roughly constant ~20% reduction in squared error at every lead time, so the forecast does not fall apart as it extends.

**[5] What does the trained model use?** (one input removed from the test data only)

| Input removed | Australia skill | UK skill | Severity accuracy |
|---|---|---|---|
| none | +0.217 | +0.217 | 65.4% |
| **rainfall** | **+0.013** | **+0.004** | 67.5% |
| terrain (flat DEM) | +0.216 | +0.213 | **52.5%** |
| depth-history order scrambled (last frame kept) | +0.217 | +0.217 | 65.4% |
| depth history frozen (all frames = the last) | +0.209 | +0.213 | 65.4% |

**Reading.** The depth skill is **rainfall-driven**: without rainfall the model is essentially persistence. Terrain and the
*temporal history* of depth contribute less than 0.01. The forecast is, in effect, "last frame + current rainfall -> change".
Consequences: (a) the claim that the LSTM "captures temporal context" is **not supported** - the decoder LSTM does unroll the
per-step change, but the encoder's history is not being exploited; (b) the UK transfer is best described as transfer of a
rainfall-to-depth-change relationship, not of learned flood dynamics; (c) the **severity head does use terrain** (-13 points
without it) and not rainfall, which fits the overfitting story (terrain can act as a patch fingerprint) but does not prove it.
Caveats: this is test-time removal on one trained model - it measures reliance, not what a differently trained model could
learn - and the rainfall channel is near-uniform across a patch, so it can also act as a storm-phase clock. **[Correction, round 2: this describes the ORIGINAL model (v1) only. v1 could not use the depth history (each frame is squeezed through one 128-number dense layer and depth is divided by 5 m) and its last-frame rain channel contained look-ahead rain; later baselines showed the depth history IS informative and carries the skill of the final models ('Completion, round 2', sections 2, 7 and 8).]**

**[7] Severity vs reading the last frame:** both right 147, only the capsule right 10, only the threshold right 76, neither 7
(n = 240). The capsule's predicted class correlates +0.681 with the last frame's maximum depth (the true label: +0.748): it is
a noisy imitation of the threshold it is trying to approximate.

## Completing the dataset: Pakistan, Mozambique and the 60 m data (September 2026)

**Why this section exists.** The pipeline described above read **13.47 GB of a 26.87 GB dataset - 50%** (Australia and UK at 30 m).
Two of the four regions had been dropped in the pre-flight and then described as "the only version the data allowed" and
"cannot be aligned"; the 60 m versions of Australia and the UK (3.35 GB) had never been opened; and the "full audit" audited only
what was used. That was an incomplete job presented as a complete one. This section is the correction.

### 1. What had been used

| Part of the dataset | Size | Used before? |
|---|---|---|
| Depth 30 m: Australia (2,881 frames), UK (865) + their rainfall, DEM, land use | 13.47 GB | yes |
| Depth 60 m: Australia (2,881), UK (865) | 3.35 GB | **no** |
| Depth 480 m: Pakistan (4,033), Mozambique (1,729) | 5.90 GB | **no** |
| Rainfall / DEM / land use / initial conditions of Pakistan and Mozambique | 3.72 GB + 0.4 GB | **no** |

### 2. Whole-dataset integrity audit (`audit_full_dataset.py`: every one of the 14,868 files read)

Every depth frame of all six region/resolution sets: numeric names, exactly even 300 s spacing, one shape per set, float32, **no
NaN, no NoData, no negative depth** (ranges: Australia 30 m 0-19.2 m, 60 m 0-15 m; UK 0-6.1 / 0-4.9 m; Pakistan 0-26.7 m;
Mozambique 0-12.4 m). Rainfall: no NaN or negative values anywhere; the NoData sentinel appears only in the padding ring of the
UK, Pakistan and Mozambique rainfall and terrain rasters (all files); Australia's are clean (rainfall up to 26.3 mm/h [round-2 correction: mm/h intensities, about 13 mm per 30 min],
Pakistan up to 56.8 mm/h). The four regions' world files (DEM, land use, georeferenced) are identical to each other and describe the
**30 m** grid. Nothing else anomalous. Event dates: the first rainfall file of each event falls on the paper's stated start date and
the rainfall duration equals the depth duration (Pakistan 14 d, UK 3 d, Australia 10 d, Mozambique 6 d), so the rainfall-to-depth
alignment is confirmed **at date level** (the hour is still inferred). Pakistan's rainfall has 672 files = exactly 14 days (interval
starts), the others N+1; the loader's floor mapping is correct for both. **[Correction, round 2: alignment is also confirmed physically at 30-minute resolution (lag 0). The floor mapping pairs each frame with the interval that CONTAINS it: a correct alignment, but for the last observed frame that interval is partly future.]**

### 3. What the paper actually specifies (read from its open-access text)

Baselines are exactly U-Net, FNO and FNO+ (no LSTM). Three transfer tests: **Pakistan -> Mozambique at 480 m**, **Australia -> UK at
60 m**, and UK at 30 m as zero-shot downscaling. The coarse grids are the 30 m results "resampled ... using bilinear interpolation".
Its task is different from ours: inputs are coordinates plus the initial water depth (FNO+ adds terrain and the rainfall series) and
the model receives the true depth at t=1 of a 20-frame sample (300 s spacing) and predicts t=2..20 (19 steps) over the whole domain (FNO+ also receives terrain and the rainfall series); ours is 5 observed frames -> 5 forecast frames at 30-minute spacing on
128-px patches. Its Table 5 (relative error - unitless, NOT metres - / NSE): Pakistan -> Mozambique FNO+ 0.0786 / 0.955, FNO 0.163 / 0.453, U-Net 1.60 / -72.4;
Australia -> UK (60 m) FNO+ 0.0248 / 0.9989, FNO 0.0290 / 0.9986, U-Net 0.525 / 0.690. **Our numbers are not comparable to these**
(different task and horizon), and we say so wherever both appear.

### 4. Recovering the geometry of the coarse grids

The first Mozambique search assumed an integer x16 block-average plus crop and correlated depth with terrain directly; it found
nothing because the large-scale terrain trend swamps local structure (the correlation barely changes with a 1-cell shift). The
**probe was rebuilt**: high-pass both the flood footprint (max depth over the event) and the terrain, then correlate. It was
**validated on cases whose answer is known before being trusted**: Australia 30 m -> offset (0,0) at r = -0.28 (19.8 robust-sd), UK 30 m -> (5,5)
at r = -0.41 (3.6 robust-sd); a water-body probe also re-found the UK offset (r = +0.22 at (5..6,5..6)).

| Region / resolution | Recovered geometry (base pixels per cell, crop offset) | Evidence |
|---|---|---|
| Mozambique 480 m | 16, (150, 50) | terrain-structure r = -0.270, plateau of +-2 px, 5.2 robust-sd below the median; 0 of 104,936 DEM NoData pixels fall inside the crop; the main river channel visibly lines up (`multires/alignment_evidence.png`) |
| Pakistan 480 m | 16, (42, 178) | terrain-structure r = -0.091 (5.5 robust-sd) **and** an independent baseline-water probe r = +0.101 agree on the same neighbourhood; 0 of 1,934,218 NoData pixels inside. **Weaker than Mozambique** (the flood covers 69% of the domain, so there is little terrain contrast); the two signals agree to about half a cell |
| Australia 60 m | 2, (0, 0) | terrain-structure r = -0.258 at (0..1, 0..1), 4.5 robust-sd |
| UK 60 m | 2, (5, 5) | terrain-structure r = -0.449 at (5..6, 5..6) |

Precision is about +-2 base pixels (60 m), under 6% of a 480 m cell; the NoData test is a consistency constraint, not a sharp locator.
This precision cannot matter for the experiments: rainfall's native resolution is ~10 km, and the depth branch does not use terrain [round 2: v1's ablation; removing terrain also changes the skill of the newer models by less than 0.01]
(section on ablations).

### 5. The loader was generalised and re-tested

`real_data_loader.LAYER_GEOMETRY` holds (scale, offset) per (region, resolution); DEM and rainfall are block-averaged to the depth cells
(land use: cell-centre pixel); a combination with no registered geometry is refused. New audit check A8: for each coarse region,
random windows from the loader equal an independent block-average of the raw 30 m files (all four pass). The 30 m results are unchanged
(A4, A6, A7 and `verify_stage1`: dry 0.270, wet 10.964). The NoData guard also applies to the coarse layers. `depth_experiment.py` runs the
two coarse transfer tests (depth only: at these scales the severity label is degenerate - a 128-px patch spans 7.7 km / 61 km);
`flood_forecasting_multires_colab.ipynb` reproduces them on Colab (all cells executed locally first).

### 6. The 60 m Australia model collapsed - diagnosis and fix

The first 60 m run (Australia -> UK) ended with skill +0.045 (held-out) and +0.016 (UK), and **every ablation gave the identical
number**: the network had collapsed to an input-independent constant offset (its 1 mm CSI was 0.77 against 0.97 for persistence, since a
constant bump makes every dry cell "wet"). It was **not a finding about 60 m data**. Diagnosis, all measured:

1. *Not a heavy tail*: the share of squared change carried by the top 5% of windows is the same at 30 m and 60 m Australia (46% / 42%).
2. *Not a missing signal*: in the 60 m training windows rain correlates +0.79 with the signed change and +0.88 with its magnitude (Pakistan +0.81 / +0.77).
3. *Reproduced*: re-running the recipe on the same 400 windows, by step 100 the predicted change had **std 0.000 mm across windows**
   and the active-unit fractions were frozen at exactly 0.34 / 0.36 / 0.31 / 0.50; an 80-window subset had stayed alive. The trigger is a few
   extreme windows (rain intensity 26 mm/h, depth 15 m) hitting Adam at full learning rate on the first steps.
4. *Fixed*: a 200-step linear learning-rate warm-up (`WarmupCosine`) keeps the same 400-window run alive (predicted change std 1-3.7 mm; healthy unit fractions).

Two safeguards were added so this cannot pass silently again: `build_and_train` raises `DEAD NETWORK` if the predicted change is identical for
every validation window after epoch 2, and the main notebook's ablation cell prints a warning when skill is identical whichever input is removed
(tested on a tiny input-blind model). **The 30 m runs were not affected** (their ablations respond strongly to rainfall) but they used the
no-warm-up recipe, so the collapse risk existed there too; they were not re-run. The v1 (no-warm-up) outputs are kept as
`*_recipe_v1_no_warmup.*`; both coarse experiments were re-run under the warm-up recipe so they are comparable.

### 7. Results (seed 42, 400 training windows, 40 epochs, warm-up recipe; depth only; skill = 1 - MSE_model / MSE_persistence)

| Experiment / evaluation | Skill | RMSE model / persistence (mm) | CSI(1 mm) model / persistence | CSI(0.1 m) model / persistence | Skill by lead time (30 min ... 2.5 h) | Windows |
|---|---|---|---|---|---|---|
| Pakistan -> Mozambique, 480 m: Pakistan held-out tiles | **+0.089** | 14.0 / 14.7 | 0.965 / 0.993 | 0.989 / 0.989 | +0.09  +0.09  +0.08  +0.08  +0.09 | 150 |
| Pakistan -> Mozambique, 480 m: **Mozambique (never seen)** | **+0.047** | 16.7 / 17.1 | 0.916 / 0.975 | 0.976 / 0.974 | +0.04  +0.04  +0.04  +0.05  +0.05 | 238 |
| Australia -> UK, 60 m: Australia held-out tiles | **+0.410** | 6.5 / 8.5 | 0.958 / 0.974 | 0.991 / 0.990 | +0.21  +0.39  +0.41  +0.42  +0.42 | 150 |
| Australia -> UK, 60 m: **UK (never seen)** | **+0.176** | 14.2 / 15.7 | 0.954 / 0.963 | 0.974 / 0.973 | +0.16  +0.16  +0.17  +0.18  +0.18 | 120 |

**What the trained models use** (skill with one input removed at test time; columns: none | rainfall | terrain | history order scrambled | history frozen):

| Evaluation | none | rainfall removed | terrain removed | history scrambled | history frozen |
|---|---|---|---|---|---|
| Pakistan held-out (480 m) | +0.089 | +0.003 | +0.107 | +0.089 | +0.087 |
| Mozambique (480 m) | +0.047 | +0.001 | +0.040 | +0.047 | +0.048 |
| Australia held-out (60 m) | +0.410 | +0.018 | +0.410 | +0.410 | +0.409 |
| UK (60 m) | +0.176 | +0.003 | +0.170 | +0.176 | +0.176 |

For reference the first Pakistan -> Mozambique run, **before** the warm-up fix (it did not collapse), gave +0.122 held-out and +0.067 on Mozambique - i.e. the warm-up recipe moved the numbers by about 0.03, the same size as the run-to-run noise floor measured at 30 m.

**Reading.**
- **Depth beats "copy the last frame" in all four evaluations** - the held-out tiles and the unseen region, at 480 m and at 60 m - so both of the paper's transfer directions now have a result: Pakistan -> Mozambique is small but positive; Australia -> UK at 60 m is clearly positive (about the size of the 30 m UK result). **[Correction, round 2: true for v1, but the bar is low: a per-pixel linear model of the depth history alone scores +0.15 (480 m) to +0.35 (30 m); the final models score +0.81 to +0.87 on never-seen regions, and a two-feature lookup table +0.74 to +0.79 ('Completion, round 2', sections 2, 7 and 8).]**
- **Rainfall drives the skill at every resolution.** Removing rainfall takes every evaluation to about zero; terrain and the depth history change it by less than 0.02 (Pakistan actually improves slightly without terrain). Together with the 30 m audit that is **ablations on trained models at three resolutions and four regions**, all saying the same thing: the forecast is "last frame + current rainfall", and the LSTM's history is not exploited. **[Correction, round 2: this describes the ORIGINAL model (v1) only. v1 could not use the depth history (each frame is squeezed through one 128-number dense layer and depth is divided by 5 m) and its last-frame rain channel contained look-ahead rain; later baselines showed the depth history IS informative and carries the skill of the final models ('Completion, round 2', sections 2, 7 and 8).]**
- **Australia at 60 m, seed 42, scores +0.41 - about twice the 30 m skill.** This turned out to be one seed's favourable set of held-out tiles; the second seed (section 8 below) gives about +0.24, in line with 30 m. Do not quote +0.41.
- **At the paper's 1 mm wet threshold the model is worse than persistence in every evaluation** (see CSI(1 mm)): the anchored decoder leaves millimetre-scale noise in cells persistence keeps at exactly zero. At 0.1 m it is equal or slightly better. This is a genuine weakness of the output, not of the metric. **[Correction, round 2: a statement about v1.]**
- **Not comparable to the paper's tables** (FNO+ relative error 0.0786 Pakistan -> Mozambique, 0.0248 Australia -> UK at 60 m, unitless and not metres; plain persistence scores as well or better on most of the paper's own measures, see 'Completion, round 2', section 3): the paper forecasts 19 steps from the initial state over the whole domain; this forecasts 5 steps from 5 observed frames on 128-px patches, and is judged against persistence.
- **Caveats.** This table is seed 42 (a second seed is in section 8 below); Mozambique is one storm with 4 heavily overlapping tiles; the UK at 60 m (85x137) is smaller than one patch, so it is scored on padded windows and real pixels only; the coarse-region numbers are development-set numbers like the 30 m ones; the 30 m experiments were produced without the warm-up and were not re-run.

### 8. Replication with a second seed (seed 7: different training windows and different held-out tiles; same unseen regions)

| Evaluation | Skill, seed 42 | Skill, seed 7 | Rainfall removed (42 / 7) | CSI(1 mm) model / persistence (seed 42; seed 7) | CSI(0.1 m) model / persistence (seed 42; seed 7) |
|---|---|---|---|---|---|
| Pakistan held-out tiles (480 m) | **+0.089** | **+0.098** | +0.003 / +0.019 | 0.965 / 0.993; 0.959 / 0.990 | 0.989 / 0.989; 0.994 / 0.993 |
| **Mozambique, never seen** (480 m) | **+0.047** | **+0.074** | +0.001 / +0.008 | 0.916 / 0.975; 0.925 / 0.975 | 0.976 / 0.974; 0.977 / 0.974 |
| Australia held-out tiles (60 m) | **+0.410** | **+0.235** | +0.018 / -0.007 | 0.958 / 0.974; 0.907 / 0.942 | 0.991 / 0.990; 0.987 / 0.986 |
| **UK, never seen** (60 m) | **+0.176** | **+0.183** | +0.003 / +0.005 | 0.954 / 0.963; 0.943 / 0.963 | 0.974 / 0.973; 0.974 / 0.973 |

**Reading the two seeds together.**
- **Every evaluation beats persistence on both seeds.** Ranges: Pakistan held-out +0.089 to +0.098; **Mozambique (unseen) +0.047 to +0.074** (the first run, before the warm-up fix, gave +0.067); Australia 60 m held-out +0.235 to +0.410; **UK 60 m (unseen) +0.176 to +0.183**. Quote ranges, not single numbers.
- **The unseen-region results are the stable ones.** The UK at 60 m moves by less than 0.01 between seeds; the held-out *Australian* tiles move by 0.18 because each seed holds out different tiles (4 of 16), and some tiles are much easier to beat persistence on than others. This also corrects the seed-42-only remark above: Australia 60 m is **not** twice as skilful as 30 m - seed 42's +0.41 was a favourable set of held-out tiles; seed 7 gives +0.235, in line with the 30 m runs (+0.05 to +0.23).
- **The mechanism replicates on every one of the eight trained-model evaluations (4 evaluations x 2 seeds):** removing rainfall takes skill to about zero (between -0.01 and +0.02); removing terrain, scrambling the depth history and freezing it change it by less than 0.02 (scrambling/freezing: less than 0.01). With the 30 m audit, that is the same finding at three resolutions and four regions. **[Correction, round 2: this describes the ORIGINAL model (v1) only. v1 could not use the depth history (each frame is squeezed through one 128-number dense layer and depth is divided by 5 m) and its last-frame rain channel contained look-ahead rain; later baselines showed the depth history IS informative and carries the skill of the final models ('Completion, round 2', sections 2, 7 and 8).]**
- **CSI:** worse than persistence at the paper's 1 mm wet threshold in all eight cases (the model leaves millimetre-scale noise in dry cells that persistence keeps at exactly zero); equal or marginally better at 0.1 m. Reported as measured; not fixed here.
- **No dead networks:** the guard did not fire in either replication (all four evaluation sets have input-dependent outputs, confirmed by the ablations).
- **Still true:** one storm per unseen region; Mozambique's 4 tiles overlap heavily; UK 60 m is smaller than one patch (padded windows); development-set numbers; the 30 m runs were not re-run with the warm-up; the paper's tables are not comparable.

<!-- ROUND2 START -->
## Completion, round 2 (20 September 2026): what a second audit found, and the study that replaced the first results

**Why this section exists.** After the dataset was completed (previous section) the remaining "not established" items were worked through one by one. Doing so showed that
several of the earlier *results and statements* were wrong or misleading. The corrections below are the most important part of the project record. Everything here is
reproducible from `completion/` (scripts, logs, result files) and was independently re-checked (section 6).

### 1. Rainfall timing: measured, and the "no future information" claim was wrong for rain

`completion/a_rain_lag.py` correlates, for every 30-minute interval `j` of a simulation, the domain-mean depth GAIN over that interval with the domain-mean rainfall of file `j + k`.
In all four regions the correlation peaks at lag k = 0 (0.9998 in Australia, 0.9996 in Mozambique) and the regression slope of depth gain (mm) on the file value is 0.497 (Australia),
0.502 (UK), 0.501 (Mozambique), 0.494 (Pakistan): the files are **mm/h intensities**, and file `j` is the rain that **falls during** `[1800 j, 1800 (j+1))` seconds of the simulation.
The original loader gave the frame at time `t` the file `floor(t/1800)`, i.e. the interval that CONTAINS `t`; for the last observed frame that is rain falling AFTER the observation
(up to 30 minutes of it). Earlier statements "no future information enters any input" (verified for depth only) and "rainfall hour inferred" are therefore corrected: the alignment is now
confirmed physically at 30-minute resolution, and the original input was a **hybrid** of past and future rain.
Three well-defined input regimes are used from here on (same architecture, same recipe): **causal** (only the last completed interval), **hybrid** (the original), **forced** (hybrid plus the
following 5 rainfall files; a rain-forced surrogate, the setting the paper's FNO+ works in). Independent re-computation from the raw files (agent report `completion/review/independent_verification.md`)
reproduced the correlations and slopes to ~1e-6. Caveats found there: the Australia statistics use the 394 of 480 intervals with activity (using all 480: r = 0.99979, slope 0.4972);
the "total rain mm" fields of the JSON files are sums of mm/h intensities (mislabelled); Mozambique's depth gain exceeds the accumulated rain by ~4.5% (a constant 0.032 mm per 30 minutes non-rain source).

### 2. The baseline ladder: the original CNN-LSTM was beaten by a straight line

Persistence ("copy the last frame") was the only baseline used before. `completion/b_baselines.py` adds, scored on exactly the windows the neural models use: S1 a per-pixel linear model of the
depth history only (state + last change + change over the history), S2 the same with rainfall and terrain but no trend, S3 rain + state + trend, S5 a **two-feature lookup table** (mean future
change per bin of current depth x last one-step change), S4 a small **per-pixel MLP** (no convolution, no recurrence); "c" versions clip at zero depth (clipping changes nothing).
First fold-0 results: S1 (depth history only, linear) skill +0.34 on the never-seen UK at 60 m and +0.13 on Mozambique against +0.18 / +0.05 for the original CNN-LSTM; the per-pixel MLP scores
~+0.86 (UK 60 m) and ~+0.84 (Mozambique). The original model could not use the depth history: it flattened each frame's features into ONE 128-number dense layer before the LSTM and scaled depth
by 5 m, so millimetre-scale changes were ~1e-4 in network units. The earlier finding "the skill is driven by rainfall and the depth history is not used" (test-time ablations of v1) was an
artefact of that weakness. The full ladder for every experiment / regime / fold is in the results tables below.

### 3. The paper's own protocol, and what persistence scores on it

The FloodCastBench paper (Xu et al. 2025; read from the open-access text via Europe PMC PMC11903882) forecasts **19 steps of 300 s over the whole domain from the true depth at t=1**
(FNO+ also gets the terrain and the rainfall series). Its metric named "RMSE" is a **relative (normalised) error, not metres**: every earlier statement "FNO+ RMSE 0.0786 m / 0.0248 m" was wrong
(dimensionless). `completion/f_paper_protocol.py` scores plain persistence (copy the frame at t=1) on 20-frame samples with the paper's metrics (whole event):

| region | persistence: rel. L2 / NSE / CSI(1 mm) / CSI(1 cm) | paper FNO+: "RMSE" / NSE / CSI(1 mm) / CSI(1 cm) |
|---|---|---|
| Mozambique 480 m | 0.0210 / 0.99951 / 0.983 / 0.973 | 0.0786 / 0.955 / 0.934 / 0.913 |
| UK 60 m | 0.0243 / 0.99935 / 0.980 / 0.965 | 0.0248 / 0.99895 / 0.860 / 0.963 |
| UK 30 m | 0.0247 / 0.99934 / 0.975 / 0.961 | 0.0252 / 0.99895 / 0.811 / 0.959 |

Which samples the paper used for its transfer tests is not stated; on the last 20% of each event persistence scores even better. Persistence is therefore as good as or better than the paper's best
model on most of the paper's own measures, and the paper reports no persistence baseline. **This is not a criticism of FNO+ as a surrogate** (it also forecasts from a dry initial state in other settings), but it
shows that the benchmark's metrics are dominated by "the water is mostly already there". Our task (5 observed frames -> 5 forecast frames, 30-minute steps, 128-px patches) differs from the paper's and
our numbers are **not comparable** with its tables.

### 4. Georeferencing and alignment checks

`completion/e_georef_coarse.py` overlays the simulated flood on OpenStreetMap (Esri imagery returned HTTP 403 from this machine that day): Mozambique 480 m (the simulated river follows the mapped
river to the coast; the estuary matches), Pakistan 480 m (two lakes, the desert and forest boundaries and the Indus channel line up) and the UK at 60 m (the River Eden meanders through Carlisle match
loop for loop) all fall on the right ground at kilometre / tens-of-metres precision (`completion/results/georef_*.png`). This confirms the coordinate frame (UTM zone, world file origin, recovered crop
offset) for those regions; Australia 30 m was checked earlier. **Negative result:** a physical water-surface probe (terrain + depth should be smoother than terrain where water is ponded,
`completion/c_wse_probe.py`) FAILED its own validation on Mozambique (it did not recover the known geometry) and is not used as evidence. The independent verifier could not confirm the crop offsets
independently either (DEM check flat; rain check peaked ~1 cell east); they are supported by the earlier terrain probe, the NoData-ring constraint and the map overlays, and the model results
do not depend on them at that precision (removing the rainfall altogether costs only 0.01-0.02 skill for the final models; note that the `rain_shifted_cells` ablation is partly a no-op because of
edge clipping and must not be cited for offset insensitivity).

### 5. The study that replaced the first results (protocol frozen before the final runs)

`completion/PREREGISTRATION.md` (md5 2b8b2ef69639, written 10:12 on 20 September, before any final run) fixes the recipe (Adam, 1e-3 peak with 200-step warm-up and cosine decay, gradient clip 1.0,
loss = MSE / persistence MSE, batch 1, 300 training windows, 30 epochs, last-epoch weights, no early stopping), the 3 spatial tile folds (each tile held out once), the three regimes, the three
experiments (E60 Australia -> UK at 60 m, E30 the same at 30 m, E480 Pakistan -> Mozambique at 480 m; depth only), the metrics and the hypotheses H1-H6. The design of the new models used ONLY held-out
tiles of the training region (Australia 60 m, fold 0): v2 (a ConvLSTM at 16x16 with temporal skips) +0.57, v2 + augmentation +0.65, v2lin (v2 + local path) +0.71, **local CNN (4,101 parameters, no
recurrence) +0.70, clstm (a full-resolution ConvLSTM, 4,741 parameters) +0.71**. The simplest recurrent model, clstm, is the project's CNN-LSTM; `local` is the no-recurrence control;
`v1` is the original model; `v2lin` the heavy comparator. Disclosure (code review F10): four heavy design runs also evaluated the never-seen UK and their log lines were visible; the two chosen
architectures had only blind design runs (no test-region evaluation), and no decision would have changed.

**Deviations from the frozen protocol, all after seeing results and all labelled exploratory or as trims:** (a) baselines re-run with the clipped-linear, per-pixel MLP (S4) and lookup-table (S5) rungs
after the first fold showed the per-pixel MLP was the strongest baseline; (b) kernel-size ablation (batches K) and an optimisation-robustness check (batch 8, lr 3e-3; OPT) added; (c) the ConvLSTM
variant with the future rain as side channels (clstmf, batch FX) added after the code review showed the forced layout handicaps the recurrent model; (d) **cells first trimmed to fit the compute budget and then reinstated** on 20 September evening when the instruction was "don't drop anything": the 30 m recurrent model in all regimes, the training-time ablations at 30 m, the severity study under the causal regime (closing code-review finding F8), the heavy comparator (653k parameters; on the two independent pairs, E60 and E480), the severity study with augmentation and the seed repeats. **Dropped after analysis as redundant, with reasons:** training-time ablations of the recurrent model (the trained ConvLSTM's test-time ablations and the local CNN's training-time ablations already answer the question), the 30 m ConvLSTM cells for the strict split, the side-channel variant, the causal/forced regimes and the optimisation check (30 m is the same Australia/UK storm pair as 60 m, not independent evidence), and severity under the forced regime (no use case for a known-future-rain severity label); (e) **post-hoc additions after seeing results, all exploratory and labelled:** a learning curve for the local CNN and the baselines (n = 25/50/100/300/600 windows), 90-epoch training of the local CNN and later of the ConvLSTM (added because the CNN gained from longer training and the recurrence comparison had to be fair), strict-split reruns of the plain CNN on Pakistan with four more seeds and the batch-8 recipe, reverse-direction transfers (`completion/reverse_transfer.py`), guard-disabled reruns of runs the dead-network guard aborted (batches CG and FXG), two retries of one side-channel fold, a `--no-guard` engine switch and a change of the guard's window sampling (see section 7).

### 6. Independent verification and code review (both by separate agents; reports in `completion/review/`)

**Independent re-computation from the raw files** (no project code): rain lag and slope (Australia 30 m, Mozambique 480 m), the persistence RMSE on the never-seen test windows (Mozambique n = 238,
17.0757 mm; UK 60 m n = 120, 15.7004 mm; agreement 1e-8), bit-exact equality of 100 random cached windows per pool with the raw frames, and the paper-protocol persistence numbers (agreement 1e-15): **all match**.
**Adversarial code review:** no critical finding (no target/future depth reaches an input, no tile or time-block overlap between training and held-out data, padded pixels are masked, metrics and bootstrap
statistics correct). Five major findings, all handled: the aggregation would have silently merged non-frozen runs into headline cells (fixed: frozen-recipe runs only, extras reported separately, complete
folds required); the start-time bootstrap treated strongly overlapping windows as independent (fixed: block bootstraps over time, two-way over tiles x time; intervals are now wider); the H3/H4 verdict rules
were coded looser than pre-registered (fixed); the forced-regime input layout handicaps the recurrent model (disclosed; variant clstmf added); the Grad-CAM and its deletion test measured different
quantities and never involved the severity head (redesigned: same target for map and deletion, positive-heat top set, two controls, plus a severity-class map). Minor findings disclosed: partial no-op of
`rain_shifted_cells`; hybrid-regime rule inputs contain partly-future rain and the recalibrated rule thresholds are fitted on true rates but applied to predicted ones; the UK-60 padding shifts the
patch-relative terrain channel by ~8 m (removing terrain changes skill by 0.000); provenance hashes are stamped at run end; each never-seen region is ONE flood event and E60 and E30 share the same
Australia and UK events, so the evidence amounts to **two independent train -> test pairs**, not nine.

**Second round of independent checks (21 September, after the extra experiments; reports `completion/review/verification_round2b.md` and `code_review_round2b.md`).** *The verifier* (raw files only, no project code): the stored per-window errors of all 284 run files recompute to the stored skills; for every window of the six test-role and three train-role pools (about 14,000) the cached depth frames equal a fresh read of the raw files and the stored persistence error equals an independent recomputation; the two-feature lookup-table baseline was re-implemented independently and reproduces five stored cells to about 1e-8; the separation of training and held-out windows (frames, tiles, 54-step embargo) was rebuilt for all 56 strict and reverse runs (minimum start gap 55 = one window length, no shared frame); a headline skill and its interval were reproduced. Its additional findings: the Pakistan failures (strict split, reverse transfer) sit in the windows at start index 0; the result files carry five different engine hashes; the reverse runs have no spatial hold-out; all skills are for wet windows only. *The code review* (adversarial, new code only): 0 critical, 3 major, 14 minor. Major: (F1) the learning curve confounded the number of windows with the number of optimiser steps - relabelled, equal-steps cells added (with the steps held equal, doubling the data changes the skill by -0.002 to +0.008: the plain CNN is limited by training, not data); (F2) guard-disabled runs were unmarked in the headline cells and could have hidden a cell if duplicated - dagger marks, a data-warning section, and the obsolete patch script removed; (F3) the queue runner could start every queued job at once if its process count failed - it now never returns 0 and counts its own jobs. Minor items fixed: the guard-explanation wording, an uncaveated `rain_shifted_cells` column (dropped), the mixed-aggregator label of the held-out columns, the seed-repeat held-out label, the guard's window sampling, file-handle leaks and the queue's active-log window. Noted, not changed: unmasked padding of the UK 60 m windows in the reverse runs (R60 baselines move by 0.02-0.05), the R480 ablation subsample, the `blocked: false` label of the reverse runs, the engine hash stamps (a same-seed re-run with the current code reproduces a stored run to 9e-6).

### 7. Results (generated by `completion/g_report.py` from the result files; skill = 1 - SSE_model / SSE_persistence)

Held-out tiles = tiles of the TRAINING region never used for training (3 folds, every tile held out once); never-seen region = a different flood. Intervals: 95% block bootstraps (time blocks; tiles x time blocks for the held-out score) - wider than the naive ones because neighbouring windows overlap and each region is ONE event. Frozen recipe, pre-registered (`completion/PREREGISTRATION.md`).

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

### 8. Reading the results

**8.1 Where the skill is.** Never-seen region, hybrid input, mean of 3 tile folds (0 = copying the last frame; intervals in the tables above). Australia -> UK at 60 m: straight-line extrapolation of the depth history +0.32, linear with rain/state/trend +0.34, **a two-feature lookup table (current depth x last one-step change) +0.79**, per-pixel MLP +0.85, local CNN +0.84 [+0.80, +0.89], **small ConvLSTM +0.87 [+0.81, +0.91]**. At 30 m: +0.34 / +0.36 / +0.76 / +0.85 / +0.83 / +0.84. Pakistan -> Mozambique at 480 m: +0.15 / +0.18 / +0.74 / +0.83 / +0.81 / +0.81. The original CNN-LSTM scored roughly +0.05 to +0.2 on the same tasks (see 8.9). The forecast is, to first order, **a nonlinear per-pixel function of the recent depth trajectory**: a table with a few hundred cells reaches about 90% of the best score; adding rain, terrain, neighbouring pixels or recurrence moves it by at most a few hundredths.

**8.2 Rainfall regimes (H5).** For the local CNN the ordering causal < hybrid < forced holds in all three experiments (UK 60 m +0.82 / +0.84 / +0.87; UK 30 m +0.81 / +0.83 / +0.85; Mozambique +0.79 / +0.81 / +0.83): the original input's partly-future rain was worth about +0.01 to +0.025, and known future rain about +0.02 to +0.03 more. The leak is real but small; it does not change any conclusion. For the ConvLSTM the ordering fails on Mozambique (causal +0.81, hybrid +0.81, forced +0.77) and is flat on the UK at 60 m (+0.85 / +0.87 / +0.85). The code review (F4) suspected the input layout (its own wording: "my hypothesis, unverified"): the forced rain is appended as five extra time steps after the observed history, which the recurrent state must carry; a variant that gives it the future rain as side channels is reported in 8.8.

**8.3 What the trained models use (test-time ablations, hybrid).** Normal / rain removed / terrain removed / depth history frozen / rain strictly past: ConvLSTM UK 60 m +0.87 / +0.86 / +0.87 / +0.02 / +0.86; local CNN UK 60 m +0.84 / +0.83 / +0.84 / -0.01 / +0.83; ConvLSTM Mozambique +0.81 / +0.79 / +0.81 / -0.01 / +0.79; local CNN Mozambique +0.81 / +0.79 / +0.81 / -0.01 / +0.78. Putting the depth history in the wrong order makes the skill strongly negative (-2.6 to -3.7). **The models use the depth history and almost nothing else** (H3 for the test-time version); rainfall is worth 0.01-0.02 on the never-seen region (0.02-0.07 on held-out tiles); the terrain nothing. (The `rain_shifted_cells` ablation is partly a no-op because of edge clipping and is not used.)

**8.4 Does spatial context matter?** Kernel 1 / 5 / 11 (a per-pixel network / 5x5 / 11x11), never-seen region: UK 60 m +0.85 / +0.84 / +0.84; UK 30 m +0.84 / +0.83 / +0.83; Mozambique +0.82 / +0.81 / +0.79 — **no benefit from looking at neighbouring pixels**; a per-pixel MLP (+0.85, +0.85, +0.83) does as well as any CNN. On the held-out Pakistan tiles the plain CNNs are unstable with the frozen batch-1 recipe (pooled skill +0.24 for kernel 1, +0.43 for kernel 5, -0.66 for kernel 11, with intervals that include 0), while the ConvLSTM (+0.84) and the per-pixel MLP (+0.87) are steady; whether that is a training-recipe effect is examined in 8.7.

**8.5 Does recurrence matter (H4, pre-registered rule: equivalent if |difference| < 0.02 and the interval lies inside +-0.03)?** ConvLSTM minus local CNN, never-seen region, paired block bootstrap: UK 60 m hybrid **+0.025 [+0.004, +0.043]** (ConvLSTM larger; causal +0.031), Mozambique hybrid +0.004 [-0.011, +0.019] (equivalent); Mozambique forced -0.069 (local larger; layout effect, see 8.2); UK 30 m hybrid +0.009 [-0.014, +0.031] and Mozambique causal +0.016 [-0.004, +0.034] favour the ConvLSTM by the rule's "larger wins" clause but their intervals include 0 (the report marks them "clstm larger (by the rule, but the interval includes 0)"). Per-pixel MLP versus ConvLSTM: the ConvLSTM is better only on the UK at 60 m (+0.02). **At the frozen 30-epoch recipe the ConvLSTM is ahead on one of the two independent pairs and equivalent on the other. But the plain CNN is slightly under-trained at 30 epochs (8.7). With BOTH models trained for 90 epochs (a fairness check added after seeing the CNN's 90-epoch gain) the ConvLSTM-minus-local difference is +0.004 [-0.021, +0.027] on the UK at 60 m and -0.014 [-0.029, +0.006] on Mozambique: equivalent by the pre-registered rule in both independent pairs. The recurrent CNN-LSTM is therefore NOT shown to beat a recurrence-free CNN once training length is equal; its skill is equivalent within the pre-registered margin (difference inside +-0.03).**

**8.6 Training-time ablations (H3, pre-registered rule: skill without depth history lower by more than the fold-to-fold sd).** Local CNN trained without one ingredient, never-seen region, hybrid (UK 60 m / Mozambique; full model +0.841 / +0.807): without the depth history **+0.172 / +0.086**; without depth AND rain history +0.159 / +0.085; without the rain history only +0.839 / +0.802; without any rain +0.835 / +0.798; without terrain +0.842 / +0.812. **H3 holds by a wide margin:** the depth history is worth +0.669 and +0.721 of skill (fold sd of the full model 0.003 and 0.008); rain is worth about 0.005-0.009 and terrain nothing. This is the same answer as the test-time ablations (8.3) and as the linear ladder (a linear model with rain and state but no depth trend reaches only +0.21 / +0.19 / +0.10 on the never-seen region for UK 60 m / UK 30 m / Mozambique).

**8.7 Strict space+time hold-out (H6), optimisation, training length: what limits the same-region hold-out score.** With a hidden time block as well as hidden tiles, the same-region held-out score falls: E60 local +0.756 -> +0.716 (within the pre-registered 0.05), E60 ConvLSTM +0.782 -> +0.645, E30 local +0.730 -> +0.686 (within 0.05), E480 local +0.430 -> **-6.56**, E480 ConvLSTM +0.845 -> +0.742. H6 passes in 2 of 5 cells. (The strict split for the ConvLSTM at 30 m was not run: 30 m is the same storm pair as 60 m.) The failure sits in the windows at the very START of the flood (start index 0, absent from training in the fold whose time block contains it): 6 of the 480 strict held-out windows carry 98% of the plain CNN's squared error (in that fold the six start-0 windows carry a squared error of 33,901 against 1,262 for persistence, out of 34,294 against 3,769 for the whole fold; the single worst window alone is 14,853 against 780), and without them its strict score is +0.842 (ConvLSTM +0.866); in the tile-only hold-out 12 of 900 windows carry 82% of its error (+0.865 without them). Never-seen-region skill is unaffected by the strict split (E480 local +0.807 -> +0.803; E60 +0.841 -> +0.837). A further check separates "the data lack the onset" from "one training run went wrong": the plain CNN retrained on the Pakistan fold-0 strict split with four more seeds fails in the same way (held-out skill -6.9, -9.1, -8.1, -8.9; the frozen seed -8.1) and so does the batch-8 / lr 3e-3 recipe (-5.7), while the never-seen skill stays +0.76 to +0.79 in all of them. **The failure of the plain CNN reproduces across five seeds and a second recipe on this fold (-5.7 to -9.1), so it is not one unlucky run; that the cause is the missing onset is an inference: the ConvLSTM on the same fold scores +0.722, the lookup table +0.509, and the plain CNN's other two folds +0.848 and +0.738.** **Optimisation (batch 8, learning rate 3e-3, a deviation from the frozen recipe):** never-seen skill is 0.008-0.026 *lower* than under the frozen recipe (E60 local +0.829 vs +0.841, E60 ConvLSTM +0.853 vs +0.866, E30 local +0.822 vs +0.830, E480 local +0.798 vs +0.807, E480 ConvLSTM +0.785 vs +0.811), so a larger batch and a higher learning rate do not improve the never-seen score (the frozen recipe is not limited by its step size), but the frozen 30-epoch recipe does under-train the plain CNN in number of steps (see Training length); but the plain CNN's held-out Pakistan-tile score goes from +0.430 to **+0.771**, which suggests (one seed) that its tile-level instability on Pakistan is optimisation noise, not missing data ; the ConvLSTM on Mozambique is also slightly worse under the batch-8 recipe (+0.785 vs +0.811; held-out tiles +0.837 vs +0.845). **Training length:** 90 instead of 30 epochs raises the local CNN's never-seen skill by +0.021 (E60, +0.862), +0.008 (E30, +0.838) and +0.024 (E480, +0.831) and its Pakistan held-out score from +0.430 to +0.699 pooled (+0.383 to +0.678 as the mean of the three fold skills): the frozen 30-epoch recipe under-trains the plain CNN slightly ; the ConvLSTM gains almost nothing from 90 epochs (UK 60 m +0.866 -> +0.866, Mozambique +0.811 -> +0.818; Mozambique held-out tiles +0.843 -> +0.861), so at equal length the two are level (8.5). **Reading:** the plain CNN's score on hidden Pakistan tiles is unstable across seeds (sd 0.33 over five seeds) while the ConvLSTM (+0.83 to +0.86) and the per-pixel MLP (+0.87) are steady; a model that has never seen a flood START cannot be trusted on one (under the strict split the lookup table stays at +0.66 and the ConvLSTM at +0.742; the per-pixel MLP (-2.65, mean of folds) and the plain CNN (-6.56) do not).

**8.8 Future rain as side channels (does the input layout explain the ConvLSTM's forced-rain deficit? code review F4).** Giving the ConvLSTM the five future rain maps as extra channels on its last hidden map instead of as five extra time steps lifts its forced-regime never-seen skill on Pakistan -> Mozambique from +0.765 (folds +0.763 / +0.780 / +0.752) to **+0.822** (+0.822 / +0.827 / +0.816) and on Australia -> UK at 60 m from +0.854 (+0.860 / +0.855 / +0.847) to **+0.880** (+0.876 / +0.885 / +0.880). The layout explanation is therefore supported: with the future rain side-loaded the ConvLSTM has the highest forced-regime skill on the UK (+0.880, against the per-pixel MLP +0.872 and the local CNN +0.866; differences of this size are within seed noise) and close to the local CNN on Mozambique (+0.834). Removing the known future rain at test time costs the side-channel model 0.06-0.08 (E60 +0.880 -> +0.803, E480 +0.822 -> +0.764) and the local CNN 0.06-0.13, so models trained with future rain do use it. Fold 1 of the UK experiment was aborted twice by the dead-network guard (see 8.15); finished with the guard disabled (same seed) it scored +0.885, and the two post-hoc retries (seed +1: aborted again; seed +2 with learning rate 5e-4: +0.865) are listed separately in the report. Not run at 30 m (same storm pair as 60 m).

**8.9 The original model (v1) under the same folds and recipe (hybrid input).** Never-seen skill per fold: UK 60 m +0.184 / +0.195 / +0.181 (mean +0.187), UK 30 m +0.181 / +0.193 / +0.189 (+0.188), Mozambique +0.060 / +0.063 / +0.062 (+0.061); held-out tiles +0.209 / +0.259 / +0.178, +0.174 / +0.220 / +0.184, +0.099 / +0.167 / +0.056. The final local CNN and ConvLSTM score +0.841 / +0.866, +0.830 / +0.840, +0.807 / +0.811 on the same tasks, i.e. **about 0.65 (UK) and 0.75 (Mozambique) higher**. The v1 figures agree, by a different code path (the study engine instead of the original notebooks), with the first study's typical UK skill of about +0.19 (its seven runs spanned +0.04 to +0.22, and identical configurations moved by up to 0.03 between machines). Test-time ablations of v1: removing the rain takes it to +0.003 / +0.003 / +0.002; freezing or scrambling the depth history changes nothing (+0.187 -> +0.187). So the earlier statement "the skill is rainfall-driven and the depth history is not used" is **correct for v1** and was wrongly generalised to "the CNN-LSTM": v1 cannot read the depth history (one 128-number bottleneck per frame, depth scaled by 5 m). The UK v1 runs were aborted by the dead-network guard at epoch 2-5 (a false alarm, 8.15) and finished with the guard disabled; the Mozambique runs ran with the guard on.

**8.10 How steady are the numbers? Seed repeats of the local CNN (five seeds per fold, hybrid).** The never-seen skill varies by only 0.006-0.010 between seeds within a fold (UK 60 m 0.007, UK 30 m 0.010, Mozambique 0.006) and by 0.001-0.008 between folds after averaging the seeds. **Differences smaller than about 0.02 in never-seen skill are therefore not resolvable**; the ConvLSTM's +0.025 edge over the local CNN on the UK at 60 m (30-epoch recipe) is only about 3 seed standard deviations and it disappears when the plain CNN is trained as long as the ConvLSTM (90 epochs: +0.004 [-0.021, +0.027]). The held-out-tile score is much noisier (sd 0.015 at 60 m, 0.027 at 30 m, **0.33 on Pakistan**: single seeds range from -0.37 to +0.77).

**8.11 Learning curve (does the number of training windows limit the models?).** Never-seen skill against 25 / 50 / 100 / 300 / 600 training windows (hybrid, mean of 3 folds). Local CNN: UK 60 m +0.760 / +0.794 / +0.823 / +0.841 / +0.861; UK 30 m +0.666 / +0.762 / +0.787 / +0.830 / +0.850; Mozambique +0.643 / +0.760 / +0.766 / +0.807 / +0.833 (still rising at 600). Per-pixel MLP: +0.829 / +0.857 / +0.859 / +0.846 / +0.868; +0.803 / +0.836 / +0.837 / +0.854 / +0.836; +0.809 / +0.807 / +0.831 / +0.829 / +0.844 (within about 0.04 of its 600-window score already with 25-50 windows). Lookup table: +0.710 -> +0.798, +0.623 -> +0.764, +0.489 -> +0.758. The straight-line trend does not improve (+0.365 -> +0.327 at 60 m). At 600 windows the plain CNN (+0.861 / +0.850 / +0.833) is level with the MLP (+0.868 / +0.836 / +0.844): **neighbouring pixels still add nothing**, but note that with the epochs fixed at 30 the number of optimiser steps grows with the number of windows, so this curve mixes 'more data' with 'more training' (separated by the equal-steps cells: with the number of optimiser steps held equal, doubling the data changes the never-seen skill by only -0.002 to +0.008 (UK 60 m +0.841 -> +0.847 at 9,000 steps and +0.863 -> +0.861 at 18,000; UK 30 m +0.830 -> +0.838 and +0.846 -> +0.850; Mozambique +0.807 -> +0.811 and +0.835 -> +0.833), i.e. inside seed noise: between 300 and 600 windows the plain CNN is limited by the amount of TRAINING, not by the number of windows (the equal-steps test was not made at 25-100 windows)); the per-pixel structure is learnable from a few dozen windows.

**8.12 Reverse-direction transfers (exploratory; train on the small region, test on the big one).** The small regions have no room for tile folds (Mozambique 1 valid 128-px tile, UK 30 m 2, UK 60 m none), so folds are hidden thirds of the START TIMES of the small region's corner tiles (four for Mozambique and for UK 30 m, two padded 85 x 85 squares for UK 60 m) (protocol in `completion/reverse_transfer.py`); the numbers are not comparable cell by cell with the forward experiments. Never-seen skill (hybrid; mean of 3 time folds): **UK -> Australia at 60 m** local CNN +0.738 [+0.571, +0.797], per-pixel MLP +0.659, lookup table +0.601; **at 30 m** local CNN +0.662 [+0.420, +0.746], MLP +0.588, table +0.606; both keep the ordering causal < hybrid < forced (60 m +0.702 / +0.738 / +0.772; 30 m +0.637 / +0.662 / +0.718). **Mozambique -> Pakistan at 480 m fails for the plain networks** (local CNN -0.884 [-2.665, +0.853], MLP -0.312), but only on the first windows of the Pakistan simulation: 54 of 3,240 window scores (start index 0) carry 93% of the local CNN's squared error, and without them it scores +0.834. In those windows the domain already holds 8-12 m of water that recedes to 2-7 m, a pattern absent from the small Mozambique flood, and the model's squared error in the ten worst windows of fold 0 is 4-11 times persistence's (over all 54 start-0 windows the ratio has median 7.7 and ranges from 0.3 to 27). The lookup table (+0.549, bounded bins) and the ConvLSTM (+0.444 [+0.277, +0.649]; +0.550 without the start windows, i.e. lower than the plain CNN on ordinary windows) are less exposed. On the training region's own hidden time thirds the local CNN scores +0.80 to +0.85 (pooled), the per-pixel MLP +0.83 to +0.88 and the ConvLSTM +0.76 to +0.85 (pooled); the lookup table (+0.64 to +0.74) and the linear baselines (negative to +0.05) are lower. Two caveats: the UK 60 m windows are padded squares (85 px of data in 128 px) and neither the training loss nor the baseline fits mask the padding, which moves the R60 baselines by 0.02-0.05 (the never-seen scoring itself is masked); and the ablations of the reverse runs use a 400-window subsample and are not tabulated. In this direction the ConvLSTM (hybrid) scores +0.695 [+0.499, +0.757] at 60 m and +0.584 [+0.399, +0.661] at 30 m, below the local CNN (by 0.04 at 60 m and 0.08 at 30 m, with overlapping intervals).

**8.13 Does capacity matter? The heavy comparator (v2lin: a ConvLSTM with a local path, 652,918 parameters, about 150 times the small models).** Same recipe, hybrid, 3 folds: never-seen skill UK 60 m **+0.840** [+0.790, +0.893] (small ConvLSTM +0.866, local CNN +0.841) and Mozambique **+0.802** [+0.758, +0.865] (+0.811 / +0.807); held-out tiles +0.770 and +0.582 [+0.242, +0.903] (the same Pakistan-tile instability as the plain CNN). Its test-time ablations match the small models' (depth history frozen +0.006 / -0.015, scrambled strongly negative). **A model 150 times larger does not do better: capacity is not what limits the skill.** (Run for UK 60 m and Mozambique only; 30 m is the same storm pair as 60 m.)

**8.14 Severity: augmentation, and the causal-rain regime.** The severity numbers of the first severity paragraphs (8b) were obtained with the original training (400 class-balanced windows per fold, drawn from the 41-42 tiles that are not held out, no augmentation). With the eight dihedral augmentations (identical for input and label) the capsule head's natural-mix accuracy rises from 0.678 to **0.803** and the dense head's from 0.602 to **0.798** (balanced accuracy 0.593 -> 0.697 and 0.479 -> 0.660). The capsule-minus-dense gap of +0.076 [+0.051, +0.101] shrinks to **+0.005 [-0.014, +0.022]**: the capsule design's earlier advantage disappears with augmentation, which is consistent with a small-sample regularisation effect (not tested directly) and gives no evidence that capsules suit the task. Both remain far below the trivial rules (last-frame threshold 0.974, forecast threshold 0.986; on the 47 class-changing windows the capsule is right in 22, the dense head in 27, the forecast threshold applied to their forecasts in 31 and 30). The rule layer still lowers accuracy after augmentation (capsule 0.803 -> 0.741, 8 fixed / 119 broken; dense 0.798 -> 0.751, 12 fixed / 96 broken; recalibrated thresholds 0.745 / 0.754); its stated depth-trend reasons stay true about 82-84% of the time. Grad-CAM for the depth forecast passes the deletion test in 84-92% of windows for all four models; the severity map remains weak for the capsule head (top beats both controls in 55-56%) and moderate for the dense head (73-75%). **Causal-rain regime (no look-ahead rain in the inputs; addresses the rain part of code-review finding F8 (the recalibration is still fitted on true rates and applied to predicted ones)):** without augmentation the capsule head scores 0.623 and the dense head 0.568 (hybrid: 0.678 / 0.602); the capsule-minus-dense gap is +0.056 [+0.029, +0.082]; the trivial thresholds are unchanged (0.974 last frame, 0.986-0.988 forecast); the rules again lower accuracy (capsule 0.623 -> 0.575, recalibrated 0.588; dense 0.568 -> 0.547 / 0.556) and their depth-trend reasons are true 84-85% of the time; Grad-CAM for the depth forecast beats both control masks in 93-97% of windows and the severity map in 57-77%. None of the severity conclusions depended on the partly-future rain in the hybrid input. (Severity under the forced regime was not run: there is no use case for a known-future-rain severity label.)

**8.15 The dead-network guard raised false alarms (a mistake found and fixed on 21 September).** The engine aborts a run when the predicted change map is identical across validation windows (per-pixel spread below 1e-5 m). It fired eight times: six original-model runs (UK 60 m and 30 m, epoch 2-5) and two side-channel runs (UK 60 m fold 1: the frozen seed at epoch 13 and a retry with the next seed at epoch 6). Seven of the eight were verified as **false alarms** by finishing them with the guard disabled and the same seed: the six original-model runs trained normally (+0.18 to +0.195) and the frozen-seed side-channel run scored +0.885; the eighth (the retry) was not re-run. Cause: the check used the 12 EARLIEST held-out windows (code review finding F13), in which the flood has barely started, so even a healthy network predicts nearly no change; slow-starting models (the original) look "dead" for the first epochs. Fix (21 September, 00:00; refined at about 05:50 to evenly spaced indices and no abort when fewer than two check windows exist): the check now samples 12 windows spread over the whole held-out period; it only changes the decision to abort (argued from the code; a bit-for-bit re-training comparison was not run), so runs that finished under the old check are unaffected by the change; whether they would also pass the new check was not tested (the 12 windows are taken at a stride and cover 84-92% of the held-out set). I first described the aborts as "collapses" in messages during the day; that was wrong and is corrected here and in the report table "Runs aborted by the dead-network guard".

### 8b. Severity, symbolic rules and Grad-CAM (hybrid regime; Australia 30 m; 3 folds pooled; details in section 9)

**Capsule vs its control (the design hypothesis).** On the natural class mix (1,800 held-out windows) the capsule head is right 67.8% of the time and a plain dense head 60.2%: **capsule - dense = +0.076, 95% CI [+0.051, +0.101]**. The capsule design therefore does beat its own control on this task, by a modest margin. **Both lose badly to the trivial rule:** thresholding the last observed frame scores 97.4% and thresholding the model's own forecast 98.7%, because the label (maximum depth over the forecast in the patch) is almost fixed by the last frame. On the 47 windows where the class changes, the capsule is right in 22 (47%), the dense head in 24, the forecast-threshold in 34 (72%); the last-frame rule is wrong there by construction. The earlier statement "no evidence the head anticipates escalation" stands: it does not do better than thresholding the forecast.

**The symbolic rule layer now lowers accuracy.** With the original thresholds the rules change 161 of the capsule's answers: 25 fixed, 109 broken, accuracy 67.8% -> 63.1%; with thresholds re-derived on each fold's training windows 64.0% (51 fixed, 119 broken); for the dense head 60.2% -> 56.5% / 58.4%. (With the original weak model the net effect on accuracy was between -0.0 and +0.4 points in two full evaluations: 4 fixed / 4 broken and 2 fixed / 1 broken.) The rules' stated depth-trend reasons are true in about 84% of windows (the predicted trend has the same sign as the actual one; 'rising' claims 84%, 'receding' 91%). The reasons were not reviewed by a flood expert; the rule inputs use rainfall that is partly future in the hybrid regime and the recalibration is fitted on true rates but applied to predicted ones (code review F8).

**Grad-CAM, redesigned test.** For the depth forecast, deleting the top-10% heat pixels changes the predicted change maps more than **both** a wet/dry-matched random mask and the same mask moved elsewhere in 88% (capsule model) and 92% (dense model) of the windows in which the deletion had any effect (30-40 of the 40 fixed windows per fold; mean over the 3 folds); wet cells are hotter than dry ones in 79% / 76% of windows (median ratio 3.7 / 3.8). The heat maps of two models trained with different seeds agree moderately (median rank correlation 0.52 / 0.54): the hot spots are partly stable, partly not. For the severity class the map is weaker and less stable (top beats both controls in 55% of windows for the capsule head, 73% for the dense head; seed-to-seed rank correlation 0.17 for the capsule, 0.76 for the dense head). Grad-CAM for the depth forecast is a usable explanation; for the capsule's severity class it is not established.

### 9. Severity, symbolic rules and Grad-CAM on the new encoder (`completion/severity_study.py`; Australia 30 m; 3 tile folds)

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

### 10. Mistakes made in round 2 (kept, because they are the useful part)

- Told you "no future information enters any input" after checking only the depth frames; rainfall was never checked against the simulation's own timing (wrong: up to 30 minutes of future rain in the last frame).
- Compared the original CNN-LSTM only with persistence, never with a trend baseline, so its inability to use the depth history went unnoticed for weeks; then explained the resulting behaviour ("rainfall drives the skill", "history not used") as a property of floods.
- Wrote the paper's relative "RMSE" as metres in the log, explainer, deck and notebooks.
- Two of my own new diagnostics were wrong and were caught by smoke tests before any result used them: a dead-network guard that measured the spread of window-mean predicted change (sparse changes cancel in the mean: false alarm) and a Grad-CAM deletion effect that counted the removed water itself. A third, `rain_shifted_cells`, was partly a no-op because of edge clipping (caught by the code review; not to be cited).
- Two operational slips: a shell command that killed its own process by matching its own command line (twice), and a half-finished edit saved to a file the running queue re-imports (no job was affected; a regression check reproduced an earlier result exactly).
- Read the learning curve as "the plain CNN is data-hungry"; the curve had fixed 30 epochs, so more windows also meant more optimiser steps, and the equal-steps runs showed the gain comes from training, not data (found by the code review, then tested).
- **The dead-network guard raised false alarms, and I called the aborted runs "collapses".** Even after the earlier fix the guard looked at the 12 EARLIEST held-out windows, where a flood has barely started and every healthy network predicts almost no change. It aborted six original-model runs and two side-channel runs; seven of the eight were re-run with the guard disabled and trained normally (the original model +0.18 to +0.195, the side-channel run +0.885). I told the user the original model had "collapsed" and that a side-channel fold had needed a third attempt; both statements were wrong and were corrected the same night. The guard now samples 12 windows across the whole held-out period.
- Read a per-epoch validation printout (48 early windows) as evidence that the batch-8 / lr 3e-3 recipe was "worse"; the final scores were about equal (never-seen -0.008 to -0.013). Per-epoch printouts are not evidence.
- **The frozen 30-epoch recipe under-trained the plain CNN slightly**, which inflated the ConvLSTM's apparent +0.025 edge on the UK at 60 m; it vanished (+0.004 [-0.021, +0.027]) when both were trained for 90 epochs. Found only because the extra checks (seed repeats, longer training) were added after the main matrix; the pre-registered comparison (H4) is reported as pre-registered, and the equal-length comparison next to it.
- Operational: several queue runners started at once and briefly ran five training jobs on a 15.5 GB laptop (the limit was four); memory ran out, the laptop hung and was restarted. Every finished result file was intact (all read back correctly); only the jobs in flight were lost and re-run. A single runner with a hard cap of four jobs was used from then on. Completion-time estimates given during the day were too optimistic because they ignored re-runs and the reinstated cells.
- Earlier in the same day: dropping two regions and calling the project complete while half of the dataset was unused (previous section).
<!-- ROUND2 END -->

## What is and is not established

### Established (measured; code, logs and result files are in this handoff; all numbers are in the round-2 section above)

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

### Not established, or not supported

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

## Local setup notes

- Python 3.11 venv, TensorFlow 2.21 (CPU-only: native Windows TF >= 2.11 has no GPU support).
- `hf` replaced `huggingface-cli` (`hf auth login`).
- `snapshot_download`'s `allow_patterns` is fnmatch, so `*` also matches across `/`.
- **The notebook's Hugging Face pull was tested against the live repo** (UK subset): the patterns matched
  1,019 files = 865 depth + 145 rainfall + 2 DEM + 2 land-use + 5 georeferencing, exactly as expected; the
  downloaded layout loads in `RegionData` with the UK offset applied. It took ~315 s locally. Australia is
  ~15-16GB (2,881 depth frames x 4.6MB), so Colab's first cell will be the slow one.
- On **native Windows**, `snapshot_download`'s default cache mode needs symlinks (Developer Mode or admin) and
  the repo's deep folder names can hit the 260-character path limit. Use `local_dir="C:\\somewhere\\short"`
  when pulling locally. Neither applies on Colab (Linux).
