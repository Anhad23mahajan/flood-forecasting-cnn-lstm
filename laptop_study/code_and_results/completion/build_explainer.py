"""Builds Project_Explainer_End_to_End_v2.md (the deliverable in the DL Project folder) from: the unchanged sections of the previous explainer, the rewritten sections below,
and the generated result tables (completion/results/REPORT.md, SEVERITY_REPORT.md).  Refuses to publish (--final) while any [PENDING ...] marker is left.
Usage: python completion/build_explainer.py [--final]      (without --final it writes a draft next to the results and never touches the deliverable)"""
import os
import re
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DL = r"C:\Users\user\OneDrive - NBCC (India) Limited\Desktop\Sem 5\Deep Learning\Lab\DL Project\Round2_Corrected_Study_21Sep2026\1_Presentation_and_Explainers"   # the new deliverables folder (the original documents stay one level up, untouched)
OLD = os.path.join(DL, "Project_Explainer_End_to_End_v2.md")
BACKUP = "completion/results/explainer_previous_version.md"
FINAL = "--final" in sys.argv
src = open(BACKUP if os.path.exists(BACKUP) else OLD, encoding="utf-8").read()
if not os.path.exists(BACKUP):
    open(BACKUP, "w", encoding="utf-8").write(src)          # keep the previous version (the base for the unchanged sections)


def section(text, num):
    """Text of '## <num>. ...' up to the next '## '."""
    m = re.search(rf"^## {num}\. .*?(?=^## |\Z)", text, re.S | re.M)
    assert m, num
    return m.group(0).rstrip() + "\n"


def read(p):
    return open(p, encoding="utf-8").read() if os.path.exists(p) else "[PENDING: %s not generated yet]" % p


REPORT = read("completion/results/REPORT.md")
SEV = read("completion/results/SEVERITY_REPORT.md")
report_body = REPORT.split("\n", 1)[1] if REPORT.startswith("#") else REPORT
sev_body = SEV.split("\n", 1)[1] if SEV.startswith("#") else SEV
report_body = re.sub(r"^## ", "#### ", report_body, flags=re.M)      # keep the section numbering of the explainer intact
sev_body = re.sub(r"^## ", "#### ", sev_body, flags=re.M)

HEADER = """# CNN-LSTM Based Spatiotemporal Flood Evolution Forecasting

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
"""

DATASET_PATCH = [
    ("- **Rainfall ↔ depth alignment is an inference.** Nothing in the archive says how they line up. For Australia, rainfall's 481 files × 30 min span exactly the 10 days depth spans, and the paper states the simulation period as 20 February to 2 March 2022. The *hour* is not independently confirmed.",
     "- **Rainfall ↔ depth alignment (confirmed physically).** Rainfall file *j* is the rain that **falls during** `[1800 j, 1800 (j+1))` s of the simulation: the domain-mean depth gain over each 30-minute interval correlates 1.00 with file *j* (lag 0) in all four regions, with slope 0.494–0.502 (the files are mm/h intensities, so 0.5 h × the value), and the files' dates and durations match the paper's events. An independent re-computation from the raw files reproduced this. The original loader used the file of the interval that *contains* each frame's time — so the last observed frame carried up to 30 minutes of future rain (the *hybrid* regime)."),
    ("- **Georeferencing:** the depth TIFFs carry no CRS. The archive's `Spatial_References.txt` (UTM zone per region, e.g. Australia = UTM 56S) and one `.tfw` world file per region supply it. **The world files describe the 30 m grid for all four regions, so they are valid only for Australia and the UK.**",
     "- **Georeferencing:** the depth TIFFs carry no CRS. The archive's `Spatial_References.txt` (UTM zone per region, e.g. Australia = UTM 56S) and one `.tfw` world file per region supply it. The world files describe the 30 m grid; for the coarse grids the recovered scale and offset are applied on top. **Checked against real maps** (OpenStreetMap; Esri imagery returned HTTP 403 that day): Mozambique (simulated river, estuary and coast), Pakistan (two lakes, the desert and forest edges, the Indus channel) and the UK at 60 m (the River Eden meanders through Carlisle) all fall on the right ground at kilometre / tens-of-metres precision (`completion/results/georef_*.png`)."),
]
DATASET_EXTRA = """
- **Negative result and a caveat on the alignment.** A physical water-surface probe (terrain + depth should be smooth where water is ponded) **failed its own validation** on Mozambique and is not used as evidence. The independent verifier could not confirm the crop offsets by its own probes either (inconclusive). They rest on the earlier terrain probe, the NoData-ring constraint and the map overlays; the final models are insensitive to them at that precision (removing the rainfall altogether costs only 0.01–0.02 skill).
"""

SCOPE = """## 5. The Research Gap, and What Is Explicitly Out of Scope

**Gaps we address:**
- No CNN-LSTM/ConvLSTM baseline is reported for FloodCastBench (only U-Net and FNO/FNO+); we add one. **More importantly, the paper reports no trivial baseline; we add a full baseline ladder** (copy the last frame, linear extrapolation, a two-feature lookup table, a per-pixel MLP) and find that on this task most of the skill sits in the simplest rungs (Section 10).
- To our knowledge FloodCastBench's cross-regional transfer has not been tried with CNN-LSTM; we run the paper's three same-resolution settings.

**Scope limits (deliberate boundaries):**
- No new hydrodynamic simulations; we rely on FloodCastBench's simulated sequences.
- **Patch-based** (128×128) forecasts, 5 observed frames → 5 forecast frames at 30-minute steps (2.5 hours ahead). No whole-scene training, no claim of operational early-warning accuracy.
- **All four regions and three resolutions are used, never mixed in one model.** Severity is scored only for Australia at 30 m (the label is degenerate elsewhere). Our task differs from the paper's (19 steps at 300 s from the initial state over the whole domain), so our numbers are **not comparable** to its tables.
- Each never-seen region is **one flood event**, and E60/E30 share the same Australia and UK events: the evidence amounts to **two independent train → test pairs**.
- No new Indian dataset. Capsules have no precedent for dense regression, so they are scoped to severity classification; the motivation (routing-by-agreement may preserve spatial relationships) is a **hypothesis we tested, not an assumption** (Section 10).
"""

ARCH = """## 6. Architecture (as built)

**The final model — a small full-resolution CNN-LSTM (4,741 parameters).** Input: 5 observed frames (30-minute steps, 128×128) of depth, terrain and rainfall, plus the depth change since the previous frame (in centimetres).
A **ConvLSTM2D** (8 channels, 3×3) runs over the frames at the *native resolution* — the temporal state is a map, not a vector, so per-pixel change stays visible. A 3×3 convolution and a zero-initialised 1×1 convolution turn its final hidden map
into 5 change maps; **forecast = last observed depth + change** (persistence-anchored: an untrained model equals "copy the last frame" exactly). See `completion/results/architecture_v3.png` (and slide 7).

**Controls and comparators (same recipe, same folds).** `local`: a 4,101-parameter CNN with no recurrence (the temporal information enters only through the frame-difference channels); `local1` / `local11`: the same with 1×1 or 11×11 kernels (kernel 1 = a per-pixel network, no spatial context);
`v1`: the original CNN-LSTM (each frame flattened through one 128-number dense layer; kept as the record); heavier ConvLSTM variants for one fold. The **baseline ladder** (no deep learning): S1 linear on depth history, S3 linear with rain and terrain, **S5 a lookup table of the mean future change per bin of (current depth × last one-step change)**, S4 a per-pixel MLP.

**Why the original failed.** It flattened the 16×16×64 CNN features of each frame into a single 128-number dense layer before the LSTM and scaled depth by 5 m, so millimetre-scale frame-to-frame changes were ~10⁻⁴ in network units and no per-pixel temporal difference could survive the bottleneck; its decoder saw only the last frame's features. A per-pixel linear model of the depth history therefore beat it by ~2×.

**Severity branch and explanations.** The joint model shares the ConvLSTM encoder: the depth head above and a severity head (three stride-2 convolutions to 16×16×64, then either the **capsule** head — PrimaryCaps → routing-by-agreement → class capsules, margin loss — or a plain **dense** head as the control). The severity label is derived from the maximum depth in the patch over the forecast horizon (none < 0.3 m, minor < 1 m, moderate < 2 m, severe > 2 m); it is only informative for Australia at 30 m.
**Grad-CAM** explains the depth forecast (and, separately, the severity head's predicted class) at the shared encoder's hidden map; the **symbolic rule layer** is post hoc — deterministic rules over predicted depth rate, rainfall, slope and drainage proximity adjust the severity class by ±1 and attach a reason; **neither changes the forward pass.**
"""

PIPELINE = """## 7. End-to-End Pipeline

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
"""

VIZ = """## 9. Visualization Pipeline

`GeoTIFF (predicted depth) → rasterio (CRS + transform) → folium (overlay on satellite imagery)`

The per-region UTM zone and pixel transform come from FloodCastBench's own world files and spatial-reference table (which describe the 30 m grid; for the coarse grids the recovered scale and offset are applied on top). The transform includes the half-pixel shift for the pixel-centre origin.
Verified against real imagery/maps: Australia 30 m (Esri imagery), and — in the second audit — Mozambique and Pakistan at 480 m and the UK at 60 m against OpenStreetMap (rivers, lakes, coast, meanders line up). The interactive map itself (folium) was built and executed for Australia.
"""

WHERE = """## 13. Where Everything Is

- **This folder:** the updated deck (`..._v2.pptx`), this explainer, and `architecture_diagram/` (the diagram of the *original* model, kept as the record; the final architecture figure is in the code handoff at `completion/results/architecture_v3.png`). The original deck and explainer are unchanged.
- **Code handoff** (`flood_forecasting_handoff.zip`, and `C:\\Dev\\flood-forecast-demo` on Anhad's laptop): `PROJECT_LOG.md` (full record with corrections), `HANDOFF_README.md`, **`flood_forecasting_study_colab.ipynb` (the current study: run on Colab, Runtime → GPU → Run all)**, `flood_forecasting_colab.ipynb` and `flood_forecasting_multires_colab.ipynb` (the ORIGINAL model, marked superseded), the study engine and analysis scripts in `completion/`, the pre-registration `completion/PREREGISTRATION.md`, the two independent reviews in `completion/review/`, and every result file and log.
"""

RESULTS_INTRO = """## 10. Evaluation and Results

**How to read the numbers.** Skill = 1 − (model squared error ÷ persistence squared error): **0 = no better than copying the last observed frame, 1 = perfect, negative = worse.** "Held-out tiles" are map squares of the *training* region never used for training (3 spatial folds; every tile is held out once); "never-seen region" is a different flood in a different place (the UK, Mozambique) — the stronger test. Intervals are 95% **block bootstraps** over time blocks (and tiles): neighbouring windows overlap in time and each region is one event, so they are wider than a naive bootstrap.
All models use one frozen recipe, written down before the final runs (`completion/PREREGISTRATION.md`); design choices for the new models used only held-out tiles of the training region.

**The main result.** On the task as defined (5 observed frames → 5 forecast frames, 30-minute steps), a forecast that merely **continues each pixel's recent depth trajectory** already reaches most of the achievable skill:
a lookup table keyed on just two numbers per pixel (current depth, last one-step change) scores about +0.79 on the never-seen UK (60 m) and +0.74 on Mozambique; a per-pixel MLP (no convolution, no recurrence) about +0.85 and +0.83; the small local CNN +0.84 and +0.81; the small ConvLSTM **+0.87 and +0.81** — against +0.32 / +0.15 for a straight-line (linear) extrapolation and roughly +0.05 to +0.2 for the original CNN-LSTM. Spatial and recurrent structure adds nothing measurable: under the frozen 30-epoch recipe the ConvLSTM is ahead of the local CNN only on the UK at 60 m (+0.025 [+0.004, +0.043], hybrid regime) and equivalent on Mozambique, but that edge is a training-length effect of the plain CNN: with both models trained for 90 epochs the difference is +0.004 [−0.021, +0.027] (UK 60 m) and −0.014 [−0.029, +0.006] (Mozambique), i.e. equivalent; and at 600 training windows the plain CNN is level with the per-pixel MLP. The original CNN-LSTM scores +0.19 on the UK and +0.06 on Mozambique on the same folds (it uses only the rain and cannot read the depth history).

"""

STATUS_HEAD = "## 12. Decided, Built, Not Established\n"

with_tables = RESULTS_INTRO + "\n### Result tables (generated from the result files)\n\n" + report_body + "\n"
SEVERITY_SECTION = "\n### Severity, symbolic rules and Grad-CAM (re-evaluated on the new encoder; Australia 30 m)\n\n" + sev_body + "\n"
EXTRA10 = read("completion/results/explainer_section10_extra.md")      # narrative for ablations / regimes / severity written after the batches finish
STATUS = read("completion/results/explainer_section12.md")
AUDIT = read("completion/results/explainer_section14.md")

new = [HEADER, section(src, 1), section(src, 2)]
s3 = section(src, 3)
old, rep = DATASET_PATCH[0]
assert s3.count(old) == 1, old[:60]
s3 = s3.replace(old, rep)
m = re.search(r"- \*\*Georeferencing:\*\*.*?(?=\n\n|\Z)", s3, re.S)
assert m, "georeferencing bullet not found"
s3 = s3[:m.start()] + DATASET_PATCH[1][1] + s3[m.end():]
new += [s3.rstrip() + "\n" + DATASET_EXTRA, section(src, 4), SCOPE, ARCH, PIPELINE, section(src, 8), VIZ, with_tables + EXTRA10 + SEVERITY_SECTION, section(src, 11), STATUS, WHERE, AUDIT]
text = "\n".join(t.rstrip() + "\n" for t in new) + "\n---\n\n*If the architecture or the results change again, regenerate this file with `completion/build_explainer.py` rather than hand-editing, to avoid drift.*\n"
pend = re.findall(r"\[PENDING[^\]]*\]", text)
out = OLD if FINAL else "completion/results/explainer_draft.md"
if FINAL:
    assert not pend, f"unresolved placeholders: {pend[:5]}"
open(out, "w", encoding="utf-8").write(text)
print("wrote", out, "| words:", len(text.split()), "| pending markers:", len(pend))
