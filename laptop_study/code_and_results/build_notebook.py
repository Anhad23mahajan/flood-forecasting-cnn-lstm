"""Assembles the real-scale Colab notebook from the verified local modules (embedded via
%%writefile cells) plus real-scale training, evaluation, visualization, and the
cross-regional test.

The first draft of this notebook was invalid - a local pre-flight found a degenerate
sampler, leaky split, misaligned regions, no input normalization, a decoder that couldn't
reach a persistence-level forecast, and no baselines (see PROJECT_LOG.md, "Pre-flight
before Colab"). Regenerate with:  python build_notebook.py
"""
import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": text.splitlines(keepends=True)})


def writefile_cell(filename, content):
    code(f"%%writefile {filename}\n" + content)


# ============================================================================
md("""> **SUPERSEDED (September 2026) - kept as the record of the ORIGINAL model (v1).** A later audit found three problems with this notebook's setup:
> (1) the rainfall input carried up to 30 minutes of rain that had **not yet fallen** (rainfall file `j` is the rain falling *during* `[1800 j, 1800 (j+1))` s, but each input frame
> received the file of the interval *containing* it); (2) a trivial per-pixel linear model beats this CNN-LSTM, and a tiny per-pixel/local network scores far higher;
> (3) this model's dense bottleneck could not use the depth history, so its "rainfall drives the forecast" finding is an artefact of that weakness.
> The current, corrected study is **`flood_forecasting_study_colab.ipynb`** (see `PROJECT_LOG.md`, "Completion, round 2"). Numbers printed here describe v1 only.

# CNN-LSTM Flood Forecasting - Real-Scale Training (Colab)

Trains the pipeline at full size on a Colab GPU (T4 is enough: ~10 s per epoch), then evaluates it, visualizes it
and runs a cross-regional test. Every cell of this notebook was executed locally at tiny size first
(`dryrun_notebook.py`); the first full-size run happened on Colab and its numbers are recorded in
`PROJECT_LOG.md`. **Runtime -> Change runtime type -> GPU** first.

## Read this before trusting any number

A local dry-run of the first draft of this notebook found problems that would have produced
misleading results. Each is fixed; the table says what and why.

| Finding (measured on the real data) | What changed |
|---|---|
| **Copying the last frame is almost exact**: persistence RMSE is 2 mm at the native 5-min step (13 mm at a 30-min stride). Thresholding the last frame's max depth predicts severity **99.9%** correctly (96.7% at 30 min). | Every depth and severity result is reported **next to these baselines**. A model that doesn't beat them has learned nothing useful, however small its error looks. |
| The decoder rebuilt the whole depth field from a 128-number vector with no link to the observed frame - it could never reach a near-exact copy. (This, not only "too little data", is why the Stage 2 predictions were blurry.) | **Persistence-anchored decoder**: output = last observed frame + a learned change, zero-initialised so an untrained model *is* the persistence forecast. |
| At the 5-min step, 5 history frames span 25 min, but rainfall is only sampled every 30 min - the rainfall channel was effectively constant. | **`TIME_STRIDE = 6`** (30 min): each step sees its own rainfall value; horizon is 0.5-2.5 h ahead. |
| Severity (max depth in a 128px patch) is **not scale-invariant**: UK is 100% severe, Pakistan ~100%, Mozambique 84%; only Australia has all four classes. | Severity is trained/validated on **Australia only**, and is **not scored on the UK** (an "always severe" model would look perfect). |
| The 60m / 480m depth rasters are resampled, cropped copies of the 30m grid while DEM/rainfall/land-use stay on the 30m grid; UK's layers are offset by 5px. | Every region/resolution has a **measured, cross-checked geometry** (`LAYER_GEOMETRY`); one without it is refused. Pakistan, Mozambique and the 60m data are covered by `flood_forecasting_multires_colab.ipynb`. |
| First-draft sampler never reached the second region; split was random over windows from the *same* patches (leaks). | Random sampling per region; **split by spatial position** - validation patches are never seen in training. |
| Raw inputs: DEM in hundreds of metres next to rainfall in mm. | Fixed normalization inside the model; DEM is patch-relative so it can transfer across regions. |

**This notebook: train on Australia, test on the UK at 30m.** The paper's other two transfer tests -
Australia -> UK at 60m and Pakistan -> Mozambique at 480m - are in `flood_forecasting_multires_colab.ipynb`.
Mixing resolutions in one training set is deliberately not attempted: a 128px patch spans 3.8 km, 7.7 km or
61 km depending on the resolution. The severity task is also close to trivial under this label definition; see below.""")

md("## Setup")
code("""!pip install -q huggingface_hub rasterio scipy folium
import warnings
import tensorflow as tf
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
print("GPU available:", tf.config.list_physical_devices('GPU'))
print("TensorFlow version:", tf.__version__)""")

md("""## Configuration

Edit here. Defaults suit a Colab GPU session; reduce `PER_CLASS_TRAIN` / `EPOCHS` for a quick first pass.""")
code("""PATCH_SIZE = 128
T_IN, HORIZON = 5, 5            # 5 observed steps -> 5 forecast steps
TIME_STRIDE = 6                 # depth steps between frames: 6 x 5 min = 30 min (rainfall's native resolution)
PER_CLASS_TRAIN = 100           # class-balanced training windows per severity class
PER_CLASS_VAL = 60              # class-balanced validation windows per class (25 gave +-4.6 points of noise on accuracy)
EPOCHS = 40
LEARNING_RATE = 1e-3            # peak; decays on a cosine schedule to 2% of this by the last step
CLIPNORM = 1.0                  # gradient clipping; batch size 1 makes single steps noisy
N_START_INDICES = 80           # time starts scanned when building Australia's window pool
N_UNIFORM_AU = 150              # unbalanced Australia val windows, for the like-for-like comparison with UK
UK_START_STEP = 12              # UK: every 12th start index (x 2 spatial patches)
CENSUS_START_STEP = 24          # start-index step for the severity-class census
VAL_POSITION_FRACTION = 0.25    # fraction of Australia's patches held out entirely for validation
DEM_NODATA_STD = 1.0            # skip patches whose DEM is (near-)flat: outside the simulation domain
SEED = 42
AUGMENT = False                 # random flips / 90-degree rotations of training windows. OFF: helped severity on 2 of 2 seeds but destabilised depth on one (PROJECT_LOG.md)""")

md("""## Pull the dataset from Hugging Face

Only Australia and UK, and only their own DEM / land-use / rainfall files. **This is still large:
about 13.5GB in 4,385 files, almost all of it Australia's 2,881 depth frames (4.6MB each)** - so expect this
first cell to be the slowest part of the notebook (measured on Colab: 7 min 43 s).
Colab's disk is ephemeral, so this re-downloads each fresh session unless you cache it to Drive.""")
code("""from huggingface_hub import snapshot_download

DATA_ROOT = snapshot_download(
    repo_id="mahajananhad/floodcastbench",
    repo_type="dataset",
    allow_patterns=[
        "High-fidelity flood forecasting/30m/Australia/*",
        "High-fidelity flood forecasting/30m/UK/*",
        "Relevant data/DEM/Australia_DEM.*",
        "Relevant data/DEM/UK_DEM.*",
        "Relevant data/Land use and land cover/Australia.*",
        "Relevant data/Land use and land cover/UK.*",
        "Relevant data/Rainfall/Australia flood/*",
        "Relevant data/Rainfall/UK flood/*",
        "Relevant data/Georeferenced files/*",
    ],
    max_workers=16,  # thousands of small files: more parallel downloads help
)
print("Downloaded to:", DATA_ROOT)""")

md("## Core modules (verified locally)")
for fname in ["real_data_loader.py", "capsule_layers.py", "model.py", "grad_cam.py",
              "symbolic_rule_layer.py", "predict_severity.py", "windows.py"]:
    writefile_cell(fname, open(fname, encoding="utf-8").read())

md("## Load the two usable regions")
code("""import sys
sys.path.insert(0, ".")
import numpy as np
from real_data_loader import RegionData, SEVERITY_CLASSES, depth_to_severity

REGIONS = {
    "Australia": "High-fidelity flood forecasting/30m",
    "UK": "High-fidelity flood forecasting/30m",
}
regions = {name: RegionData(root_dir=DATA_ROOT, region=name, fidelity_dir=fid) for name, fid in REGIONS.items()}
for name, rd in regions.items():
    print(f"{name}: {len(rd.depth_files)} depth frames, grid {rd.grid_height}x{rd.grid_width}, "
          f"{rd.pixel_size_m:.0f}m, layer offset {rd.layer_offset}")

print("(Pakistan, Mozambique and the 60m data are not downloaded here - see flood_forecasting_multires_colab.ipynb.)")""")

md("""## Window pool and severity-class census

A window is one (spatial patch, start time). Severity is computed exactly as the loader computes it
(max depth in the patch over the forecast frames). The census reproduces the pre-flight finding.""")
code("""import rasterio
from windows import (valid_positions, start_indices_for, scan_windows, class_fractions, balanced_sample)

print("Severity-class census (fraction of windows in each class):")
for name, rd in regions.items():
    positions = valid_positions(rd, PATCH_SIZE, DEM_NODATA_STD)
    starts = list(range(0, rd.max_start_index(T_IN, HORIZON, TIME_STRIDE) + 1, CENSUS_START_STEP))
    fr = class_fractions(scan_windows(rd, positions, starts, PATCH_SIZE, T_IN, HORIZON, TIME_STRIDE))
    print(f"  {name:10s} patches={len(positions):2d}  " + "  ".join(f"{c}={100 * fr[c]:.0f}%" for c in SEVERITY_CLASSES))
print("-> Only Australia has real class diversity. Severity is trained and validated there only.")""")

md("""## Australia: split by spatial position, then class-balance within each side

Validation patches are **never seen in training** - not even at other times.""")
code("""rd_au = regions["Australia"]
rng = np.random.default_rng(SEED)

au_positions = valid_positions(rd_au, PATCH_SIZE, DEM_NODATA_STD)
perm = rng.permutation(len(au_positions))
n_val_pos = max(1, int(round(len(au_positions) * VAL_POSITION_FRACTION)))
val_positions = [au_positions[i] for i in perm[:n_val_pos]]
train_positions = [au_positions[i] for i in perm[n_val_pos:]]
print(f"Australia patches: {len(au_positions)} valid -> {len(train_positions)} train / {len(val_positions)} val")
assert not set(train_positions) & set(val_positions)

starts = start_indices_for(rd_au, N_START_INDICES, T_IN, HORIZON, TIME_STRIDE)
print(f"scanning {len(starts)} start times x {len(au_positions)} patches ...")
train_pool = scan_windows(rd_au, train_positions, starts, PATCH_SIZE, T_IN, HORIZON, TIME_STRIDE)
val_pool = scan_windows(rd_au, val_positions, starts, PATCH_SIZE, T_IN, HORIZON, TIME_STRIDE)

train_windows = balanced_sample(train_pool, PER_CLASS_TRAIN, rng)
val_windows = balanced_sample(val_pool, PER_CLASS_VAL, rng)
uniform_idx = rng.choice(len(val_pool), size=min(N_UNIFORM_AU, len(val_pool)), replace=False)
au_uniform_windows = [val_pool[i] for i in uniform_idx]
print(f"train={len(train_windows)}  val(balanced)={len(val_windows)}  val(uniform, for UK comparison)={len(au_uniform_windows)}")""")

md("## Load window arrays")
code("""def load_examples(rd, windows):
    examples = []
    for (row, col, start_idx, expected_sev, _baseline) in windows:
        input_seq, depth_targets, severity, _ = rd.load_patch_sequence(
            row, col, PATCH_SIZE, start_idx, T_IN, HORIZON, TIME_STRIDE
        )
        assert severity == expected_sev, "scan and loader disagree on severity - they must use the same rule"
        examples.append((input_seq, depth_targets, severity))
    return examples


def shuffled(items, seed):
    order = np.random.default_rng(seed).permutation(len(items))
    return [items[i] for i in order]


train_windows = shuffled(train_windows, 1)
val_windows = shuffled(val_windows, 2)

print("loading training windows ...")
train_data = load_examples(rd_au, train_windows)
print("loading validation windows ...")
val_data = load_examples(rd_au, val_windows)
au_uniform_data = load_examples(rd_au, au_uniform_windows)
print(f"train {len(train_data)}  val {len(val_data)}  au_uniform {len(au_uniform_data)}")""")

md("""## The baselines everything is judged against

- **Depth - persistence:** predict that every future frame equals the last observed frame.
- **Severity - threshold persistence:** apply the severity thresholds to the max depth of the
  last observed frame. No model, no training.

The training loss below is `MSE / persistence MSE`: **1.0 means exactly persistence, below 1.0
means better than persistence** (validation skill = 1 - that ratio).""")
code("""def persistence_mse(examples):
    return float(np.mean([np.mean((y - x[-1, ..., 0][None]) ** 2) for x, y, _ in examples]))


PERSISTENCE_MSE = max(persistence_mse(train_data), 1e-12)
print(f"persistence MSE on training windows: {PERSISTENCE_MSE:.3e} m^2  (RMSE {np.sqrt(PERSISTENCE_MSE) * 1000:.1f} mm)")

val_baseline_acc = float(np.mean([w[4] == w[3] for w in val_windows]))
print(f"severity baseline (threshold the last observed frame): {val_baseline_acc:.1%} on the balanced validation windows")
print(f"severity chance level on the balanced validation windows: 25.0%")""")

md("""## Build the model and compiled train / predict steps

Combined depth + severity loss through the ONE shared encoder (Stage 3's verified architecture),
`@tf.function`-compiled for GPU speed.""")
code("""import model as flood_model
from model import decode_horizon
from capsule_layers import margin_loss
import tensorflow as tf

tf.keras.utils.set_random_seed(SEED)
encoder = flood_model.build_encoder(hidden_dim=128, dropout=0.3)
decoder = flood_model.build_decoder_step(hidden_dim=128, dropout=0.3)
capsule_head = flood_model.build_capsule_head(return_lengths=True)  # raw capsule lengths -> margin loss
# Constant-LR batch-size-1 Adam made epoch-to-epoch validation swing by several points in local runs
# (and left the final epoch's weights to luck). Cosine decay makes the LAST weights the stable ones.
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    LEARNING_RATE, decay_steps=EPOCHS * len(train_data), alpha=0.02)
optimizer = flood_model.OPTIMIZERS["adam"](learning_rate=lr_schedule, clipnorm=CLIPNORM)
all_vars = encoder.trainable_variables + decoder.trainable_variables + capsule_head.trainable_variables

X_SPEC = tf.TensorSpec(shape=[1, T_IN, PATCH_SIZE, PATCH_SIZE, 3], dtype=tf.float32)
Y_SPEC = tf.TensorSpec(shape=[HORIZON, PATCH_SIZE, PATCH_SIZE], dtype=tf.float32)
S_SPEC = tf.TensorSpec(shape=[1], dtype=tf.int32)


@tf.function(input_signature=[X_SPEC, Y_SPEC, S_SPEC])
def train_step(x, depth_targets, severity_label):
    with tf.GradientTape() as tape:
        enc_out = encoder(x, training=True)
        frames = decode_horizon(decoder, enc_out, x, HORIZON, training=True)
        depth_loss = tf.reduce_mean(tf.square(frames[0] - depth_targets)) / PERSISTENCE_MSE
        severity_lengths = capsule_head(enc_out[2][:, -1], training=True)
        severity_loss = margin_loss(severity_label, severity_lengths)
        loss = depth_loss + severity_loss
    grads = tape.gradient(loss, all_vars)
    optimizer.apply_gradients((g, v) for g, v in zip(grads, all_vars) if g is not None)
    return depth_loss, severity_loss


@tf.function(input_signature=[X_SPEC])
def predict_step(x):
    enc_out = encoder(x, training=False)
    frames = decode_horizon(decoder, enc_out, x, HORIZON, training=False)
    severity_lengths = capsule_head(enc_out[2][:, -1], training=False)
    return frames[0], tf.argmax(severity_lengths[0])


def predict(input_seq):
    frames, sev = predict_step(tf.constant(input_seq[np.newaxis, ...], dtype=tf.float32))
    return frames.numpy(), int(sev)""")

md("""## Train

Each epoch prints validation **depth skill vs persistence** (positive = beats it) and severity
accuracy on patches the model has never seen. Epoch 0 is a sanity check: the untrained model
*is* persistence, so its skill must be ~0.""")
code("""import time


def val_scores(examples):
    mses, pers, correct = [], [], 0
    for x, y, s in examples:
        frames, pred_sev = predict(x)
        mses.append(np.mean((frames - y) ** 2))
        pers.append(np.mean((x[-1, ..., 0][None] - y) ** 2))
        correct += int(pred_sev == s)
    return 1 - np.mean(mses) / np.mean(pers), correct / len(examples)


skill0, acc0 = val_scores(val_data)
print(f"epoch  0 (untrained)  val_depth_skill={skill0:+.4f}  val_sev_acc={acc0:.1%}   <- skill must be ~0")
assert abs(skill0) < 1e-3, "untrained model should equal persistence"

def augment(x, y, rng):
    \"\"\"One random element of the 8 flips/rotations of the square patch, applied identically to the input
    frames and the depth targets. A mirrored or rotated flood scene is an equally valid scene (terrain,
    water and rain are transformed together) and the severity label does not change.\"\"\"
    k, flip = int(rng.integers(4)), bool(rng.integers(2))
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(1, 2))
    if flip:
        x, y = x[:, :, ::-1], y[:, :, ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


aug_rng = np.random.default_rng(SEED + 1)
history = [(0, 1.0, float("nan"), skill0, acc0)]
for epoch in range(1, EPOCHS + 1):
    t0 = time.perf_counter()
    dls, sls = [], []
    for i in np.random.permutation(len(train_data)):
        input_seq, depth_targets, severity = train_data[i]
        if AUGMENT:
            input_seq, depth_targets = augment(input_seq, depth_targets, aug_rng)
        dl, sl = train_step(
            tf.constant(input_seq[np.newaxis, ...], dtype=tf.float32),
            tf.constant(depth_targets, dtype=tf.float32),
            tf.constant([severity], dtype=tf.int32),
        )
        dls.append(float(dl)); sls.append(float(sl))
    skill, acc = val_scores(val_data)
    history.append((epoch, np.mean(dls), np.mean(sls), skill, acc))
    print(f"epoch {epoch:2d}/{EPOCHS}  train_depth/persist={np.mean(dls):.4f}  train_sev_loss={np.mean(sls):.4f}  "
          f"val_depth_skill={skill:+.4f}  val_sev_acc={acc:.1%}  ({time.perf_counter() - t0:.1f}s)")""")

code("""import matplotlib.pyplot as plt

ep, tr_d, tr_s, va_skill, va_acc = zip(*history)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(ep, tr_d); axes[0].axhline(1.0, color="gray", linestyle="--", label="= persistence")
axes[0].set_title("Train depth loss / persistence loss"); axes[0].legend()
axes[1].plot(ep, va_skill); axes[1].axhline(0.0, color="gray", linestyle="--", label="= persistence")
axes[1].set_title("Val depth skill (>0 beats persistence)"); axes[1].legend()
axes[2].plot(ep, va_acc, label="capsule")
axes[2].axhline(0.25, color="gray", linestyle="--", label="chance")
axes[2].axhline(val_baseline_acc, color="green", linestyle="--", label="threshold baseline")
axes[2].set_title("Val severity accuracy"); axes[2].legend()
plt.tight_layout(); plt.show()""")

md("""**Reading the severity curve.** The capsule head is trained with the CapsNet paper's *margin loss* on
capsule lengths (true class above 0.9, others below 0.1). The first version used softmax
cross-entropy on the lengths; because lengths are bounded in [0, 1), that softmax can never exceed
~0.475 for one of 4 classes, and locally the head collapsed to predicting a single class.

Honest caveats that remain regardless of loss:
- The label is "max depth in the patch" - exactly what the threshold baseline measures - so a capsule head
  that merely matches the baseline adds nothing a threshold doesn't. The interesting question is the windows
  where the class *changes*; the evaluation cell reports the capsule's accuracy there separately.
- **Overfitting.** With only ~46 training patches the head can memorise them (in the first Colab run the
  training margin loss fell to 0.026 while validation accuracy stayed flat). `AUGMENT` (random flips and
  rotations of each training window) was added to attack this. In local A/B tests on two seeds it raised
  severity accuracy both times (+18 and +13.5 points) but, on the second seed, training destabilised late and
  depth skill fell (UK +0.21 -> +0.04). It is therefore **off by default**; set `AUGMENT = True` to try it.
  Details and the caveats are in `PROJECT_LOG.md`.
- Whether capsule routing dilutes a single deep pixel is a suspicion, not something that was tested.""")

md("""## Evaluation

RMSE, NSE, r, CSI (flood-extent skill; a pixel is wet above 0.1 m). **Skill** = 1 - (model RMSE /
persistence RMSE)^2; positive means the model beats persistence.

**Comparison with the FloodCastBench paper's U-Net / FNO / FNO+:** take the published numbers from the
paper's own tables. They are deliberately not typed in here - they were not independently verified,
and the paper's evaluation setup (whole scenes, its own horizons) differs from this patch-based one.""")
code("""def compute_metrics(pred, obs, wet_threshold=0.1):
    p, o = pred.flatten(), obs.flatten()
    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    denom = np.sum((o - o.mean()) ** 2)
    nse = float(1 - np.sum((o - p) ** 2) / denom) if denom > 0 else float("nan")
    r = float(np.corrcoef(p, o)[0, 1]) if p.std() > 0 and o.std() > 0 else float("nan")
    pw, ow = p > wet_threshold, o > wet_threshold
    hits, misses, false_alarms = int(np.sum(pw & ow)), int(np.sum(~pw & ow)), int(np.sum(pw & ~ow))
    denom_csi = hits + misses + false_alarms
    csi = hits / denom_csi if denom_csi > 0 else float("nan")
    return {"RMSE": rmse, "NSE": nse, "r": r, "CSI": csi}


def skill(model_m, base_m):
    return float(1 - (model_m["RMSE"] / base_m["RMSE"]) ** 2)


def evaluate_depth(examples):
    preds, obs, persistence = [], [], []
    for input_seq, depth_targets, _ in examples:
        frames, _ = predict(input_seq)
        preds.append(frames)
        obs.append(depth_targets)
        persistence.append(np.repeat(input_seq[-1, ..., 0][None], HORIZON, axis=0))
    obs = np.concatenate(obs)
    return compute_metrics(np.concatenate(preds), obs), compute_metrics(np.concatenate(persistence), obs)


def severity_report(examples, baseline_labels):
    y_true = np.array([s for _, _, s in examples])
    y_pred = np.array([predict(x)[1] for x, _, _ in examples])
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    present = [k for k in range(4) if (y_true == k).any()]
    recalls = [cm[k, k] / cm[k].sum() for k in present]
    baseline = np.array(baseline_labels)
    base_right = baseline == y_true
    return {"capsule accuracy": float((y_true == y_pred).mean()),
            "capsule balanced accuracy": float(np.mean(recalls)),
            "threshold-baseline accuracy": float(base_right.mean()),
            "n baseline right": int(base_right.sum()),
            "capsule acc where baseline is right": float((y_pred == y_true)[base_right].mean()) if base_right.any() else float("nan"),
            "n baseline wrong": int((~base_right).sum()),
            "capsule acc where baseline is wrong": float((y_pred == y_true)[~base_right].mean()) if (~base_right).any() else float("nan"),
            "per-class recall": {SEVERITY_CLASSES[k]: float(cm[k, k] / cm[k].sum()) for k in present},
            "confusion": cm}


au_model, au_base = evaluate_depth(val_data)
print("Depth - Australia held-out patches (class-balanced windows):")
print(f"  {'metric':6s} {'model':>10s} {'persistence':>13s}")
for k in au_model:
    print(f"  {k:6s} {au_model[k]:10.5f} {au_base[k]:13.5f}")
print(f"  skill vs persistence = {skill(au_model, au_base):+.4f}   (positive = beats persistence)")

sev = severity_report(val_data, [w[4] for w in val_windows])
print("\\nSeverity - Australia held-out patches (balanced, chance = 25%):")
for k in ("capsule accuracy", "capsule balanced accuracy", "threshold-baseline accuracy"):
    print(f"  {k:28s} {sev[k]:.1%}")
print(f"  where the threshold baseline is right (n={sev['n baseline right']}): capsule {sev['capsule acc where baseline is right']:.1%}")
wrong_acc = sev['capsule acc where baseline is wrong']
print(f"  where the threshold baseline is wrong (n={sev['n baseline wrong']}; the class changes over the forecast): "
      f"capsule {'n/a' if sev['n baseline wrong'] == 0 else format(wrong_acc, '.1%')}  (baseline 0% by definition)")
print("  per-class recall:", {k: f"{v:.0%}" for k, v in sev["per-class recall"].items()})
print("  confusion matrix (rows = true none/minor/moderate/severe, cols = capsule prediction):")
print(sev["confusion"])""")

md("""## Grad-CAM on a real trained example

Locally (9 training examples) the heatmap was a smooth positional gradient unrelated to the flood
channel. The decoder is now persistence-anchored, so this heatmap explains what drives the model's
predicted **change** from the last frame, in whichever direction the model predicts it (Grad-CAM's
ReLU would otherwise blank the whole map whenever the model predicts net drying - seen in a local run).
The number to read is the gap between the two means: a clearly higher mean on wet pixels is what a
meaningful explanation looks like.""")
code("""from grad_cam import grad_cam

heat_idx = int(np.argmax([x[-1, ..., 0].mean() for x, _, _ in val_data]))  # wettest validation example
sample_input, _, _ = val_data[heat_idx]
heatmap, net_change_m = grad_cam(encoder, decoder, sample_input, horizon_step=0)
n_pixels = PATCH_SIZE * PATCH_SIZE
print(f"model predicts a net {'RISE' if net_change_m >= 0 else 'FALL'} of "
      f"{abs(net_change_m) / n_pixels * 1000:.2f} mm in mean depth at the first forecast step "
      f"-> the map shows evidence for that direction")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
last_depth = sample_input[-1, ..., 0]
axes[0].imshow(last_depth, cmap="Blues"); axes[0].set_title("last observed depth"); axes[0].axis("off")
axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM heatmap"); axes[1].axis("off")
axes[2].imshow(last_depth, cmap="Blues"); axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
axes[2].set_title("overlay"); axes[2].axis("off")
plt.tight_layout(); plt.show()

wet = last_depth > last_depth.mean()
print(f"mean heatmap on above-average-depth pixels: {heatmap[wet].mean():.4f}")
print(f"mean heatmap on below-average-depth pixels: {heatmap[~wet].mean():.4f}")

# One window is an anecdote: in local runs this same number flipped from 0.34-vs-0.02 to 0.09-vs-0.18 on
# different windows. Measure it over many evenly-spaced validation windows that actually contain water.
picked = np.linspace(0, len(val_data) - 1, min(40, len(val_data))).astype(int)
wet_heat, dry_heat = [], []
for i in picked:
    x_i, _, _ = val_data[i]
    d_i = x_i[-1, ..., 0]
    if d_i.max() <= 0.1:          # no water to compare against
        continue
    hm_i, _ = grad_cam(encoder, decoder, x_i, horizon_step=0)
    w_i = d_i > d_i.mean()
    if hm_i.max() == 0 or w_i.all() or not w_i.any():
        continue
    wet_heat.append(hm_i[w_i].mean()); dry_heat.append(hm_i[~w_i].mean())
wet_heat, dry_heat = np.array(wet_heat), np.array(dry_heat)
if len(wet_heat):
    print(f"across {len(wet_heat)} validation windows with water: heat is higher on wet than on dry pixels in "
          f"{(wet_heat > dry_heat).mean():.0%} of them (50% would be chance); "
          f"median wet/dry ratio {np.median(wet_heat / np.maximum(dry_heat, 1e-9)):.1f}x")""")

md("""## Symbolic reasoning on held-out Australian patches

The rule thresholds were derived from Australia at 30m using the native 5-minute step; depth-rate is
computed per hour of real time, so it stays comparable at this 30-minute stride. This demo is
Australia only (the function warns if pointed at any other resolution).""")
code("""from predict_severity import predict_severity_with_reasoning

for (row, col, start_idx, _sev, _base) in val_windows[:8]:
    result = predict_severity_with_reasoning(
        encoder, decoder, capsule_head, rd_au, row, col, PATCH_SIZE, start_idx, T_IN, HORIZON, TIME_STRIDE
    )
    f = result["features"]
    print(f"true={result['true_severity']:9s} raw={result['raw_severity']:9s} reasoned={result['reasoned_severity']:9s}")
    print(f"  depth_rate={f['depth_rate_m_per_hr']:.4f} m/hr  rainfall={f['rainfall_mm']:.2f} mm  "
          f"slope={f['slope_pct']:.2f}%  drainage={f['drainage_proximity_m']:.0f} m")
    print(f"  {result['justification']}\\n")""")

md("""## Does the symbolic layer improve accuracy?

The demo above shows 8 windows, which proves nothing about accuracy. This runs the full capsule -> rules
pipeline on **every** balanced validation window and counts what the rules did to the capsule's answer:
how many they changed, how many of those changes fixed a wrong answer, and how many broke a right one.

Caveat: the rule thresholds were derived from real Australian windows before this notebook existed, so they
are not tuned on this validation set, but they were not held out from Australian patches in general either.""")
code("""from predict_severity import predict_severity_with_reasoning

raw_ok = reasoned_ok = changed = fixed = broken = 0
for (row, col, start_idx, _sev, _base) in val_windows:
    res = predict_severity_with_reasoning(
        encoder, decoder, capsule_head, rd_au, row, col, PATCH_SIZE, start_idx, T_IN, HORIZON, TIME_STRIDE)
    true, raw, reasoned = res["true_severity"], res["raw_severity"], res["reasoned_severity"]
    raw_ok += raw == true
    reasoned_ok += reasoned == true
    if raw != reasoned:
        changed += 1
        fixed += reasoned == true      # wrong -> right
        broken += raw == true          # right -> wrong
n_val = len(val_windows)
print(f"capsule alone      : {raw_ok / n_val:.1%}")
print(f"capsule + rules    : {reasoned_ok / n_val:.1%}")
print(f"rules changed the class in {changed} of {n_val} windows: {fixed} fixed a wrong answer, {broken} broke a right one, "
      f"{changed - fixed - broken} moved one wrong answer to another wrong class")""")

md("""## Visualization

Attaches the region's real CRS and affine transform (from the archive's world files - the raw depth
TIFFs carry none) to a predicted depth grid and renders it on a real basemap.

**Basemap choice (learned the hard way):** folium's default OpenStreetMap tiles do not work in Colab -
OSM's servers reject requests that carry no `Referer` header, which Colab's output frame does not send,
and answer every tile with a "403 Access blocked" picture. CartoDB's free tiles now stamp "API KEY
REQUIRED" across the map. Esri World Imagery returned real tiles with and without a referrer, so it is
used here. A static plot with no network dependency follows the interactive map either way.

Scope: the world files describe the **30m base grid**, so this is valid for the 30m depth rasters (Australia, UK).
The 60m / 480m depth rasters sit on coarser cells (scale 2 / 16 with a crop offset, `LAYER_GEOMETRY`), so the transform would
need that scale and offset applied.

The world-file origin is the *centre* of the upper-left pixel; an affine transform wants that pixel's
*corner*, so the origin is shifted by half a pixel.""")
code("""import re
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
import folium


def utm_zone_to_epsg(zone_str):
    m = re.search(r"Zone_(\\d+)([NS])", zone_str)
    return (32600 if m.group(2) == "N" else 32700) + int(m.group(1))


def load_spatial_reference(data_root, region_name):
    with open(f"{data_root}/Relevant data/Georeferenced files/Spatial_References.txt") as f:
        for line in f:
            if region_name.lower() in line.lower():
                return utm_zone_to_epsg(line.split(":")[1].strip())
    raise ValueError(f"No spatial reference entry for {region_name}")


def load_world_file(data_root, region_name):
    with open(f"{data_root}/Relevant data/Georeferenced files/{region_name}.tfw") as f:
        a, _d, _b, e, c, fy = (float(x) for x in f.read().split())
    return a, e, c, fy  # pixel width, pixel height (negative), x/y of upper-left pixel CENTRE


row, col, start_idx, _, _ = val_windows[heat_idx]  # same (wettest) example the Grad-CAM cell used
input_seq, _, _, _ = rd_au.load_patch_sequence(row, col, PATCH_SIZE, start_idx, T_IN, HORIZON, TIME_STRIDE)
pred = predict(input_seq)[0][0]  # first forecast step

epsg = load_spatial_reference(DATA_ROOT, "Australia")
pixel_w, pixel_h, cx, cy = load_world_file(DATA_ROOT, "Australia")
corner_x, corner_y = cx - pixel_w / 2, cy - pixel_h / 2
transform = Affine(pixel_w, 0, corner_x + col * pixel_w, 0, pixel_h, corner_y + row * pixel_h)

out_path = "predicted_depth_georeferenced.tif"
with rasterio.open(out_path, "w", driver="GTiff", height=pred.shape[0], width=pred.shape[1], count=1,
                   dtype="float32", crs=f"EPSG:{epsg}", transform=transform) as dst:
    dst.write(pred.astype("float32"), 1)
print(f"wrote {out_path} with CRS EPSG:{epsg}")

with rasterio.open(out_path) as src:
    lon_min, lat_min, lon_max, lat_max = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
print(f"patch spans lat {lat_min:.4f}..{lat_max:.4f}, lon {lon_min:.4f}..{lon_max:.4f}")

# scrollWheelZoom=False: otherwise scrolling down the notebook with the pointer over the map zooms it away.
m = folium.Map(location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], zoom_start=13, tiles=None,
               scrollWheelZoom=False)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    name="Esri World Imagery", max_zoom=19,
).add_to(m)
folium.raster_layers.ImageOverlay(
    image=np.clip(np.sqrt(pred / (pred.max() + 1e-9)), 0, 1),  # sqrt so shallow water is visible next to deep
    bounds=[[lat_min, lon_min], [lat_max, lon_max]],
    opacity=0.6, colormap=lambda x: (0, 0, 1, x),
).add_to(m)
folium.Rectangle(bounds=[[lat_min, lon_min], [lat_max, lon_max]], color="red", weight=2, fill=False,
                 tooltip="predicted patch").add_to(m)
m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])

# static version - needs no tile server. The axes are lon/lat of the (UTM) patch's bounding box.
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(pred, cmap="Blues", extent=[lon_min, lon_max, lat_min, lat_max])
ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("predicted depth, first forecast step (m)")
fig.colorbar(im, ax=ax, shrink=0.8); plt.tight_layout(); plt.show()
m""")

md("""## Cross-regional generalization (Australia -> UK)

The model never saw the UK. Depth is scored against the same persistence baseline. The Australia rows
use a **uniformly sampled** (not class-balanced) set so the two regions are like-for-like.

**Severity is not scored on the UK, on purpose:** every UK window is "severe" under this label
definition, so an "always severe" model would look perfect. The census below shows that directly.""")
code("""rd_uk = regions["UK"]
uk_positions = valid_positions(rd_uk, PATCH_SIZE, DEM_NODATA_STD)
uk_starts = list(range(0, rd_uk.max_start_index(T_IN, HORIZON, TIME_STRIDE) + 1, UK_START_STEP))
uk_windows = scan_windows(rd_uk, uk_positions, uk_starts, PATCH_SIZE, T_IN, HORIZON, TIME_STRIDE)
uk_counts = {c: sum(1 for w in uk_windows if w[3] == i) for i, c in enumerate(SEVERITY_CLASSES)}
print(f"UK: {len(uk_positions)} patches x {len(uk_starts)} times = {len(uk_windows)} windows")
print("UK severity census:", uk_counts, "-> severity is NOT assessable on the UK")

uk_data = load_examples(rd_uk, uk_windows)
uk_model, uk_base = evaluate_depth(uk_data)
au_u_model, au_u_base = evaluate_depth(au_uniform_data)

print("\\nDepth, uniformly-sampled windows (like-for-like):")
print(f"  {'':24s} {'RMSE':>9s} {'NSE':>9s} {'r':>9s} {'CSI':>9s}")
for label, mtr in [("Australia model", au_u_model), ("Australia persistence", au_u_base),
                   ("UK model (held out)", uk_model), ("UK persistence", uk_base)]:
    print(f"  {label:24s} {mtr['RMSE']:9.5f} {mtr['NSE']:9.5f} {mtr['r']:9.5f} {mtr['CSI']:9.5f}")

sk_au, sk_uk = skill(au_u_model, au_u_base), skill(uk_model, uk_base)
print("\\n=== SUMMARY ===")
print(f"Depth, Australia held-out patches : skill vs persistence = {sk_au:+.4f}  "
      f"({'beats' if sk_au > 0 else 'does NOT beat'} persistence)")
print(f"Depth, UK (never seen)            : skill vs persistence = {sk_uk:+.4f}  "
      f"({'beats' if sk_uk > 0 else 'does NOT beat'} persistence)")
print(f"Severity, Australia held-out      : capsule {sev['capsule accuracy']:.1%} vs threshold baseline "
      f"{sev['threshold-baseline accuracy']:.1%} (chance 25%)")
print("A cross-regional transfer claim is supported only if the UK skill is positive.")""")

md("""## What does the trained model actually use?

A positive skill does not say *why*. This removes one input at a time from the trained model's test data and re-measures
skill (and severity accuracy). A large drop means the model relies on that input; a drop near zero means it does not - and
then a claim that the model exploits it (for example "the LSTM captures temporal context") is unsupported.

In the audited local runs the answer was stark: removing **rainfall** took skill from about +0.22 to about 0; removing
terrain, scrambling the order of the depth history, or freezing it changed skill by less than 0.01. The forecast is
essentially *last frame + current rainfall*. (Test-time removal measures reliance, not what a differently-trained model could
learn; and the severity head *does* use terrain.) Also reported: skill at each lead time.""")
code("""def all_preds(examples, transform=None):
    preds, obs, pers, ok = [], [], [], 0
    for x, y, s in examples:
        frames, sev = predict(x if transform is None else transform(x))
        preds.append(frames); obs.append(y)
        pers.append(np.repeat(x[-1, ..., 0][None], HORIZON, axis=0))  # persistence always uses the TRUE last frame
        ok += int(sev == s)
    return np.stack(preds), np.stack(obs), np.stack(pers), ok / len(examples)


def skill_np(p, o, q):
    return 1.0 - np.mean((p - o) ** 2) / np.mean((q - o) ** 2)


def no_rain(x):
    y = x.copy(); y[..., 2] = 0.0; return y


def flat_terrain(x):
    y = x.copy(); y[..., 1] = 0.0; return y  # the DEM is made patch-relative inside the model, so zero = flat


def scrambled_history(x):
    return np.concatenate([x[:-1][::-1], x[-1:]], axis=0)  # reverse the first frames, keep the last (the anchor)


def frozen_history(x):
    return np.repeat(x[-1:], x.shape[0], axis=0)  # every history frame = the last one: no temporal information


rows_skill = []
print(f"{'input removed at test time':32s} {'AU skill':>9s} {'UK skill':>9s} {'severity acc':>13s}")
for name, fn in [("none (normal)", None), ("rainfall", no_rain), ("terrain (flat DEM)", flat_terrain),
                 ("history order scrambled", scrambled_history), ("history frozen (no dynamics)", frozen_history)]:
    pa, oa, qa, acc_a = all_preds(val_data, fn)
    pu, ou, qu, _ = all_preds(uk_data, fn)
    rows_skill.append(skill_np(pa, oa, qa))
    print(f"{name:32s} {skill_np(pa, oa, qa):+9.3f} {skill_np(pu, ou, qu):+9.3f} {acc_a:12.1%}")
if max(rows_skill) - min(rows_skill) < 5e-4:
    print("WARNING: skill is IDENTICAL whichever input is removed -> the network is input-blind (it collapsed to a constant "
          "offset during training). These numbers are NOT a result; rerun with a different SEED or a lower LEARNING_RATE.")

print("skill by lead time (30-minute steps):")
for name, ex in [("Australia held-out", val_data), ("UK (never seen)", uk_data)]:
    p, o, q, _ = all_preds(ex)
    print(f"  {name:20s}", "  ".join(f"{1 - np.mean((p[:, k] - o[:, k]) ** 2) / np.mean((q[:, k] - o[:, k]) ** 2):+.3f}"
                                    for k in range(HORIZON)))""")

md("## Save the trained model")
code("""encoder.save_weights("encoder.weights.h5")
decoder.save_weights("decoder.weights.h5")
capsule_head.save_weights("capsule_head.weights.h5")
print("saved. To persist beyond this session: from google.colab import drive; drive.mount('/content/drive'), "
      "then copy these files (and predicted_depth_georeferenced.tif) to your Drive.")""")


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("flood_forecasting_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Wrote flood_forecasting_colab.ipynb with {len(cells)} cells")
