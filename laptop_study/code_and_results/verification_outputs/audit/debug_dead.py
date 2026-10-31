"""Does the network die during training on Australia 60 m?  Re-run the first steps of the E2 recipe (same seed, same loss, same optimiser)
and watch, every 10 steps: the loss, the gradient norm BEFORE clipping, the fraction of active (non-zero) ReLU units in each encoder / decoder layer,
and how much the predicted change varies across windows (a dead network predicts the same thing for every window).
Usage: python debug_dead.py <australia60|pakistan> [lr]"""
import os
import sys

import numpy as np
import tensorflow as tf

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ".")
import depth_experiment as de
import model as fm
from model import decode_horizon
from real_data_loader import RegionData
from windows import valid_positions, start_indices_for

kind = sys.argv[1]
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
NW = int(sys.argv[3]) if len(sys.argv) > 3 else 80
NEP = int(sys.argv[4]) if len(sys.argv) > 4 else 2
WARM = int(sys.argv[5]) if len(sys.argv) > 5 else 0
CLIP = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
region, res = {"australia60": ("Australia", 60), "pakistan": ("Pakistan", 480)}[kind]
rd = RegionData(root_dir=de.ROOT, region=region, fidelity_dir=de.FID[res])
rng = np.random.default_rng(42)
pos = valid_positions(rd, de.PATCH, 1.0)
perm = rng.permutation(len(pos))
tr_pos = [pos[i] for i in perm[max(1, int(round(len(pos) * 0.25))):]]
pool = de.scan_wet_windows(rd, tr_pos, start_indices_for(rd, 80, de.T_IN, de.HORIZON, de.STRIDE))
CACHE = os.path.join(os.environ.get("DEBUG_CACHE", "."), f"_cache_{kind}_{NW}.npz")   # keep the (0.5 GB) cache out of the project
if os.path.exists(CACHE):
    z = np.load(CACHE)
    tr = [(z["x"][i], z["y"][i], np.ones((128, 128), bool)) for i in range(len(z["x"]))]
else:
    tr = de.load_examples(rd, [pool[i] for i in rng.choice(len(pool), size=NW, replace=False)])
    np.savez(CACHE, x=np.stack([t[0] for t in tr]), y=np.stack([t[1] for t in tr]))
print(f"{region} {res} m: {len(tr)} windows; input maxima  depth {max(x[..., 0].max() for x, _, _ in tr):.2f} m  rain {max(x[..., 2].max() for x, _, _ in tr):.1f} mm  "
      f"DEM range {min(x[..., 1].min() for x, _, _ in tr):.0f}..{max(x[..., 1].max() for x, _, _ in tr):.0f}")

tf.keras.utils.set_random_seed(42)
enc = fm.build_encoder(hidden_dim=128, dropout=0.3)
dec = fm.build_decoder_step(hidden_dim=128, dropout=0.3)
pers = float(np.mean([np.mean((y - x[-1, ..., 0][None]) ** 2) for x, y, _ in tr]))
sched = de.WarmupCosine(LR, 40 * 400, WARM) if WARM > 0 else tf.keras.optimizers.schedules.CosineDecay(LR, decay_steps=40 * 400, alpha=0.02)
opt = fm.OPTIMIZERS["adam"](learning_rate=sched, clipnorm=CLIP)
vars_ = enc.trainable_variables + dec.trainable_variables
probe = tf.keras.Model(enc.input, [enc.get_layer(n).output for n in ("cnn_encoder_1", "cnn_encoder_2", "cnn_encoder_3", "dense_bottleneck")])
sample = tf.constant(np.stack([x for x, _, _ in tr[:12]]), tf.float32)


def snapshot(step, loss, gnorm):
    acts = probe(sample, training=False)
    act = [float(tf.reduce_mean(tf.cast(a > 0, tf.float32))) for a in acts]
    eo = enc(sample, training=False)
    fr = decode_horizon(dec, eo, sample, de.HORIZON, training=False).numpy()      # (12, H, steps...)
    delta = fr - sample.numpy()[:, -1, ..., 0][:, None]
    per_win = delta.mean(axis=(1, 2, 3))
    print(f"step {step:4d} loss {loss:8.3f} gnorm(pre-clip) {gnorm:9.3f} | active ReLU frac enc1/2/3/dense {act[0]:.2f}/{act[1]:.2f}/{act[2]:.2f}/{act[3]:.2f} | "
          f"predicted mean change across 12 windows: mean {per_win.mean() * 1000:+.2f} mm, std {per_win.std() * 1000:.3f} mm")


snapshot(0, float("nan"), float("nan"))
step = 0
for ep in range(NEP):
    for i in np.random.permutation(len(tr)):
        x, y, _ = tr[i]
        with tf.GradientTape() as tape:
            frames = decode_horizon(dec, enc(tf.constant(x[None], tf.float32), training=True), tf.constant(x[None], tf.float32), de.HORIZON, training=True)
            loss = tf.reduce_mean(tf.square(frames[0] - tf.constant(y, tf.float32))) / pers
        g = tape.gradient(loss, vars_)
        gn = float(tf.linalg.global_norm([gi for gi in g if gi is not None]))
        opt.apply_gradients((gi, v) for gi, v in zip(g, vars_) if gi is not None)
        step += 1
        if step % 25 == 0:
            snapshot(step, float(loss), gn)
