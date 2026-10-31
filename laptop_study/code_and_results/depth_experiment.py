"""Depth-forecast experiments on the COARSE parts of FloodCastBench, following the paper's own two transfer protocols:

  480 m  low-fidelity :  train on Pakistan, test on Mozambique      (the paper's low-fidelity transfer test)
   60 m  high-fidelity:  train on Australia, test on the UK          (the paper's high-fidelity transfer test)

Depth only. At 480 m and 60 m the severity label is degenerate (a 128-px patch covers 61 km / 7.7 km, so nearly every patch is
"severe"), so no severity is scored here. Same model as the 30 m experiment (persistence-anchored CNN-LSTM), same loss
(MSE / persistence MSE), same metric (skill vs persistence, = 1 - MSE_model / MSE_persistence), so results are comparable
in kind. NOT comparable to the paper's tables: those forecast 19 steps at 300 s from the true depth at t = 1 over the whole domain and report a relative error (not metres);
plain persistence under that protocol is comparable to the paper's best model (completion/f_paper_protocol.py).

Also reports, for each trained model, (a) skill by lead time and (b) what the model uses (test-time ablations), because the
30 m audit of the ORIGINAL model (v1) found its skill was driven by rainfall and not by the depth history (a limitation of v1, not of the flood; see PROJECT_LOG 'Completion, round 2').

Run:  python depth_experiment.py pakistan_mozambique        |   python depth_experiment.py australia60_uk60
"""
import json
import os
import sys
import time

import numpy as np
import rasterio
import tensorflow as tf

import model as flood_model
from model import decode_horizon
from real_data_loader import RegionData
from windows import valid_positions, start_indices_for

PATCH, T_IN, HORIZON, STRIDE = 128, 5, 5, 6          # 6 x 5 min = 30 min: rainfall's native step, same as the 30 m experiment
ROOT = "data/extracted/FloodCastBench"
FID = {30: "High-fidelity flood forecasting/30m", 60: "High-fidelity flood forecasting/60m", 480: "Low-fidelity flood forecasting/480m"}


# ------------------------------------------------------------------------------------------------- windows
def scan_wet_windows(rd, positions, starts, patch=PATCH, wet_m=0.05):
    """(row, col, start) for windows whose patch is wet (>= wet_m) at the last observed OR the last forecast frame.
    Dry windows carry no signal (persistence is exact) and would only dilute the metrics."""
    out = []
    for st in starts:
        hist, fut = rd.window_frame_indices(int(st), T_IN, HORIZON, STRIDE)
        with rasterio.open(rd.depth_files[hist[-1]]) as s:
            last = s.read(1)
        with rasterio.open(rd.depth_files[fut[-1]]) as s:
            end = s.read(1)
        for (r, c) in positions:
            sl = (slice(r, r + patch), slice(c, c + patch))
            if max(float(last[sl].max()), float(end[sl].max())) >= wet_m:
                out.append((r, c, int(st)))
    return out


def load_examples(rd, windows, patch=PATCH):
    ex = []
    for (r, c, st) in windows:
        x, y, _, _ = rd.load_patch_sequence(r, c, patch, st, T_IN, HORIZON, STRIDE)
        ex.append((x, y, np.ones((patch, patch), bool)))
    return ex


def load_padded_examples(rd, positions, starts, size, wet_m=0.05):
    """For a domain smaller than one patch in some direction (UK at 60 m is 85x137): read the largest square that fits, pad it
    to 128x128 (depth 0; terrain and rain edge-replicated) and return a mask of the real pixels, so scoring ignores the padding."""
    ex = []
    for st in starts:
        for (r, c) in positions:
            x, y, _, mx = rd.load_patch_sequence(r, c, size, int(st), T_IN, HORIZON, STRIDE)
            if max(float(x[-1, ..., 0].max()), float(y[-1].max())) < wet_m:
                continue
            p = PATCH - size
            xp = np.pad(x, ((0, 0), (0, p), (0, p), (0, 0)), mode="edge")
            xp[:, size:, :, 0] = 0.0
            xp[:, :, size:, 0] = 0.0
            yp = np.pad(y, ((0, 0), (0, p), (0, p)))
            mask = np.zeros((PATCH, PATCH), bool)
            mask[:size, :size] = True
            ex.append((xp, yp, mask))
    return ex


# ------------------------------------------------------------------------------------------------- model
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up to `peak` over `warmup_steps`, then cosine decay to alpha*peak. The 60 m Australia run collapsed to an
    input-independent network within ~100 steps at a constant-start LR of 1e-3 (Adam + a few extreme windows); warm-up removes that."""

    def __init__(self, peak, total_steps, warmup_steps, alpha=0.02):
        self.peak, self.total, self.warm, self.alpha = float(peak), float(total_steps), float(max(1, warmup_steps)), float(alpha)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.peak * (step + 1.0) / self.warm
        prog = tf.clip_by_value((step - self.warm) / tf.maximum(1.0, self.total - self.warm), 0.0, 1.0)
        cos = self.peak * (self.alpha + (1.0 - self.alpha) * 0.5 * (1.0 + tf.cos(np.pi * prog)))
        return tf.where(step < self.warm, warm, cos)

    def get_config(self):
        return dict(peak=self.peak, total_steps=self.total, warmup_steps=self.warm, alpha=self.alpha)


def prediction_spread(predict, examples, n=12):
    """Std across windows of the predicted mean change (metres). ~0 means the network answers the same for every input (dead)."""
    d = [(predict(x) - x[-1, ..., 0][None]).mean() for x, _, _ in examples[:n]]
    return float(np.std(d))


def build_and_train(train, val, epochs, seed, lr=1e-3, clipnorm=1.0, warmup_steps=200, log=print):
    tf.keras.utils.set_random_seed(seed)
    encoder = flood_model.build_encoder(hidden_dim=128, dropout=0.3)
    decoder = flood_model.build_decoder_step(hidden_dim=128, dropout=0.3)
    pers_mse = max(float(np.mean([np.mean((y - x[-1, ..., 0][None]) ** 2) for x, y, _ in train])), 1e-12)
    sched = WarmupCosine(lr, epochs * len(train), warmup_steps)
    opt = flood_model.OPTIMIZERS["adam"](learning_rate=sched, clipnorm=clipnorm)
    vars_ = encoder.trainable_variables + decoder.trainable_variables
    xs = tf.TensorSpec([1, T_IN, PATCH, PATCH, 3], tf.float32)
    ys = tf.TensorSpec([HORIZON, PATCH, PATCH], tf.float32)

    @tf.function(input_signature=[xs, ys])
    def step(x, y):
        with tf.GradientTape() as tape:
            frames = decode_horizon(decoder, encoder(x, training=True), x, HORIZON, training=True)
            loss = tf.reduce_mean(tf.square(frames[0] - y)) / pers_mse
        g = tape.gradient(loss, vars_)
        opt.apply_gradients((gi, v) for gi, v in zip(g, vars_) if gi is not None)
        return loss

    @tf.function(input_signature=[xs])
    def pred(x):
        return decode_horizon(decoder, encoder(x, training=False), x, HORIZON, training=False)[0]

    def predict(x):
        return pred(tf.constant(x[None], tf.float32)).numpy()

    def val_skill():
        return skill(all_preds(predict, val))

    log(f"persistence MSE on training windows {pers_mse:.3e} m^2 (RMSE {np.sqrt(pers_mse) * 1000:.1f} mm); "
        f"epoch 0 val skill {val_skill():+.4f} (must be ~0)")
    hist = []
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        losses = []
        for i in np.random.permutation(len(train)):
            x, y, _ = train[i]
            losses.append(float(step(tf.constant(x[None], tf.float32), tf.constant(y, tf.float32))))
        sk = val_skill()
        hist.append((ep, float(np.mean(losses)), sk))
        spread = prediction_spread(predict, val)
        if ep >= 2 and spread < 1e-5:
            raise RuntimeError(f"DEAD NETWORK after epoch {ep}: the predicted change is identical for every validation window "
                               f"(std {spread:.2e} m). Its skill would be an input-independent constant offset, not a result.")
        log(f"epoch {ep:2d}/{epochs}  train/persist {np.mean(losses):.4f}  val skill {sk:+.4f}  ({time.perf_counter() - t0:.1f}s)")
    return predict, hist, pers_mse


# ------------------------------------------------------------------------------------------------- evaluation
def all_preds(predict, examples, transform=None):
    P, O, Q, M = [], [], [], []
    for x, y, m in examples:
        P.append(predict(x if transform is None else transform(x)))
        O.append(y)
        Q.append(np.repeat(x[-1, ..., 0][None], HORIZON, axis=0))
        M.append(np.broadcast_to(m, (HORIZON,) + m.shape))
    return np.stack(P), np.stack(O), np.stack(Q), np.stack(M)


def skill(t):
    p, o, q, m = t
    return 1.0 - np.mean(((p - o) ** 2)[m]) / np.mean(((q - o) ** 2)[m])


def metrics(p, o, m, wet):
    pp, oo = p[m], o[m]
    rmse = float(np.sqrt(np.mean((pp - oo) ** 2)))
    nse = float(1 - np.sum((oo - pp) ** 2) / np.sum((oo - oo.mean()) ** 2))
    r = float(np.corrcoef(pp, oo)[0, 1])
    pw, ow = pp > wet, oo > wet
    csi = float((pw & ow).sum() / max(1, (pw | ow).sum()))
    return dict(RMSE=rmse, NSE=nse, r=r, CSI=csi)


def report(name, predict, examples, log=print):
    p, o, q, m = all_preds(predict, examples)
    row = {"n_windows": len(examples), "skill": float(skill((p, o, q, m)))}
    row["model"] = {f"CSI@{w}": metrics(p, o, m, w) for w in (0.001, 0.1)}
    row["persistence"] = {f"CSI@{w}": metrics(q, o, m, w) for w in (0.001, 0.1)}
    lead = [float(1 - np.mean(((p[:, k] - o[:, k]) ** 2)[m[:, k]]) / np.mean(((q[:, k] - o[:, k]) ** 2)[m[:, k]])) for k in range(HORIZON)]
    pr = [float(np.sqrt(np.mean(((q[:, k] - o[:, k]) ** 2)[m[:, k]])) * 1000) for k in range(HORIZON)]
    row["lead_skill"], row["persistence_rmse_mm"] = lead, pr
    a, b = row["model"]["CSI@0.001"], row["persistence"]["CSI@0.001"]
    log(f"\n{name}: {len(examples)} windows")
    log(f"  skill vs persistence {row['skill']:+.4f}")
    log(f"  RMSE  model {a['RMSE']:.4f} m   persistence {b['RMSE']:.4f} m")
    log(f"  NSE   model {a['NSE']:.4f}     persistence {b['NSE']:.4f}      r  model {a['r']:.4f}  persistence {b['r']:.4f}")
    log(f"  CSI(0.001 m) model {a['CSI']:.4f} persistence {b['CSI']:.4f};  CSI(0.1 m) model {row['model']['CSI@0.1']['CSI']:.4f} "
        f"persistence {row['persistence']['CSI@0.1']['CSI']:.4f}")
    log("  skill by lead time (30 min ... 2.5 h): " + "  ".join(f"{v:+.3f}" for v in lead) +
        "   | persistence RMSE mm: " + " ".join(f"{v:.1f}" for v in pr))
    return row


def ablations(name, predict, examples, log=print):
    def no_rain(x):
        y = x.copy(); y[..., 2] = 0.0; return y

    def flat_dem(x):
        y = x.copy(); y[..., 1] = 0.0; return y

    def scrambled(x):
        return np.concatenate([x[:-1][::-1], x[-1:]], axis=0)

    def frozen(x):
        return np.repeat(x[-1:], x.shape[0], axis=0)

    out = {}
    log(f"\n{name}: what does the model use? (skill when one input is removed at test time)")
    for label, fn in (("none (normal)", None), ("rainfall removed", no_rain), ("terrain removed", flat_dem),
                      ("history order scrambled", scrambled), ("history frozen", frozen)):
        out[label] = float(skill(all_preds(predict, examples, fn)))
        log(f"    {label:26s} {out[label]:+.3f}")
    return out


# ------------------------------------------------------------------------------------------------- the two experiments
def experiment(kind, epochs=40, n_train=400, n_val=150, n_starts=80, seed=42, log=print):
    rng = np.random.default_rng(seed)
    if kind == "pakistan_mozambique":
        res_train, res_test, tr_name, te_name = 480, 480, "Pakistan", "Mozambique"
    elif kind == "australia60_uk60":
        res_train, res_test, tr_name, te_name = 60, 60, "Australia", "UK"
    else:
        raise SystemExit(f"unknown experiment {kind}")
    rd_tr = RegionData(root_dir=ROOT, region=tr_name, fidelity_dir=FID[res_train])
    rd_te = RegionData(root_dir=ROOT, region=te_name, fidelity_dir=FID[res_test])
    log(f"=== {tr_name} ({res_train} m, depth {rd_tr.grid_height}x{rd_tr.grid_width}) -> {te_name} ({res_test} m, depth "
        f"{rd_te.grid_height}x{rd_te.grid_width}); epochs {epochs}, seed {seed}")
    pos = valid_positions(rd_tr, PATCH, 1.0)
    perm = rng.permutation(len(pos))
    nv = max(1, int(round(len(pos) * 0.25)))
    val_pos = [pos[i] for i in perm[:nv]]
    tr_pos = [pos[i] for i in perm[nv:]]
    assert not set(val_pos) & set(tr_pos)
    log(f"{tr_name}: {len(pos)} valid {PATCH}px tiles -> {len(tr_pos)} train / {len(val_pos)} validation (split by tile)")
    starts = start_indices_for(rd_tr, n_starts, T_IN, HORIZON, STRIDE)
    tr_pool = scan_wet_windows(rd_tr, tr_pos, starts)
    va_pool = scan_wet_windows(rd_tr, val_pos, starts)
    log(f"wet windows available: train {len(tr_pool)}, validation {len(va_pool)}")
    tr_w = [tr_pool[i] for i in rng.choice(len(tr_pool), size=min(n_train, len(tr_pool)), replace=False)]
    va_w = [va_pool[i] for i in rng.choice(len(va_pool), size=min(n_val, len(va_pool)), replace=False)]
    train, val = load_examples(rd_tr, tr_w), load_examples(rd_tr, va_w)
    log(f"loaded {len(train)} training and {len(val)} validation windows")

    # test region
    te_starts = start_indices_for(rd_te, 60, T_IN, HORIZON, STRIDE)
    small = min(rd_te.grid_height, rd_te.grid_width) < PATCH
    if small:                                        # the domain is smaller than a patch in one direction: padded + masked
        size = min(rd_te.grid_height, rd_te.grid_width)
        cols = sorted({0, rd_te.grid_width - size})
        rows = sorted({0, rd_te.grid_height - size})
        test = load_padded_examples(rd_te, [(r, c) for r in rows for c in cols], te_starts, size)
        log(f"{te_name}: domain {rd_te.grid_height}x{rd_te.grid_width} < one patch -> {size}px windows padded to {PATCH}, scored on real pixels only")
    else:
        rows = sorted({0, rd_te.grid_height - PATCH})
        cols = sorted({0, rd_te.grid_width - PATCH})
        tw = scan_wet_windows(rd_te, [(r, c) for r in rows for c in cols], te_starts)
        test = load_examples(rd_te, tw)
        log(f"{te_name}: {len(rows) * len(cols)} (overlapping) {PATCH}px tiles x {len(te_starts)} times -> {len(test)} wet windows")

    predict, hist, pers = build_and_train(train, val, epochs, seed, log=log)
    out = {"experiment": kind, "seed": seed, "epochs": epochs, "n_train": len(train), "n_val": len(val), "n_test": len(test),
           "history": hist, "persistence_mse_train": pers}
    out["validation"] = report(f"{tr_name} held-out tiles", predict, val, log)
    out["test"] = report(f"{te_name} (never seen)", predict, test, log)
    out["ablation_validation"] = ablations(f"{tr_name} held-out tiles", predict, val, log)
    out["ablation_test"] = ablations(f"{te_name} (never seen)", predict, test, log)
    return out


if __name__ == "__main__":
    kind = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    n_train = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    res = experiment(kind, epochs=epochs, seed=seed, n_train=n_train, n_val=min(150, n_train // 2), n_starts=80 if n_train >= 100 else 20)
    os.makedirs("verification_outputs/multires", exist_ok=True)
    path = f"verification_outputs/multires/{kind}_seed{seed}_ep{epochs}.json"
    json.dump(res, open(path, "w"), indent=1, default=float)
    print("\nsaved", path)
