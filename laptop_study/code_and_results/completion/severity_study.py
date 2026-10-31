"""Severity branch, symbolic rule layer and Grad-CAM, re-evaluated on the frozen depth architecture (clstm) with tile folds.

Joint model = the shared encoder (a full-resolution ConvLSTM, `model_v2.build_clstm`) + the depth head (persistence-anchored) + a severity head on the
encoder's hidden map: three stride-2 convolutions down to 16x16x64, then EITHER the capsule head (PrimaryCaps -> routing-by-agreement -> class capsules,
margin loss; the project's design) OR a plain dense head (Flatten -> Dense(64) -> Dense(4), cross-entropy; the control).  Severity label = max depth in the
patch over the forecast horizon, binned 0.3 / 1 / 2 m (as before); Australia 30 m only (the label is degenerate elsewhere).

Reports on held-out TILES (fold k of 3): accuracy on class-balanced (60 per class) and natural-distribution windows, macro recall, the "class-changing"
subset (windows whose label differs from the last frame's class, where the threshold baseline is wrong by construction), against
  B0  threshold the LAST OBSERVED frame,
  B1  threshold the model's OWN PREDICTED depth (max over the forecast frames),
and the symbolic rule layer (fixed original thresholds vs thresholds re-derived on the training windows of the fold): windows changed / fixed / broken,
plus whether the rule reasons are TRUE statements (sign agreement of the stated depth trend with the actual one).  Grad-CAM: wet-vs-dry heat ratio,
a deletion (faithfulness) test against random masking, and saved maps for stability across seeds.

Run: python completion/severity_study.py <regime hybrid|forced> <head capsule|dense> <fold> [--seed S] [--epochs 30] [--per-class 100]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402  (chdirs to the project root)

T_IN, HORIZON = E.T_IN, E.HORIZON
CLASSES = ["none", "minor", "moderate", "severe"]
OUT = "completion/results/severity"


def sev_labels(pool):
    from real_data_loader import depth_to_severity
    p = os.path.join(E.CACHE, f"sev_labels_{pool['region']}_{pool['res']}.npz")
    if os.path.exists(p):
        d = np.load(p)
        return d["lab"], d["base"]
    n = len(pool["meta"])
    lab = np.array([depth_to_severity(float(np.max(pool["df"][i]))) for i in range(n)])
    base = np.array([depth_to_severity(float(np.max(pool["dh"][i][-1]))) for i in range(n)])
    np.savez(p, lab=lab, base=base)
    return lab, base


def balanced(idx, lab, per_class, rng):
    out = []
    for c in range(4):
        m = [i for i in idx if lab[i] == c]
        out += list(rng.choice(m, size=min(per_class, len(m)), replace=False)) if m else []
    return sorted(int(i) for i in out)


def build_joint(T, head):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import model as m1
    import model_v2 as m2
    enc, dec = m2.build_clstm(T, HORIZON)
    h_in = keras.Input(shape=(128, 128, 8))
    z = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(h_in)
    z = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(z)
    z = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(z)
    feat = keras.Model(h_in, z, name="severity_features")
    if head == "capsule":
        sev = m1.build_capsule_head(return_lengths=True)
    else:
        i = keras.Input(shape=(16, 16, 64))
        o = layers.Dense(4, name="dense_logits")(layers.Dense(64, activation="relu")(layers.Dropout(0.3)(layers.Flatten()(i))))
        sev = keras.Model(i, o, name="dense_severity_head")
    return enc, dec, feat, sev


def rule_inputs(pool, rd_lu, i, x, pred, hours):
    """Depth rate (predicted), rainfall (last observed step), slope, drainage proximity of one window."""
    from predict_severity import _compute_slope_pct, _compute_drainage_proximity_m
    r, c, _ = (int(v) for v in pool["meta"][i])
    last = x[T_IN - 1, ..., 0]
    rate = float((pred[-1].mean() - last.mean()) / hours)
    rain = float(x[T_IN - 1, ..., 2].mean())
    slope = _compute_slope_pct(np.array(pool["dem"][r:r + 128, c:c + 128]), 30.0)
    lu = rd_lu.load_landuse_patch(r, c, 128)
    dr = _compute_drainage_proximity_m(lu, 30.0)
    return rate, rain, slope, (0.0 if np.isnan(dr) else dr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regime"); ap.add_argument("head"); ap.add_argument("fold", type=int)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--augment", action="store_true")
    a = ap.parse_args()
    seed = 2000 + 10 * a.fold + a.K if a.seed is None else a.seed
    rid = f"SEV_{a.regime}_{a.head}{'_aug' if a.augment else ''}_f{a.fold}of{a.K}_s{seed}_ep{a.epochs}"
    os.makedirs(OUT, exist_ok=True)
    out_p = os.path.join(OUT, rid + ".json")
    if os.path.exists(out_p):
        print(rid, "already done"); return
    tf = E._tf()
    from depth_experiment import WarmupCosine
    from capsule_layers import margin_loss
    from real_data_loader import depth_to_severity
    from symbolic_rule_layer import apply_symbolic_rules
    import symbolic_rule_layer as SRL
    log = lambda s: print(s, flush=True)
    pool = E.build_pool("Australia", 30, "train", log)
    lab, base = sev_labels(pool)
    tiles = sorted({(int(r), int(c)) for r, c, _ in pool["meta"]})
    held = set(map(tuple, E.folds_of_tiles(tiles, a.K, seed=0)[a.fold]))
    tile_of = [(int(r), int(c)) for r, c, _ in pool["meta"]]
    tr_all = [i for i, t in enumerate(tile_of) if t not in held]
    va_all = [i for i, t in enumerate(tile_of) if t in held]
    rng = np.random.default_rng(seed)
    tr_idx = balanced(tr_all, lab, a.per_class, rng)
    va_bal = balanced(va_all, lab, 60, rng)
    va_nat = sorted(int(i) for i in rng.choice(va_all, size=min(600, len(va_all)), replace=False))
    log(f"{rid}: {len(tiles)} tiles, held-out {len(held)}; train {len(tr_idx)} balanced windows (class counts {np.bincount(lab[tr_idx], minlength=4).tolist()}); "
        f"held-out balanced {len(va_bal)}, natural {len(va_nat)} (class fractions {(np.bincount(lab[va_nat], minlength=4) / len(va_nat)).round(3).tolist()}); "
        f"last-frame baseline on natural {float(np.mean(base[va_nat] == lab[va_nat])):.3f}")

    T = T_IN + (HORIZON if a.regime == "forced" else 0)
    tf.keras.utils.set_random_seed(seed)
    enc, dec, feat, sev = build_joint(T, a.head)
    vars_ = enc.trainable_variables + dec.trainable_variables + feat.trainable_variables + sev.trainable_variables
    pers = max(float(np.mean([np.mean((np.array(pool["df"][i]) - np.array(pool["dh"][i][-1])[None]) ** 2) for i in tr_idx])), 1e-12)
    opt = tf.keras.optimizers.Adam(learning_rate=WarmupCosine(1e-3, a.epochs * len(tr_idx), 200), clipnorm=1.0)
    import model_v2 as m2
    xs = tf.TensorSpec([1, T, 128, 128, 3], tf.float32); ys = tf.TensorSpec([HORIZON, 128, 128], tf.float32); ss = tf.TensorSpec([1], tf.int32)

    def sev_out(h, training):
        return sev(feat(h, training=training), training=training)

    @tf.function(input_signature=[xs, ys, ss])
    def step(x, y, s):
        with tf.GradientTape() as tape:
            h = enc(x, training=True)
            frames = m2.decode_horizon_local(dec, h, x, HORIZON, training=True)
            dl = tf.reduce_mean(tf.square(frames[0] - y)) / pers
            o = sev_out(h if not isinstance(h, (list, tuple)) else h[0], True)
            sl = margin_loss(s, o) if a.head == "capsule" else tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(s, o, from_logits=True))
            loss = dl + sl
        g = tape.gradient(loss, vars_)
        opt.apply_gradients((gi, v) for gi, v in zip(g, vars_) if gi is not None)
        return dl, sl

    @tf.function(input_signature=[xs])
    def pred(x):
        h = enc(x, training=False)
        hh = h if not isinstance(h, (list, tuple)) else h[0]
        frames = m2.decode_horizon_local(dec, h, x, HORIZON, training=False)
        return frames[0], tf.argmax(sev_out(hh, False)[0])

    def predict(x):
        f, s = pred(tf.constant(x[None], tf.float32))
        return f.numpy(), int(s)

    val_small = va_bal[::max(1, len(va_bal) // 48)][:48]
    cache_small = [E.compose(pool, i, a.regime)[0] for i in val_small]

    def quick():
        return float(np.mean([predict(x)[1] == lab[i] for x, i in zip(cache_small, val_small)]))

    t0 = time.perf_counter()
    for ep in range(1, a.epochs + 1):
        dls, sls = [], []
        for k in rng.permutation(len(tr_idx)):
            x, y, _ = E.compose(pool, tr_idx[k], a.regime)
            if a.augment:                                  # dihedral transform, identical for inputs and targets; the severity label is unchanged
                q, fl = int(rng.integers(4)), bool(rng.integers(2))
                x, y = np.rot90(x, q, axes=(1, 2)), np.rot90(y, q, axes=(1, 2))
                if fl:
                    x, y = x[:, :, ::-1], y[:, :, ::-1]
                x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
            dl, sl = step(tf.constant(x[None], tf.float32), tf.constant(y, tf.float32), tf.constant([int(lab[tr_idx[k]])], tf.int32))
            dls.append(float(dl)); sls.append(float(sl))
        log(f"epoch {ep:2d}/{a.epochs}  depth/persist {np.mean(dls):.3f}  sev loss {np.mean(sls):.3f}  val acc(48 balanced) {quick():.3f}  ({(time.perf_counter() - t0) / ep:.0f}s/epoch avg)")

    # ------------------------------------------------------------------ evaluation
    def evaluate_set(idx):
        preds, cls, pcls, X, P = [], [], [], [], []
        for i in idx:
            x, y, _ = E.compose(pool, i, a.regime)
            f, s = predict(x)
            cls.append(s)
            pcls.append(depth_to_severity(float(f.max())))
            X.append(x); P.append(f)
        return np.array(cls), np.array(pcls), X, P

    res = {"id": rid, "regime": a.regime, "head": a.head, "augment": a.augment, "fold": a.fold, "K": a.K, "seed": seed, "epochs": a.epochs, "held_out_tiles": sorted(held)}
    ev = {}
    for name, idx in (("balanced", va_bal), ("natural", va_nat)):
        cls, pcls, X, P = evaluate_set(idx)
        y_true, y_base = lab[idx], base[idx]
        chg = y_true != y_base
        conf = np.zeros((4, 4), int)
        for t, p in zip(y_true, cls):
            conf[t, p] += 1
        recall = [float(conf[c, c] / max(1, conf[c].sum())) for c in range(4)]
        ev[name] = dict(n=len(idx), acc=float(np.mean(cls == y_true)), baseline_last_frame=float(np.mean(y_base == y_true)),
                        forecast_threshold=float(np.mean(pcls == y_true)), macro_recall=float(np.mean(recall)), recall=recall, confusion=conf.tolist(),
                        n_changing=int(chg.sum()), acc_changing=float(np.mean(cls[chg] == y_true[chg])) if chg.any() else None,
                        forecast_threshold_changing=float(np.mean(pcls[chg] == y_true[chg])) if chg.any() else None,
                        acc_unchanged=float(np.mean(cls[~chg] == y_true[~chg])) if (~chg).any() else None,
                        per_window=dict(idx=[int(i) for i in idx], cls=cls.tolist(), pcls=pcls.tolist(), true=y_true.tolist(), base=y_base.tolist()))
        log(f"{name}: capsule/{a.head} acc {ev[name]['acc']:.3f} | last-frame threshold {ev[name]['baseline_last_frame']:.3f} | forecast threshold {ev[name]['forecast_threshold']:.3f} | "
            f"macro recall {ev[name]['macro_recall']:.3f} | changing n={ev[name]['n_changing']} acc {ev[name]['acc_changing']}")
        if name == "natural":
            nat_X, nat_P, nat_cls, nat_idx = X, P, cls, idx
    res["eval"] = ev

    # ------------------------------------------------------------------ symbolic rule layer (natural windows)
    from real_data_loader import RegionData
    rd_lu = RegionData(root_dir=E.ROOT, region="Australia", fidelity_dir=E.FID[30])
    hours = HORIZON * 6 * 300 / 3600.0
    feats = np.array([rule_inputs(pool, rd_lu, i, x, p, hours) for i, x, p in zip(nat_idx, nat_X, nat_P)])
    true_rate = np.array([(np.array(pool["df"][i])[-1].mean() - np.array(pool["dh"][i])[-1].mean()) / hours for i in nat_idx])
    # thresholds re-derived on the fold's training windows (true future depth): midpoint of the moderate and severe class means of each feature
    trf = []
    for i in tr_idx:
        r, c, _ = (int(v) for v in pool["meta"][i])
        x = E.compose(pool, i, a.regime)[0]
        rate = float((np.array(pool["df"][i])[-1].mean() - np.array(pool["dh"][i])[-1].mean()) / hours)
        from predict_severity import _compute_slope_pct, _compute_drainage_proximity_m
        dr = _compute_drainage_proximity_m(rd_lu.load_landuse_patch(r, c, 128), 30.0)
        trf.append((rate, float(x[T_IN - 1, ..., 2].mean()), _compute_slope_pct(np.array(pool["dem"][r:r + 128, c:c + 128]), 30.0), 0.0 if np.isnan(dr) else dr))
    trf = np.array(trf); trl = lab[tr_idx]
    mid = lambda col: float((trf[trl == 2, col].mean() + trf[trl == 3, col].mean()) / 2) if (trl == 2).any() and (trl == 3).any() else float("nan")
    recal = dict(rate=mid(0), rain=mid(1), slope=mid(2), drain=mid(3))
    orig = dict(rate=SRL.DEPTH_RATE_ESCALATE_THRESHOLD, rain=SRL.RAINFALL_CORROBORATE_THRESHOLD, slope=SRL.SLOPE_HIGH_ENERGY_THRESHOLD, drain=SRL.DRAINAGE_FLOODPLAIN_THRESHOLD)
    rules = {}
    for nm, th in (("original_thresholds", orig), ("recalibrated_on_fold_training_windows", recal)):
        SRL.DEPTH_RATE_ESCALATE_THRESHOLD, SRL.RAINFALL_CORROBORATE_THRESHOLD = th["rate"], th["rain"]
        SRL.SLOPE_HIGH_ENERGY_THRESHOLD, SRL.DRAINAGE_FLOODPLAIN_THRESHOLD = th["slope"], th["drain"]
        newc, trig = [], []
        for c0, (rate, rain, slope, dr) in zip(nat_cls, feats):
            c1, reasons = apply_symbolic_rules(int(c0), rate, rain, slope, dr)
            newc.append(c1); trig.append(not reasons[0].startswith("no rule"))
        newc = np.array(newc); yt = lab[nat_idx]
        changed = newc != nat_cls
        rules[nm] = dict(thresholds=th, acc_before=float(np.mean(nat_cls == yt)), acc_after=float(np.mean(newc == yt)), n_changed=int(changed.sum()),
                         n_fixed=int(np.sum(changed & (newc == yt))), n_broke=int(np.sum(changed & (nat_cls == yt))), n_triggered=int(np.sum(trig)))
        log(f"rules [{nm}]: acc {rules[nm]['acc_before']:.3f} -> {rules[nm]['acc_after']:.3f}; changed {rules[nm]['n_changed']} (fixed {rules[nm]['n_fixed']}, broke {rules[nm]['n_broke']})")
    SRL.DEPTH_RATE_ESCALATE_THRESHOLD, SRL.RAINFALL_CORROBORATE_THRESHOLD = orig["rate"], orig["rain"]
    SRL.SLOPE_HIGH_ENERGY_THRESHOLD, SRL.DRAINAGE_FLOODPLAIN_THRESHOLD = orig["slope"], orig["drain"]
    pr = feats[:, 0]
    rules["reason_truth"] = dict(sign_agreement_rate_predicted_vs_true_depth_trend=float(np.mean(np.sign(pr) == np.sign(true_rate))),
                                 corr_rate=float(np.corrcoef(pr, true_rate)[0, 1]) if pr.std() > 0 and true_rate.std() > 0 else None,
                                 n_rising_claims=int(np.sum(pr > 0)), rising_claims_true=float(np.mean(true_rate[pr > 0] > 0)) if np.any(pr > 0) else None,
                                 n_receding_claims=int(np.sum(pr < -1e-4)), receding_claims_true=float(np.mean(true_rate[pr < -1e-4] < 0)) if np.any(pr < -1e-4) else None)
    res["rules"] = rules

    # ------------------------------------------------------------------ Grad-CAM on the encoder's hidden map (full resolution), redesigned after code review F5
    # Two maps per window, both at the shared encoder's hidden map h (128x128x8):
    #   "depth"    : target = signed sum of the predicted depth over ALL leads (direction = sign of the predicted net change);
    #   "severity" : target = the score of the predicted severity class (capsule length / dense logit) - what "Grad-CAM of the severity head" means.
    # Deletion (faithfulness): remove the depth + rain history at the top-10% heat pixels (only pixels with heat > 0, so ties at zero cannot enter) and measure
    #   depth: the L1 change of the predicted CHANGE maps (forecast minus its own anchor; all leads, all pixels);  severity: the change of the predicted class's score,
    # against two controls of the same size: (a) random pixels with the SAME wet/dry mix; (b) the top mask moved 40 pixels (same shape, another place).
    @tf.function(input_signature=[xs])
    def sev_score(x):
        h = enc(x, training=False)
        return sev_out(h if not isinstance(h, (list, tuple)) else h[0], False)[0]

    def cam_of(x, kind):
        xb = tf.constant(x[None], tf.float32)
        with tf.GradientTape() as tape:
            h = enc(xb, training=False)
            hh = h[0] if isinstance(h, (list, tuple)) else h
            tape.watch(hh)
            if kind == "depth":
                frames = m2.decode_horizon_local(dec, [hh], xb, HORIZON, training=False)
                net = tf.reduce_sum(frames[0]) - HORIZON * tf.reduce_sum(xb[0, T_IN - 1, ..., 0])
                target = tf.stop_gradient(tf.where(net >= 0, 1.0, -1.0)) * tf.reduce_sum(frames[0])
            else:
                o = sev_out(hh, False)[0]
                target = tf.gather(o, tf.argmax(o))
        g = tape.gradient(target, hh)
        w = tf.reduce_mean(g, axis=(1, 2))
        cam = tf.nn.relu(tf.reduce_sum(hh[0] * w[0], axis=-1))
        return (cam / (tf.reduce_max(cam) + 1e-9)).numpy()

    def change_maps(xx):
        return predict(xx)[0] - xx[T_IN - 1, ..., 0][None]

    cam_idx = sorted(int(i) for i in np.random.default_rng(777).choice(va_all, size=min(40, len(va_all)), replace=False))   # FIXED windows, independent of the seed
    crng = np.random.default_rng(1)
    store = {kd: dict(ratio=[], eff=[], cams=[]) for kd in ("depth", "severity")}
    for i_win in cam_idx:
        x = E.compose(pool, i_win, a.regime)[0]
        wet = x[T_IN - 1, ..., 0] > 0.05
        wet_ids, dry_ids = np.flatnonzero(wet.ravel()), np.flatnonzero(~wet.ravel())
        base_change, base_score = change_maps(x), sev_score(tf.constant(x[None], tf.float32)).numpy()
        cls_pred = int(np.argmax(base_score))
        for kd in ("depth", "severity"):
            cam = cam_of(x, kd)
            store[kd]["cams"].append(cam.astype(np.float16))
            if wet.any() and (~wet).any():
                store[kd]["ratio"].append(float(cam[wet].mean() / max(cam[~wet].mean(), 1e-9)))
            pos = np.flatnonzero(cam.ravel() > 0)
            k = min(int(0.10 * cam.size), len(pos))
            if k < 20:
                continue
            top_flat = pos[np.argsort(cam.ravel()[pos])[-k:]]
            top = np.unravel_index(top_flat, cam.shape)
            n_wet_top = int(wet[top].sum())

            def effect(ids, kd=kd):
                xm = x.copy(); xm[:, ids[0], ids[1], 0] = 0.0; xm[:, ids[0], ids[1], 2] = 0.0
                if kd == "depth":
                    return float(np.abs(change_maps(xm) - base_change).sum())
                return float(abs(sev_score(tf.constant(xm[None], tf.float32)).numpy()[cls_pred] - base_score[cls_pred]))

            def random_matched():
                pick = np.concatenate([crng.choice(wet_ids, min(n_wet_top, len(wet_ids)), replace=False),
                                       crng.choice(dry_ids, min(k - n_wet_top, len(dry_ids)), replace=False)])
                return np.unravel_index(pick, cam.shape)
            e_top, e_rand = effect(top), float(np.mean([effect(random_matched()) for _ in range(3)]))
            e_roll = effect(((top[0] + 40) % 128, (top[1] + 40) % 128))
            store[kd]["eff"].append((e_top, e_rand, e_roll))
    res["gradcam"] = {}
    for kd in ("depth", "severity"):
        eff, rat = np.array(store[kd]["eff"]).reshape(-1, 3), store[kd]["ratio"]
        res["gradcam"][kd] = dict(n_windows=len(cam_idx), n_with_effect=int(len(eff)), frac_windows_wet_hotter=float(np.mean(np.array(rat) > 1.0)) if rat else None,
                                  median_wet_dry_ratio=float(np.median(rat)) if rat else None,
                                  deletion_effect_top=float(eff[:, 0].mean()) if len(eff) else None, deletion_effect_random=float(eff[:, 1].mean()) if len(eff) else None,
                                  deletion_effect_rolled=float(eff[:, 2].mean()) if len(eff) else None,
                                  frac_top_exceeds_random=float(np.mean(eff[:, 0] > eff[:, 1])) if len(eff) else None,
                                  frac_top_exceeds_both=float(np.mean((eff[:, 0] > eff[:, 1]) & (eff[:, 0] > eff[:, 2]))) if len(eff) else None)
        g = res["gradcam"][kd]
        log(f"Grad-CAM [{kd}]: wet hotter in {g['frac_windows_wet_hotter']} of windows (median ratio {g['median_wet_dry_ratio']}); deletion effect top {g['deletion_effect_top']} vs random {g['deletion_effect_random']} vs shifted {g['deletion_effect_rolled']}; top beats both controls in {g['frac_top_exceeds_both']}")
    res["gradcam_windows"] = cam_idx
    np.savez_compressed(os.path.join(OUT, rid + "_cams.npz"), cams_depth=np.stack(store["depth"]["cams"]), cams_sev=np.stack(store["severity"]["cams"]), idx=np.array(cam_idx))
    json.dump(res, open(out_p, "w"), default=float)
    log("saved " + out_p)


if __name__ == "__main__":
    main()
