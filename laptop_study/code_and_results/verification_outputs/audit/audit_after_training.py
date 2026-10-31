"""Behavioural audit of the TRAINED model. Not run on its own: dryrun_notebook.py exec()s it in the notebook's
namespace after the last cell (set DRYRUN_EXTRA=<this file>), so it can use the trained encoder/decoder/capsule head
and the exact validation / UK windows the notebook evaluated.

Questions it answers (each a claim the deck makes or implies):
  1. Is the notebook's headline skill number right?  -> recompute it independently, from raw predictions, in numpy.
  2. Is the metric code correct?                      -> feed it known answers (truth => skill 1, copy => skill 0).
  3. Does an untrained model equal persistence?        -> measured on real windows.
  4. Does the forecast "evolve"?                       -> skill per lead time (30 min ... 2.5 h).
  5. Does the model USE rainfall / terrain / the temporal history?  -> test-time ablations. (A CNN-LSTM claim needs this.)
"""
import numpy as np
import tensorflow as tf

print("\n" + "=" * 78)
print("AUDIT: behavioural checks on the trained model")
print("=" * 78)


def all_predictions(examples, transform=None):
    preds, obs, pers, sev_ok = [], [], [], 0
    for x, y, s in examples:
        xin = x if transform is None else transform(x)
        frames, sev = predict(xin)
        preds.append(frames)
        obs.append(y)
        pers.append(np.repeat(x[-1, ..., 0][None], HORIZON, axis=0))  # persistence always uses the TRUE last frame
        sev_ok += int(sev == s)
    return np.stack(preds), np.stack(obs), np.stack(pers), sev_ok / len(examples)


def skill_from(preds, obs, pers):
    """skill = 1 - MSE_model / MSE_persistence, in plain numpy (independent of the notebook's compute_metrics)."""
    return 1.0 - np.mean((preds - obs) ** 2) / np.mean((pers - obs) ** 2)


# ---------------------------------------------------------------- 1. independent recomputation of the headline number
p, o, q, acc = all_predictions(val_data)
indep = skill_from(p, o, q)
nb = skill(au_model, au_base)
print(f"\n[1] Australia held-out skill: independent numpy recomputation {indep:+.4f}   notebook {nb:+.4f}   "
      f"|diff| {abs(indep - nb):.5f}")
assert abs(indep - nb) < 2e-3, "the notebook's skill differs from an independent recomputation"

# ---------------------------------------------------------------- 2. metric code on known answers
rng_t = np.random.default_rng(0)
obs_t = rng_t.random((4, 5, 16, 16)).astype("float32")
pers_t = np.clip(obs_t + 0.1 * rng_t.standard_normal(obs_t.shape), 0, None).astype("float32")
m_truth, m_pers = compute_metrics(obs_t, obs_t), compute_metrics(pers_t, obs_t)
print(f"[2] metric code on known answers: truth -> skill {skill(m_truth, m_pers):+.3f} (must be +1.000); "
      f"copy vs copy -> skill {skill(m_pers, m_pers):+.3f} (must be 0.000); RMSE(truth)={m_truth['RMSE']:.1e} (must be 0)")
assert abs(skill(m_truth, m_pers) - 1.0) < 1e-6 and abs(skill(m_pers, m_pers)) < 1e-9 and m_truth["RMSE"] == 0

# ---------------------------------------------------------------- 3. untrained model == persistence
enc0 = flood_model.build_encoder(hidden_dim=128, dropout=0.3)
dec0 = flood_model.build_decoder_step(hidden_dim=128, dropout=0.3)
worst = 0.0
for x, _, _ in val_data[:5] + uk_data[:5]:
    xb = tf.constant(x[None], dtype=tf.float32)
    fr = decode_horizon(dec0, enc0(xb, training=False), xb, HORIZON, training=False).numpy()[0]
    worst = max(worst, float(np.abs(fr - x[-1, ..., 0][None]).max()))
print(f"[3] fresh, untrained model vs last observed frame, worst |difference| over 10 real windows: {worst:.2e} m "
      f"(must be ~0)")
assert worst < 1e-5

# ---------------------------------------------------------------- 4. skill per lead time
print("\n[4] skill per lead time (30-minute steps):")
for name, ex in (("Australia held-out", val_data), ("UK (never seen)", uk_data)):
    p, o, q, _ = all_predictions(ex)
    row = []
    for k in range(HORIZON):
        row.append(1.0 - np.mean((p[:, k] - o[:, k]) ** 2) / np.mean((q[:, k] - o[:, k]) ** 2))
    rm = [float(np.sqrt(np.mean((q[:, k] - o[:, k]) ** 2))) * 1000 for k in range(HORIZON)]
    print(f"    {name:20s} skill by step " + "  ".join(f"{v:+.3f}" for v in row) +
          "   | persistence RMSE mm " + " ".join(f"{v:5.1f}" for v in rm))


# ---------------------------------------------------------------- 5. test-time ablations
def no_rain(x):
    y = x.copy(); y[..., 2] = 0.0; return y


def flat_terrain(x):
    y = x.copy(); y[..., 1] = 0.0; return y          # DEM is made patch-relative inside the model -> zero = flat


def scrambled_history(x):
    return np.concatenate([x[:-1][::-1], x[-1:]], axis=0)   # reverse the first 4 frames, keep the anchor (last frame)


def frozen_history(x):
    return np.repeat(x[-1:], x.shape[0], axis=0)      # every history frame = the last one: no temporal information


ablations = [("normal", None), ("rainfall removed", no_rain), ("terrain removed (flat DEM)", flat_terrain),
             ("history order scrambled", scrambled_history), ("history frozen (no dynamics)", frozen_history)]
print("\n[5] does the model use each input?  (skill vs persistence, and severity accuracy, when an input is removed at test time)")
print(f"    {'':32s} {'AU skill':>9s} {'UK skill':>9s} {'severity acc':>13s}")
base = {}
for name, tf_ in ablations:
    pa, oa, qa, acc_a = all_predictions(val_data, tf_)
    pu, ou, qu, _ = all_predictions(uk_data, tf_)
    sa, su = skill_from(pa, oa, qa), skill_from(pu, ou, qu)
    base.setdefault("n", (sa, su, acc_a)) if name == "normal" else None
    print(f"    {name:32s} {sa:+9.3f} {su:+9.3f} {acc_a:12.1%}")
print("    (Reading: a large drop means the model relies on that input. A drop near 0 means it does not, "
      "and the claim that the model exploits it is unsupported.)")

# ---------------------------------------------------------------- 6. physically impossible output
print("\n[6] the anchored output can dip below zero depth (a flood cannot be negative):")
for name, ex in (("Australia held-out", val_data), ("UK (never seen)", uk_data)):
    p, o, q, _ = all_predictions(ex)
    neg = float((p < 0).mean())
    print(f"    {name:20s} share of predicted pixels < 0 m: {neg:.2%}   min predicted depth {p.min():+.3f} m   "
          f"skill as reported {skill_from(p, o, q):+.4f}   skill if clipped at 0 {skill_from(np.maximum(p, 0), o, q):+.4f}")

# ---------------------------------------------------------------- 7. the capsule's readout vs the simplest possible one
print("\n[7] severity: does the capsule add anything beyond reading the last frame's maximum depth?")
last_max = np.array([x[-1, ..., 0].max() for x, _, _ in val_data])
truth = np.array([s for _, _, s in val_data])
pred = np.array([predict(x)[1] for x, _, _ in val_data])
base_cls = np.array([sum(m >= t for t in (0.3, 1.0, 2.0)) for m in last_max])
both_right = int(((pred == truth) & (base_cls == truth)).sum())
only_caps = int(((pred == truth) & (base_cls != truth)).sum())
only_base = int(((pred != truth) & (base_cls == truth)).sum())
neither = int(((pred != truth) & (base_cls != truth)).sum())
print(f"    windows: both right {both_right}, ONLY capsule right {only_caps}, ONLY threshold right {only_base}, neither {neither}  "
      f"(n={len(truth)})")
print(f"    correlation of capsule's predicted class with the last frame's max depth: "
      f"{np.corrcoef(pred, last_max)[0, 1]:+.3f}   (true class vs last max depth: {np.corrcoef(truth, last_max)[0, 1]:+.3f})")
print("=" * 78)
