import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
tf = E._tf()
from depth_experiment import WarmupCosine
tr_pool, te_pool, tr_idx, va_idx, te_idx, held, seed = E.split_for("E60", 0, 3, log=lambda s: None)
tr_idx = tr_idx[:120]
T = 5
enc, dec, dech, OPT = E.build_nets("v2", T)
vars_ = enc.trainable_variables + dec.trainable_variables
opt = OPT["adam"](learning_rate=WarmupCosine(1e-3, 600, 100), clipnorm=1.0)
xs = tf.TensorSpec([1, T, 128, 128, 3], tf.float32); ys = tf.TensorSpec([5, 128, 128], tf.float32)
pers = np.mean([np.mean((np.array(tr_pool["df"][i]) - np.array(tr_pool["dh"][i][-1])[None]) ** 2) for i in tr_idx])
@tf.function(input_signature=[xs, ys])
def step(x, y):
    with tf.GradientTape() as tape:
        f = dech(dec, enc(x, training=True), x, 5, training=True)
        l = tf.reduce_mean(tf.square(f[0] - y)) / pers
    g = tape.gradient(l, vars_); opt.apply_gradients(zip(g, vars_)); return l
def deltas(mode_train, idxs):
    out = []
    for i in idxs:
        x, y, m = E.compose(tr_pool, i, "hybrid")
        f = dech(dec, enc(tf.constant(x[None]), training=mode_train), tf.constant(x[None]), 5, training=mode_train).numpy()[0]
        out.append(((f - x[T-1, ..., 0][None]).mean(), np.abs(f - x[T-1, ..., 0][None]).mean(), ((f - y) ** 2).mean() / pers))
    return np.array(out)
rng = np.random.default_rng(0)
for ep in range(3):
    ls = []
    for k in rng.permutation(len(tr_idx)):
        x, y, _ = E.compose(tr_pool, tr_idx[k], "hybrid")
        ls.append(float(step(tf.constant(x[None]), tf.constant(y))))
    d_eval, d_train = deltas(False, tr_idx[:10]), deltas(True, tr_idx[:10])
    print(f"epoch {ep+1} train-loss {np.mean(ls):.3f} | EVAL mode: std of mean-delta {d_eval[:,0].std():.2e}, mean |delta| {d_eval[:,1].mean():.2e}, loss/pers {d_eval[:,2].mean():.3f} | TRAIN mode: std {d_train[:,0].std():.2e}, |delta| {d_train[:,1].mean():.2e}, loss/pers {d_train[:,2].mean():.3f}", flush=True)
