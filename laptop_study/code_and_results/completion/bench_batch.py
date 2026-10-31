import sys, time, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
tf = E._tf()
arch, B = sys.argv[1], int(sys.argv[2]); T = 5
enc, dec, dech, OPT = E.build_nets(arch, T)
vars_ = enc.trainable_variables + dec.trainable_variables
opt = OPT["adam"](learning_rate=1e-3, clipnorm=1.0)
xs = tf.TensorSpec([B, T, 128, 128, 3], tf.float32); ys = tf.TensorSpec([B, 5, 128, 128], tf.float32)
@tf.function(input_signature=[xs, ys])
def step(x, y):
    with tf.GradientTape() as tape:
        f = dech(dec, enc(x, training=True), x, 5, training=True)
        l = tf.reduce_mean(tf.square(f - y))
    g = tape.gradient(l, vars_); opt.apply_gradients((gi, v) for gi, v in zip(g, vars_) if gi is not None); return l
x = tf.constant(np.random.rand(B, T, 128, 128, 3).astype("float32")); y = tf.constant(np.random.rand(B, 5, 128, 128).astype("float32"))
for _ in range(3): step(x, y)
t0 = time.perf_counter()
for _ in range(10): step(x, y)
dt = (time.perf_counter() - t0) / 10
print(f"{arch} batch {B}: {dt*1000:.0f} ms/step = {dt/B*1000:.0f} ms per window")
