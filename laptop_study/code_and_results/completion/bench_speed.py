import sys, time, numpy as np, tensorflow as tf
nthreads = int(sys.argv[1]); T = int(sys.argv[2])
tf.config.threading.set_intra_op_parallelism_threads(nthreads); tf.config.threading.set_inter_op_parallelism_threads(1)
sys.path.insert(0, "."); import model as M
from model import decode_horizon
enc = M.build_encoder(128, 0.3); dec = M.build_decoder_step(128, 0.3)
opt = M.OPTIMIZERS["adam"](learning_rate=1e-3, clipnorm=1.0); vars_ = enc.trainable_variables + dec.trainable_variables
xs = tf.TensorSpec([1, T, 128, 128, 3], tf.float32); ys = tf.TensorSpec([5, 128, 128], tf.float32)
@tf.function(input_signature=[xs, ys])
def step(x, y):
    with tf.GradientTape() as tape:
        f = decode_horizon(dec, enc(x, training=True), x, 5, training=True)
        l = tf.reduce_mean(tf.square(f[0] - y))
    g = tape.gradient(l, vars_); opt.apply_gradients(zip(g, vars_)); return l
x = tf.constant(np.random.rand(1, T, 128, 128, 3).astype("float32")); y = tf.constant(np.random.rand(5, 128, 128).astype("float32"))
for _ in range(5): step(x, y)
t0 = time.perf_counter()
for _ in range(60): step(x, y)
print(f"threads={nthreads} T={T}: {(time.perf_counter() - t0) / 60 * 1000:.1f} ms/step -> 400 steps = {(time.perf_counter() - t0) / 60 * 400:.1f} s/epoch")
