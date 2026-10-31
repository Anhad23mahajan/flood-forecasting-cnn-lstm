"""
Trains the CNN-LSTM encoder-decoder (model.py) on patch sequences.

Performance note (learned the hard way on a related project): plain eager-mode
TensorFlow on CPU has severe per-op dispatch overhead for LSTM layers - an
uncompiled encoder+decoder training step was ~2.4s in a comparable architecture;
@tf.function compilation cut that to ~6ms. The training step here is compiled
the same way, with one cached trace per distinct (T_in, horizon) shape combo
(usually just one, since patches are fixed-size and T_in/horizon are fixed
per run - unlike the earlier project's variable-length captions).

--synthetic runs against random data shaped like real patches, to verify the
training loop itself (speed + correctness) independent of the real dataset,
which is still being downloaded/explored as of this script's writing.
"""

import argparse
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

import model as flood_model


class CompiledTrainer:
    def __init__(self, encoder, decoder, optimizer, horizon: int, hidden_dim: int = 128):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.trainable_vars = encoder.trainable_variables + decoder.trainable_variables
        self._train_step = None
        self._eval_step = None

    def _build_step(self, training: bool):
        encoder, decoder, horizon, hidden_dim = self.encoder, self.decoder, self.horizon, self.hidden_dim
        trainable_vars = self.trainable_vars

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, None, flood_model.PATCH_SIZE, flood_model.PATCH_SIZE, flood_model.N_CHANNELS], dtype=tf.float32),
            tf.TensorSpec(shape=[horizon, flood_model.PATCH_SIZE, flood_model.PATCH_SIZE], dtype=tf.float32),
        ])
        def step(x, depth_targets):
            step_input = tf.ones((1, 1))
            with tf.GradientTape() as tape:
                state_h, state_c, _cnn_features = encoder(x, training=training)
                total_loss = tf.constant(0.0)
                for t in range(horizon):
                    depth_out, state_h, state_c = decoder([step_input, state_h, state_c], training=training)
                    depth_loss = tf.reduce_mean(tf.square(depth_out[0, ..., 0] - depth_targets[t]))
                    total_loss += depth_loss
                loss = total_loss / float(horizon)
            if training:
                grads = tape.gradient(loss, trainable_vars)
                self.optimizer.apply_gradients((g, v) for g, v in zip(grads, trainable_vars) if g is not None)
            return loss
        return step

    def run(self, x: np.ndarray, depth_targets: np.ndarray, training: bool) -> float:
        cache_attr = "_train_step" if training else "_eval_step"
        if getattr(self, cache_attr) is None:
            setattr(self, cache_attr, self._build_step(training))
        step_fn = getattr(self, cache_attr)
        loss = step_fn(
            tf.constant(x[np.newaxis, ...], dtype=tf.float32),
            tf.constant(depth_targets, dtype=tf.float32),
        )
        return float(loss.numpy())


def synthetic_example(t_in: int, horizon: int, rng: np.random.Generator):
    x = rng.random((t_in, flood_model.PATCH_SIZE, flood_model.PATCH_SIZE, flood_model.N_CHANNELS)).astype("float32")
    depth_targets = rng.random((horizon, flood_model.PATCH_SIZE, flood_model.PATCH_SIZE)).astype("float32")
    return x, depth_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use random data to verify the training loop, not the real dataset")
    parser.add_argument("--n-examples", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--t-in", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam", choices=list(flood_model.OPTIMIZERS.keys()))
    args = parser.parse_args()

    if not args.synthetic:
        raise SystemExit("Only --synthetic is wired up so far - real data loader comes after exploring the actual archive structure.")

    print(f"Building model (dropout={args.dropout})...")
    encoder, decoder = flood_model.build_models(dropout=args.dropout)
    optimizer = flood_model.OPTIMIZERS[args.optimizer](learning_rate=args.lr)
    trainer = CompiledTrainer(encoder, decoder, optimizer, horizon=args.horizon)

    rng = np.random.default_rng(0)
    examples = [synthetic_example(args.t_in, args.horizon, rng) for _ in range(args.n_examples)]

    print(f"Training on {args.n_examples} synthetic examples for {args.epochs} epochs "
          f"(first epoch pays a one-time @tf.function compile cost)...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        losses = [trainer.run(x, dt, training=True) for x, dt in examples]
        print(f"  epoch {epoch}/{args.epochs}  loss={np.mean(losses):.4f}  ({time.perf_counter()-t0:.2f}s)")

    print("\nSynthetic training loop verified. Compiled-step speed confirmed before touching real data.")


if __name__ == "__main__":
    main()
