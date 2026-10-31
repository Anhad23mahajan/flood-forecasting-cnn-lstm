"""
Capsule Network layers (Sabour, Frosst & Hinton 2017, arXiv:1710.09829) for the
severity-classification branch: PrimaryCaps -> routing-by-agreement -> per-class
output capsules, whose vector lengths give the 4-way severity probability.

Kept as plain tf.keras.layers.Layer subclasses (not a separate framework) so the
whole model - depth-forecasting branch and severity branch alike - trains as one
TensorFlow/Keras graph: gradients from the severity loss reach back through
routing, through PrimaryCaps, into the SAME shared CNN encoder the LSTM branch
uses. This is the piece that was previously impossible when the capsule layer
lived in a separate PyTorch model (see project plan, Stage 3).
"""
import tensorflow as tf
from tensorflow.keras import layers


def squash(vectors: tf.Tensor, axis: int = -1, epsilon: float = 1e-7) -> tf.Tensor:
    """Capsule nonlinearity: shrinks short vectors to (near) zero and long vectors
    to just under length 1, so a capsule's output length can be read as a
    probability. squash(s) = (|s|^2 / (1 + |s|^2)) * (s / |s|)."""
    squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    scale = squared_norm / (1.0 + squared_norm) / tf.sqrt(squared_norm + epsilon)
    return scale * vectors


class PrimaryCaps(layers.Layer):
    """First capsule layer: a Conv2D projects a spatial feature map into
    (num_capsules * capsule_dim) channels, then reshapes each spatial location's
    channel group into one capsule vector, and squashes it."""

    def __init__(self, num_capsules: int, capsule_dim: int, kernel_size: int = 3,
                 strides: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.conv = layers.Conv2D(
            num_capsules * capsule_dim, kernel_size, strides=strides,
            padding="same", activation=None, name="primary_caps_conv",
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """inputs: (batch, H, W, C) spatial feature map.
        Returns: (batch, n_primary_caps, capsule_dim) - flattened over space and
        the num_capsules groups, each a squashed vector."""
        x = self.conv(inputs)  # (batch, h, w, num_capsules * capsule_dim)
        batch = tf.shape(x)[0]
        h, w = x.shape[1], x.shape[2]
        n_primary_caps = h * w * self.num_capsules
        x = tf.reshape(x, (batch, n_primary_caps, self.capsule_dim))
        return squash(x, axis=-1)


class RoutingCaps(layers.Layer):
    """Routing-by-agreement between a PrimaryCaps layer and this layer's output
    capsules (one per class). Each input capsule casts a weighted "vote" for each
    output capsule (via a learned transform W); routing coefficients are refined
    over `n_routing` iterations based on how well each vote agrees with the
    current output capsule (dot-product agreement), per Sabour et al.'s
    Algorithm 1 - this is what "routing-by-agreement" in the architecture
    diagram refers to, not a fixed pooling/summation."""

    def __init__(self, num_output_capsules: int, output_capsule_dim: int,
                 n_routing: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.num_output_capsules = num_output_capsules
        self.output_capsule_dim = output_capsule_dim
        self.n_routing = n_routing

    def build(self, input_shape):
        n_input_caps, input_capsule_dim = input_shape[1], input_shape[2]
        # W: one learned transform matrix per (input capsule, output capsule) pair
        self.W = self.add_weight(
            name="routing_weights",
            shape=(n_input_caps, self.num_output_capsules, self.output_capsule_dim, input_capsule_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """inputs: (batch, n_input_caps, input_capsule_dim)
        Returns: (batch, num_output_capsules, output_capsule_dim)"""
        batch = tf.shape(inputs)[0]
        n_input_caps = inputs.shape[1]

        # votes[b,i,j,:] = W[i,j] @ inputs[b,i,:]  -> (batch, n_in, n_out, out_dim).
        # einsum broadcasts the batch dim automatically (W has none) - no manual
        # tiling needed, which is exactly where the shape bug above came from.
        votes = tf.einsum("ijod,bid->bijo", self.W, inputs)

        routing_logits = tf.zeros((batch, n_input_caps, self.num_output_capsules))
        votes_stopped = tf.stop_gradient(votes)

        outputs = None
        for i in range(self.n_routing):
            routing_weights = tf.nn.softmax(routing_logits, axis=2)  # agreement coefficients c_ij, sum over j=1
            routing_weights = routing_weights[..., tf.newaxis]        # (b, n_in, n_out, 1)

            use_votes = votes if i == self.n_routing - 1 else votes_stopped
            weighted_sum = tf.reduce_sum(routing_weights * use_votes, axis=1)  # (b, n_out, out_dim)
            outputs = squash(weighted_sum, axis=-1)

            if i < self.n_routing - 1:
                agreement = tf.reduce_sum(votes_stopped * outputs[:, tf.newaxis, :, :], axis=-1)  # (b, n_in, n_out)
                routing_logits = routing_logits + agreement

        return outputs


def margin_loss(labels: tf.Tensor, lengths: tf.Tensor, m_plus: float = 0.9,
                m_minus: float = 0.1, down_weight: float = 0.5) -> tf.Tensor:
    """The margin loss from Sabour et al. 2017, eq. 4 - the loss capsule lengths were designed for.
    labels: (batch,) integer class ids; lengths: (batch, n_classes) in [0, 1).

    Why not softmax cross-entropy on the lengths: lengths are bounded in [0, 1), so a softmax over
    them can never exceed ~e/(e+3) = 0.475 for one of 4 classes - the loss floors near 0.74 and the
    gradient signal is weak (measured locally: predictions collapsed to one class). The margin loss
    instead pushes the true class's length above m_plus and every other class's below m_minus."""
    targets = tf.one_hot(labels, depth=lengths.shape[-1], dtype=lengths.dtype)
    present = targets * tf.square(tf.nn.relu(m_plus - lengths))
    absent = down_weight * (1.0 - targets) * tf.square(tf.nn.relu(lengths - m_minus))
    return tf.reduce_mean(tf.reduce_sum(present + absent, axis=-1))


def capsule_lengths(output_capsules: tf.Tensor, epsilon: float = 1e-7) -> tf.Tensor:
    """(batch, n_classes, dim) -> (batch, n_classes) vector lengths, read as
    per-class confidence. Softmax over these gives a proper severity probability
    distribution for use with standard categorical cross-entropy."""
    return tf.sqrt(tf.reduce_sum(tf.square(output_capsules), axis=-1) + epsilon)
