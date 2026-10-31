def squash(vectors):
    # Shrink each vector's length to between 0 and 1, keeping its direction
    squared_length = tf.reduce_sum(tf.square(vectors), axis=-1, keepdims=True)
    return squared_length / (1 + squared_length) * vectors / tf.sqrt(squared_length + 1e-7)


class Squash(layers.Layer):
    # A layer that applies squash
    def call(self, vectors):
        return squash(vectors)


class ClassCapsules(layers.Layer):
    # One capsule per class, filled by routing by agreement

    def __init__(self, number_of_classes, capsule_size, routing_rounds=3, **kwargs):
        super().__init__(**kwargs)
        self.number_of_classes = number_of_classes
        self.capsule_size = capsule_size
        self.routing_rounds = routing_rounds

    def build(self, input_shape):
        number_of_inputs = input_shape[1]
        input_size = input_shape[2]

        # One weight matrix for every (small capsule, class capsule) pair
        self.vote_weights = self.add_weight(
            shape=(number_of_inputs, self.number_of_classes, self.capsule_size, input_size),
            initializer=keras.initializers.RandomNormal(stddev=0.05),
            name="vote_weights")

    def call(self, primary_capsules):
        # Every small capsule makes a vote for every class
        votes = tf.einsum("pcoi,bpi->bpco", self.vote_weights, primary_capsules)

        # At the start every vote counts the same
        routing_scores = tf.zeros_like(votes[..., 0])

        for round_number in range(self.routing_rounds):
            # How much each small capsule sends to each class
            coupling = tf.nn.softmax(routing_scores, axis=2)

            # Class capsule = weighted sum of its votes, squashed
            class_capsules = squash(tf.einsum("bpc,bpco->bco", coupling, votes))

            # Votes that agree with the result count more in the next round
            if round_number < self.routing_rounds - 1:
                routing_scores = routing_scores + tf.einsum("bpco,bco->bpc", votes, class_capsules)

        return class_capsules


class CapsuleLength(layers.Layer):
    # Length of each class capsule = how sure the network is
    def call(self, capsules):
        return tf.sqrt(tf.reduce_sum(tf.square(capsules), axis=-1) + 1e-7)


def margin_loss(true_classes, lengths):
    # The right class should be longer than 0.9, the others shorter than 0.1
    correct = true_classes * tf.square(tf.nn.relu(0.9 - lengths))
    wrong = 0.5 * (1 - true_classes) * tf.square(tf.nn.relu(lengths - 0.1))
    return tf.reduce_sum(correct + wrong, axis=-1)
