import tensorflow as tf

lr = 1e-3

class Dense_NN:
    def __init__(self):
        self.input = tf.placeholder(tf.float32, [None, 784])

        self.labels = tf.placeholder([None, 10])

        self.layer_1 = tf.layers.dense(input, 512, activation=tf.nn.sigmoid)

        self.output = tf.layers.dense(layer_1, 10, activation=tf.nn.sigmoid)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.loss = tf.losses.softmax_cross_entropy(self.labels, self.output)

        self.train_op = self.optimizer.minimize(self.loss)

    def train(self, data, epochs, batch_size):
        for _ in range(epochs):
            for _ in range(batch_size)
                sess.run([self.train_op], feed_dict={self.input: data})

