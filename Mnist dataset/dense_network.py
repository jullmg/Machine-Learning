import tensorflow as tf
import numpy as np

lr = 1e-3
activation = tf.nn.relu


class Dense_NN:
    def __init__(self, sess):
        self.sess = sess

        self.input = tf.placeholder(tf.float32, [None, 784])
        self.labels = tf.placeholder(tf.int8, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)

        self.layer_1 = tf.layers.dense(self.input, 256, activation=activation)

        self.dropout_layer_1 = tf.nn.dropout(self.layer_1, keep_prob=self.keep_prob)

        self.layer_2 = tf.layers.dense(self.dropout_layer_1, 512, activation=activation)

        self.dropout_layer_2 = tf.nn.dropout(self.layer_2, keep_prob=self.keep_prob)

        self.output = tf.layers.dense(self.dropout_layer_2, 10)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.loss = tf.losses.softmax_cross_entropy(self.labels, self.output)

        self.train_op = self.optimizer.minimize(self.loss)

    def train(self, x, y, dropout):
        _, loss, output = self.sess.run([self.train_op, self.loss, self.output], feed_dict={self.input: x, self.labels: y, self.keep_prob: dropout})

        return loss, output

    def test(self, x_test):
        prediction = self.sess.run(self.output, feed_dict={self.input: x_test, self.keep_prob: 1})

        return np.argmax(prediction)
