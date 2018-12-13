import tensorflow as tf
import numpy as np

lr = 1e-3


class Conv_NN:
    def __init__(self, sess):
        self.sess = sess

        self.input = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.labels = tf.placeholder(tf.int8, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)

        self.conv_l1 = tf.layers.conv2d(self.input, 32, [2, 2])
        self.max_pool_l1 = tf.layers.max_pooling2d(self.conv_l1, pool_size=[2, 2], strides=2)
        self.flat_l1 = tf.reshape(self.max_pool_l1, [-1, 13 * 13])

        self.layer_2 = tf.layers.dense(self.flat_l1, 512, activation=tf.nn.sigmoid)

        self.dropout_layer_2 = tf.nn.dropout(self.layer_2, keep_prob=self.keep_prob)

        self.output = tf.layers.dense(self.dropout_layer_2, 10, activation=tf.nn.sigmoid)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.loss = tf.losses.softmax_cross_entropy(self.labels, self.output)

        self.train_op = self.optimizer.minimize(self.loss)

    def train(self, x, y, dropout):
        _, loss, output = self.sess.run([self.train_op, self.loss, self.output, self.max_pool_l1], feed_dict={self.input: x, self.labels: y, self.keep_prob: dropout})

        return loss, output

    def test(self, x_test):
        prediction = self.sess.run(self.output, feed_dict={self.input: x_test, self.keep_prob: 1})

        return np.argmax(prediction)
