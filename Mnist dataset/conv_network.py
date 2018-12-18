import tensorflow as tf
import numpy as np

lr = 1e-3

class Conv_NN:
    def __init__(self, sess):
        self.sess = sess

        self.input = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.labels = tf.placeholder(tf.int8, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)

        # Convolutional layer 1
        self.conv_l1 = tf.layers.conv2d(self.input, 32, [5, 5],  padding="same", activation=tf.nn.relu)
        self.max_pool_l1 = tf.layers.max_pooling2d(self.conv_l1, pool_size=[2, 2], strides=2)

        # Convolutional layer 2
        self.conv_l2 = tf.layers.conv2d(self.max_pool_l1, 64, [5, 5], padding="same", activation=tf.nn.relu)
        self.max_pool_l2 = tf.layers.max_pooling2d(self.conv_l2, pool_size=[2, 2], strides=2)

        # Dense layer
        self.flat_l1 = tf.reshape(self.max_pool_l2, [-1, 7 * 7 * 64])
        self.dense_layer = tf.layers.dense(self.flat_l1, 1024, activation=tf.nn.relu)
        self.dropout = tf.nn.dropout(self.dense_layer, keep_prob=self.keep_prob)

        self.output = tf.layers.dense(self.dropout, 10)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.loss = tf.losses.softmax_cross_entropy(self.labels, self.output)

        self.train_op = self.optimizer.minimize(self.loss)

    def train(self, x, y, dropout):
        _, loss, output = self.sess.run([self.train_op, self.loss, self.output], feed_dict={self.input: x, self.labels: y, self.keep_prob: dropout})

        return loss, output

    def test(self, x_test):
        prediction = self.sess.run(self.output, feed_dict={self.input: x_test, self.keep_prob: 1})

        return np.argmax(prediction)
