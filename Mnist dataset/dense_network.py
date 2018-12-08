import tensorflow as tf

lr = 1e-3

class Dense_NN:
    def __init__(self, sess):
        self.sess = sess

        self.input = tf.placeholder(tf.float32, [None, 784])

        self.labels = tf.placeholder(tf.int8, [None, 10])

        self.layer_1 = tf.layers.dense(self.input, 1024, activation=tf.nn.sigmoid)

        self.layer_2 = tf.layers.dense(self.layer_1, 1024, activation=tf.nn.sigmoid)

        self.output = tf.layers.dense(self.layer_2, 10, activation=tf.nn.sigmoid)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.loss = tf.losses.softmax_cross_entropy(self.labels, self.output)

        self.train_op = self.optimizer.minimize(self.loss)

    def train(self, x, y):

        _, loss, output = self.sess.run([self.train_op, self.loss, self.output], feed_dict={self.input: x, self.labels: y})

        return loss, output