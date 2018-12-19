import tensorflow as tf
import numpy as np

nn_layer_1_activation = 'relu'
nn_layer_1_units = 512
nn_output_activation = 'linear'
output_size = 4

lr = 1e-3


class ConvDQNet:
    def __init__(self, name, env=None, sess=None, target=False):
        self.name = name
        self.env = env
        self.sess = sess

        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, [None, 3, 96, 96, 1], name="input")
            # Convolutional layer 1
            self.conv_l1 = tf.layers.conv3d(self.input, 32, [5, 5, 1], padding="same", activation=tf.nn.relu)
            self.max_pool_l1 = tf.layers.max_pooling3d(self.conv_l1, pool_size=[2, 2, 1], strides=1)

            # Convolutional layer 2
            self.conv_l2 = tf.layers.conv3d(self.max_pool_l1, 64, [5, 5, 1], padding="same", activation=tf.nn.relu)
            self.max_pool_l2 = tf.layers.max_pooling3d(self.conv_l2, pool_size=[2, 2, 1], strides=1)

            # Dense layer
            self.flat_l1 = tf.reshape(self.max_pool_l2, [-1, 7 * 7 * 64])

            self.hiddenlayer1 = tf.layers.dense(self.flat_l1, nn_layer_1_units, activation=tf.nn.relu)

            self.outputs = tf.layers.dense(self.hiddenlayer1, output_size)

            if not target:
                self.target_Q = tf.placeholder(tf.float32, [None, output_size], name="target_Q")

                #self.loss = tf.losses.mean_squared_error(self.target_Q, self.outputs)
                self.loss = tf.losses.huber_loss(self.target_Q, self.outputs)

                #self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

                # Gradient Clipping -5,5
                self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                self.gvs = self.optimizer.compute_gradients(self.loss)
                self.capped_gvs = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in self.gvs]
                self.train_op = self.optimizer.apply_gradients(self.capped_gvs)

        '''
        self.original_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.optimizer = tf.contrib.estimator.clip_gradients_by_norm(self.original_optimizer, clip_norm=0.05)
        self.train_op = self.optimizer.minimize(self.loss)
        '''

    def predict(self, observation):
        prediction = self.sess.run(self.outputs, feed_dict={self.input: observation})
        return prediction

    def train(self, data):
        x = []
        y = []

        for state, action, reward, next_state, done in data:
            x.append(state)
            target_Qvalue = reward


            #print('reward', reward)

            next_state = np.array(next_state).reshape(-1, 8)

            state = np.array(state).reshape(-1, 8)

            if not done:
                #target = reward + gamma * np.max(self.predict(next_state))  # use np.amax?
                target_Qvalue = reward + gamma * np.max(self.sess.run(dqnetwork_target.outputs, feed_dict={dqnetwork_target.input:next_state})) # use np.amax?
                #print('target', target)

            #target_f = self.predict(state)
            target_f = self.sess.run(self.outputs, feed_dict={self.input:state})

            target_f[0][action] = target_Qvalue

            y.append(target_f)

        y = np.array(y).reshape(-1, 4)
        x = np.array(x).reshape(-1, 8)

        #print(sess.run(self.capped_gvs, feed_dict={self.input: x, self.target_Q: y}))
        _, loss, target_Q, outputs = self.sess.run([self.train_op, self.loss, self.target_Q, self.outputs], feed_dict={self.input: x, self.target_Q: y})
        # print(y)
        #print(target_Q)
        #print(outputs)

    def sample_action(self, s, eps):
        s = np.array(s).reshape(-1, 3, 96, 96, 1)
        #print(np.max(self.predict(s)))

        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            prediction = np.argmax(self.sess.run(self.outputs, feed_dict={self.input: s}))

            return prediction

    def parameters(self):
        return nn_layer_1_units, nn_layer_1_activation

class ConvNN2:
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
        self.dense_layer = tf.layers.dense(self.flat_l1, 512, activation=tf.nn.relu)
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
