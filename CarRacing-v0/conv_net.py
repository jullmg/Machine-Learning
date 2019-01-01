import tensorflow as tf
import numpy as np

nn_layer_1_activation = 'relu'
nn_layer_1_units = 512
nn_output_activation = 'linear'
output_size = 4

# kernel_initializer = tf.initializers.zeros
kernel_initializer = tf.contrib.layers.xavier_initializer()

lr = 1e-3
gamma = 0.99


class ConvDQNet:
    def __init__(self, name, env=None, sess=None, target=False):
        self.name = name
        self.env = env
        self.sess = sess

        #Debug data
        self.target_qvalue = 0
        self.reward = 0
        self.network_output = 0
        self.custom_value_1 = None
        self.custom_value_2 = None
        self.custom_value_3 = None


        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, [None, 3, 96, 96, 1], name="input")

            # Convolutional layer 1
            # self.conv_l1 = tf.contrib.layers.conv3d(self.input, 32, [5, 5, 1])
            self.conv_l1 = tf.layers.conv3d(self.input, 32, [5, 5, 3], padding="same", activation=tf.nn.relu, kernel_initializer=kernel_initializer)
            self.max_pool_l1 = tf.layers.max_pooling3d(self.conv_l1, pool_size=[2, 2, 1], strides=2)

            # Convolutional layer 2
            self.conv_l2 = tf.layers.conv3d(self.max_pool_l1, 64, [5, 5, 1], padding="same", activation=tf.nn.relu, kernel_initializer=kernel_initializer)
            self.max_pool_l2 = tf.layers.max_pooling3d(self.conv_l2, pool_size=[1, 1, 1], strides=2)

            # Dense layer
            self.flat_l1 = tf.reshape(self.max_pool_l2, [-1, 36864])

            self.hiddenlayer1 = tf.layers.dense(self.flat_l1, nn_layer_1_units, activation=tf.nn.relu)

            # Output = [turn left, turn right, gas, brake]
            self.outputs = tf.layers.dense(self.hiddenlayer1, output_size)

            if not target:
                self.target_Q = tf.placeholder(tf.float32, [None, output_size], name="target_Q")

                self.loss = tf.losses.mean_squared_error(self.target_Q, self.outputs)
                #self.loss = tf.losses.huber_loss(self.target_Q, self.outputs)

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
        result = np.argmax(self.sess.run(self.outputs, feed_dict={self.input: observation}))

        action = [0, 0, 0]

        if result == 0:
            action[0] = -1
        elif result == 1:
            action[0] = 1
        elif result == 2:
            action[1] = 1
        elif result == 3:
            action[2] = 1

        return action

    def train(self, data, target_network):
        x = []
        y = []

        for state, action, self.reward, next_state, done in data:
            x.append(state)
            action = np.argmax(action)

            self.target_qvalue = self.reward

            next_state = np.array(next_state).reshape(-1, 3, 96, 96, 1)
            state = np.array(state).reshape(-1, 3, 96, 96, 1)

            if not done:
                self.target_qvalue = self.reward + gamma * np.max(self.sess.run(target_network.outputs,
                                                             feed_dict={target_network.input:next_state}))

            target_f = self.sess.run(self.outputs, feed_dict={self.input: state})

            target_f[0][action] = self.target_qvalue

            y.append(target_f)

        y = np.array(y).reshape(-1, 4)
        x = np.array(x).reshape(-1, 3, 96, 96, 1)

        custom_1 = self.input
        custom_2 = self.conv_l1
        custom_3 = self.max_pool_l1

        _, loss, target_Q, outputs, self.custom_value_1, self.custom_value_2, self.custom_value_3 = \
            self.sess.run([self.train_op, self.loss, self.target_Q, self.outputs, custom_1, custom_2, custom_3],
                          feed_dict={self.input: x, self.target_Q: y})

    def sample_action(self, s, eps):
        s = np.array(s)
        s = s.reshape(-1, 3, 96, 96, 1)

        # Returns random floats in the half-open interval [0.0, 1.0).
        if np.random.random() < eps:
            if np.random.random() > 0.5:
                raw_result = [0, 0.001, 0]
            else:
                raw_result = self.env.action_space.sample()

            if raw_result[0] > 0:
                raw_result = np.insert(raw_result, 0, 0)

            else:
                raw_result[0] = abs(raw_result[0])
                raw_result = np.insert(raw_result, 1, 0)

            train_data = raw_result
            result = np.argmax(raw_result)

        else:
            train_data = self.sess.run(self.outputs, feed_dict={self.input: s})[0]
            result = np.argmax(train_data)

            self.network_output = train_data

        action = [0, 0, 0]

        if result == 0:
            action[0] = -1
        elif result == 1:
            action[0] = 1
        elif result == 2:
            action[1] = 1
        elif result == 3:
            action[2] = 1

        return action, train_data

    def parameters(self):
        return nn_layer_1_units, nn_layer_1_activation, gamma

    def debug(self):
        return round(self.target_qvalue, 3), self.reward, self.network_output, np.shape(self.custom_value_1), \
                     np.shape(self.custom_value_2), np.shape(self.custom_value_3)


