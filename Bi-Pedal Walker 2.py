import gym
from gym import wrappers
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import time
import os
import math
import random
#import matplotlib.pyplot as plt

save_model = False
load_model = False
replay_count = 1000
render = False
# 1 to use gpu 0 to use CPU
use_gpu = 0
config = tf.ConfigProto(device_count={'GPU': use_gpu})

CER = True

tau = 0
tau_max = 5000

break_reward = 205

lr = 0.001
N = 10000
gamma = 0.99

eps = 1
eps_decay = 0.995
eps_min = 0.1

minibatch_size = 20
memory = deque(maxlen=500000)

env = gym.make('BipedalWalker-v2')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
t0 = time.time()



###################FUNCTIONS&CLASSES############################################

def plot_moving_avg(totalrewards, qty):
    Num = len(totalrewards)
    running_avg = np.empty(Num)
    for t in range(Num):
        running_avg[t] = totalrewards[max(0, t-qty):(t+1)].mean()

    plt.plot(running_avg)
    plt.title("Running Average")
    #plt.draw()
    #plt.show()
    plt.show(block=False)

def play_one(env, model, eps, gamma):
    state = env.reset()
    done = False
    totalreward = 0
    global tau

    while not done:

        action = dqnetwork.sample_action(state, eps)

        next_state, reward, done, info = env.step(action)
        totalreward += reward
        last_sequence = (state, action, reward, next_state, done)
        memory.append(last_sequence)

        state = next_state

        if len(memory) > 500:
            minibatch = random.sample(memory, minibatch_size)

            # Combined Experience Replay
            if CER:
                minibatch.append(last_sequence)

            dqnetwork.train(minibatch)

        tau += 1


        if tau > tau_max:
            update_target = update_target_graph()
            sess.run(update_target)
            tau = 0
            print("Model updated")

        if render == True:
            env.render()


    logfile.write('Last game total reward: {}\n'.format(round(totalreward, 2)))
    return totalreward

def log_parameters():
    #logfile.write(str(model.model.get_train_vars()))
    logfile.write('\nEpochs: {}\nGamma: {}\nLearning Rate: {}\nMiniBatch_Size: {} \n'.format(epochs, gamma, lr, minibatch_size))
    logfile.write('Epsilon: {} (decay: {})\nOptimizer: Adam\nLoss Function: Huber loss\n'.format(eps, eps_decay))
    logfile.write(
        'Layer 1 : units: {} activation: {}\n'.format(nn_layer_1_units,nn_layer_1_activation))
    if nn_dropout:
        logfile.write('Dropout factor: {}\n\n'.format(nn_dropout_factor))
    else:
        logfile.write('No Dropout\n\n')

    logfile.flush()

def update_target_graph():
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dqnetwork")

    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dqnetwork_target")

    op_holder = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))

    return op_holder

class DQNet:
    def __init__(self, name, env=None, target=False, actor=False, critic=False):
        self.name = name
        self.env = env

        # Critic build
        if critic:
            self.critic_inputs = tf.placeholder(tf.float32,[None, input_size], name="critic_inputs")

            self.critic_hiddenlayer1 = tf.layers.dense(self.critic_inputs, 1024, activation=tf.nn.relu)

            self.outputs = tf.layers.dense(self.hiddenlayer1, output_size, activation=tf.nn.tanh)




        # Actor build
        if actor:
            with tf.variable_scope(self.name):

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
        prediction = sess.run(self.outputs, feed_dict={self.inputs: observation})
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
                target_Qvalue = reward + gamma * np.max(sess.run(dqnetwork_target.outputs, feed_dict={dqnetwork_target.inputs:next_state})) # use np.amax?
                #print('target', target)

            #target_f = self.predict(state)
            target_f = sess.run(self.outputs, feed_dict={self.inputs:state})

            target_f[0][action] = target_Qvalue

            y.append(target_f)

        y = np.array(y).reshape(-1, 4)
        x = np.array(x).reshape(-1, 8)

        #print(sess.run(self.capped_gvs, feed_dict={self.inputs: x, self.target_Q: y}))
        _, loss, target_Q, outputs = sess.run([self.train_op, self.loss, self.target_Q, self.outputs], feed_dict={self.inputs: x, self.target_Q: y})
        # print(y)
        #print(target_Q)
        #print(outputs)

    def sample_action(self, s, eps):
        s = np.array(s).reshape(-1, input_size)
        # np.random (0.01-0.99)
        #print(np.max(self.predict(s)))

        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            #prediction = np.argmax(sess.run(self.outputs, feed_dict={self.inputs: s}))
            prediction = sess.run(self.critic_outputs, feed_dict={self.critic_inputs: s}))

            return prediction

###################FUNCTIONS&CLASSES############################################

tf.reset_default_graph()

# Instantiate Actor Network
nn_actor = DQNet(name='nn_actor', env=env)

# Instantiate Critic DQNetwork
nn_critic = DQNet(name='nn_critic', env=env)

# Instantiate Critic's Target DQNetwork
nn_critic_target = DQNet(name='nn_critic_target', env=env, target=True)

saver = tf.train.Saver()

totalrewards = np.empty(N)

#log_parameters()

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    for n in range(N):
        logfile.flush()

        if eps > eps_min:
            eps *= eps_decay

        # Play one game
        totalreward = play_one(env, nn_actor, eps, gamma)
        totalrewards[n] = totalreward

        reward_avg_last35 = totalrewards[max(0, n - 35):(n + 1)].mean()
        reward_avg_last100 = totalrewards[max(0, n - 100):(n + 1)].mean()

        if n > 1 and n % 10 == 0:
            if save_model:
                saver.save(sess, modelsave_name, global_step=n)

            tx = time.time() - t0
            output = 'Episode: ' + str(n) + "\navg reward (last 100): " + str(reward_avg_last100)
            logfile.write('{}\nElapsed time : {}s\n\n'.format(output, round(tx, 2)))

        if reward_avg_last35 >= break_reward:
            break

logfile.close()
debugfile.close()
env.close()
