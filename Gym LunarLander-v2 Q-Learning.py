'''
STATES :

    (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.lander.angle,
            20.0*self.lander.angularVelocity/FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0

Actions :

        op, fire left engine, main engine, right engine

To do :
Plot graphs
Integrate dual deep network (target network)
Integrate Combined Experience Replay (CER)
Integrade Priorized Experience Replay (PER)

Train steps?

'''

import gym
from gym import wrappers
import numpy as np
import tensorflow as tf
import time
import os
import math
import random
from collections import deque
#import matplotlib.pyplot as plt


# Pre-flight parameters
logfile_name = './LunarLander_Logs/LunarLander_Qlearn_10.log'
modelsave_name = './LunarLander_Models/LunarLander_Q_Learning_10'
modelload_name = './LunarLander_Models/LunarLander_Q_Learning_10-1850'
debug_name = './LunarLander_Logs/LunarLander_Qlearn_debug_01.log'

try:
    logfile = open(logfile_name, 'w')
    debugfile = open(debug_name, 'w')
except FileNotFoundError:
    os.mknod(FileNotFoundError.filename)
    logfile = open(FileNotFoundError.filename, 'w')

logfile.write('\n')

save_model = True
load_model = False
replay_count = 1000
render = False

nn_layer_1_activation = 'relu'
nn_layer_1_units = 512
nn_output_activation = 'linear'
nn_dropout = False
nn_dropout_factor = 0.95

tau = 0
tau_max = 10000

epochs = 1
break_reward = 205

lr = 0.001
N = 100000
eps = 1
eps_decay = 0.995
eps_min = 0.1
eps_factor = 1 # only if using formula from original script
gamma = 0.99

# 20 semble optimal
minibatch_size = 20
memory = deque(maxlen=500000)

env = gym.make('LunarLander-v2')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
t0 = time.time()

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

class DQNet:
    def __init__(self, name, env=None, target=False):
        self.name = name
        self.env = env

        with tf.variable_scope(self.name):
            self.inputs = tf.placeholder(tf.float32,[None, input_size], name="inputs")

            self.hiddenlayer1 = tf.layers.dense(self.inputs, nn_layer_1_units, activation=tf.nn.relu)

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
        s = np.array(s).reshape(-1, 8)
        # np.random (0.01-0.99)
        #print(np.max(self.predict(s)))

        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            prediction = np.argmax(sess.run(self.outputs, feed_dict={self.inputs: s}))

            return prediction

# Reset the graph
tf.reset_default_graph()

# Instantiate DQNetwork
dqnetwork = DQNet(name='dqnetwork', env=env)

# Instantiate Target DQNetwork
dqnetwork_target = DQNet(name='dqnetwork_target', env=env, target=True)

saver = tf.train.Saver()

# This function helps us to copy one set of variables to another
# In our case we use it when we want to copy the parameters of DQN to Target_network
# Thanks of the very good implementation of Arthur Juliani https://github.com/awjuliani
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

def play_one(env, model, eps, gamma):
    state = env.reset()
    done = False
    totalreward = 0
    global tau

    while not done:

        if state[6] == 1 and state[7] == 1 and state[4] < 0.1:
            action = 0
        else:
            action = dqnetwork.sample_action(state, eps)

        # state = np.array(state).reshape(-1, 8)
        # print('Network pred:', sess.run(dqnetwork.outputs, feed_dict={dqnetwork.inputs: state}))
        # print('Target  pred:', sess.run(dqnetwork_target.outputs, feed_dict={dqnetwork_target.inputs: state}))

        next_state, reward, done, info = env.step(action)
        totalreward += reward

        memory.append((state, action, reward, next_state, done))

        state = next_state


        if len(memory) > 5000:
            minibatch = random.sample(memory, minibatch_size)
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

def replay(model, num):
    totalrewards = np.empty(replay_count)

    for game in range(num):
        observation = env.reset()
        game_score = 0
        done = False

        while not done:
            if observation[6] == 1 and observation[7] == 1 and observation[4] < 0.18:
                action = 0
            else:
                observation = observation.reshape(-1, 8)
                action = np.argmax(model.predict(observation))

            observation, reward, done, info = env.step(action)
            game_score += reward

            env.render()

        print('total game score: {}'.format(game_score))
        totalrewards[game] = game_score
        if game % 10 == 0 and game > 0:
            output = 'Episode: ' + str(game) + "\navg reward (last 100): " + str(totalrewards[max(0, game - 100):(game + 1)].mean())
            print(output)

if load_model:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, modelload_name)

        replay(dqnetwork, replay_count)

totalrewards = np.empty(N)
costs = np.empty(N)

def log_parameters():
    #logfile.write(str(model.model.get_train_vars()))
    logfile.write('\nEpochs: {}\nGamma: {}\nLearning Rate: {}\nMiniBatch_Size: {} \n'.format(epochs, gamma, lr, minibatch_size))
    logfile.write('Epsilon: {} (decay: {})\nOptimizer: Adam\nLoss Function: Huber loss\n'.format(eps_factor,eps_decay))
    logfile.write(
        'Layer 1 : units: {} activation: {}\n'.format(nn_layer_1_units,nn_layer_1_activation))
    if nn_dropout:
        logfile.write('Dropout factor: {}\n\n'.format(nn_dropout_factor))
    else:
        logfile.write('No Dropout\n\n')

    logfile.flush()

log_parameters()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    for n in range(N):
        logfile.flush()


        if eps > eps_min:
            eps *= eps_decay
            #eps = eps_factor / np.sqrt(n + 1)

        # Play one game
        totalreward = play_one(env, dqnetwork, eps, gamma)
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

#plt.plot(totalrewards)
#plt.title("Rewards")
#plt.show()
#plot_running_avg(totalrewards)

logfile.close()
debugfile.close()
env.close()
