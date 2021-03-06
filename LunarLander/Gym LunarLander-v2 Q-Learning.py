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
Graph : Graph Qvalue over time
Loggin : Output hyper-parameters in a separate file
Integrade Priorized Experience Replay (PER)

To try while training:

    dropout
    2 epochs

test 123


'''

import gym
from gym import wrappers
import numpy as np
import tensorflow as tf
import time
import os
import random
from collections import deque
import matplotlib.pyplot as plt

# Pre-flight parameters
suffix = '05'
logfile_name = './LunarLander_Logs/LunarLander_{}.log'.format(suffix)
parameters_name = './LunarLander_Logs/Hyper_Parameters.log'
modelsave_name = './LunarLander_Models/LunarLander_{}'.format(suffix)
modelload_name = './LunarLander_Models/LunarLander_{}'.format(suffix)
scoressave_name = './LunarLander_Logs/Scores/Score-{}.npy'.format(suffix)
qvaluessave_name = './LunarLander_Logs/Scores/Qvalues-{}.npy'.format(suffix)

debug_name = './LunarLander_Logs/LunarLander_Qlearn_debug_01.log'

try:
    logfile = open(logfile_name, 'w')
    debugfile = open(debug_name, 'w')
    parameters_log = open(parameters_name, 'w')
except FileNotFoundError:
    os.mknod(FileNotFoundError.filename)

logfile.write('\n')

save_model = True
load_model = False
replay_count = 1000
render = True
# Render every X games
render_interval = 25

# 1 to use gpu 0 to use CPU
use_gpu = 0
config = tf.ConfigProto(device_count={'GPU': use_gpu})

# Combined Experience Replay (Appends last sequence to minibatch sample)
CER = True

nn_layer_1_activation = 'relu'
nn_layer_1_units = 512
nn_output_activation = 'linear'

nn_dropout = False
nn_dropout_factor = 0.95

tau = 0
tau_max = 5000

epochs = 1

# Break training when reaching this moving average (last 100)
break_reward = 255
# Save model when scoring this in this moving average (last 100)
save_threshold = 190

lr = 1e-3
N = 3500

eps = 1
eps_decay = 0.995
eps_min = 0.1

gamma = 0.99

minibatch_size = 20
# When memory reach this threshold, start training process
minibatch_trigger = 600

memory = deque(maxlen=500000)

env = gym.make('LunarLander-v2')
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
    plt.title(script_name)
    #plt.draw()
    #plt.show()
    plt.show()

def play_one(env, model, eps, gamma, game_index):
    state = env.reset()
    done = False
    totalreward = 0
    global tau

    target_q_total = 0

    for step in range(env.spec.timestep_limit):

        # if state[6] == 1 and state[7] == 1 and state[4] < 0.18:
        #     action = 0
        # else:
        #     action = dqnetwork.sample_action(state, eps)

        action = dqnetwork.sample_action(state, eps)

        next_state, reward, done, info = env.step(action)
        totalreward += reward
        last_sequence = (state, action, reward, next_state, done)
        memory.append(last_sequence)

        state = next_state

        if len(memory) > minibatch_trigger:
            minibatch = random.sample(memory, minibatch_size)

            # Combined Experience Replay
            if CER:
                minibatch.append(last_sequence)

            # Train on the sampled minibatch. Returns mean of target Q values for stats
            target_q = dqnetwork.train(minibatch)

            target_q_total += target_q




        tau += 1


        if tau > tau_max:
            update_target = update_target_graph()
            sess.run(update_target)
            tau = 0
            print("Model updated")

        if render == True and game_index % render_interval == 0:
            env.render()

        if done:
            mean_target_q = target_q_total / step
            break

    logfile.write('Last game total reward: {}\n'.format(round(totalreward, 2)))
    return totalreward, mean_target_q

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
                #action = np.argmax(sess.run(dqnetwork.outputs, feed_dict={dqnetwork.inputs: observation}))

            observation, reward, done, info = env.step(action)
            game_score += reward

            env.render()

        print('total game score: {}'.format(game_score))
        totalrewards[game] = game_score
        if game % 10 == 0 and game > 0:
            output = 'Episode: ' + str(game) + "\navg reward (last 100): " + str(totalrewards[max(0, game - 100):(game + 1)].mean())
            print(output)

def log_parameters():

    parameters_log.write('\nEpochs: {}\nGamma: {}\nLearning Rate: {}\nMiniBatch_Size: {} \n'.format(epochs, gamma, lr, minibatch_size))
    parameters_log.write('Epsilon: {} (decay: {})\nOptimizer: Adam\nLoss Function: Huber loss\n'.format(eps, eps_decay))
    parameters_log.write(
        'Layer 1 : units: {} activation: {}\n'.format(nn_layer_1_units,nn_layer_1_activation))
    if nn_dropout:
        parameters_log.write('Dropout factor: {}\n\n'.format(nn_dropout_factor))
    else:
        parameters_log.write('No Dropout\n\n')

        parameters_log.flush()


    # #logfile.write(str(model.model.get_train_vars()))
    # logfile.write('\nEpochs: {}\nGamma: {}\nLearning Rate: {}\nMiniBatch_Size: {} \n'.format(epochs, gamma, lr, minibatch_size))
    # logfile.write('Epsilon: {} (decay: {})\nOptimizer: Adam\nLoss Function: Huber loss\n'.format(eps, eps_decay))
    # logfile.write(
    #     'Layer 1 : units: {} activation: {}\n'.format(nn_layer_1_units,nn_layer_1_activation))
    # if nn_dropout:
    #     logfile.write('Dropout factor: {}\n\n'.format(nn_dropout_factor))
    # else:
    #     logfile.write('No Dropout\n\n')
    #
    # logfile.flush()

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

        return target_Q.mean()

    def sample_action(self, s, eps):
        s = np.array(s).reshape(-1, 8)

        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            prediction = np.argmax(sess.run(self.outputs, feed_dict={self.inputs: s}))

            return prediction

###################FUNCTIONS&CLASSES############################################

# Reset the graphe
tf.reset_default_graph()

# Instantiate DQNetwork
dqnetwork = DQNet(name='dqnetwork', env=env)

# Instantiate Target DQNetwork
dqnetwork_target = DQNet(name='dqnetwork_target', env=env, target=True)

saver = tf.train.Saver()

if load_model:
    # Not worth using GPU for replaying model
    config_replay = tf.ConfigProto(device_count={'GPU': 0})

    with tf.Session(config=config_replay) as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, modelload_name)

        replay(dqnetwork, replay_count)

        exit()

totalrewards = np.empty(N)
costs = np.empty(N)

log_parameters()

# Main Session
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    scores_for_graph = []
    qvalues_for_graph = []

    for n in range(N):
        logfile.flush()

        if n % 25 == 0:
            print('playing game {}'.format(n))

        # Decaying Epsilon if higher than minimum
        if eps > eps_min:
            eps *= eps_decay
            #eps = eps_factor / np.sqrt(n + 1)

        # Play one game, returns stats for graphs
        totalreward, target_q_value = play_one(env, dqnetwork, eps, gamma, n)
        totalrewards[n] = totalreward

        scores_for_graph.append(totalreward)
        qvalues_for_graph.append(target_q_value)

        scores_for_graph_np = np.array(scores_for_graph)
        qvalues_for_graph_np = np.array(qvalues_for_graph)

        np.save(scoressave_name, scores_for_graph_np)
        np.save(qvaluessave_name, qvalues_for_graph_np)

        reward_avg_last50 = totalrewards[max(0, n - 50):(n + 1)].mean()
        reward_avg_last100 = totalrewards[max(0, n - 100):(n + 1)].mean()

        if n > 1 and n % 10 == 0:
            if save_model and reward_avg_last100 > save_threshold:
                saver.save(sess, modelsave_name, global_step=n)

            tx = time.time() - t0
            output = 'Episode: ' + str(n) + "\navg reward (last 100): " + str(reward_avg_last100)
            logfile.write('{}\nElapsed time : {}s\n\n'.format(output, round(tx, 2)))

        if reward_avg_last50 >= break_reward:
            break

plot_moving_avg(totalrewards, 100)

logfile.close()
debugfile.close()
env.close()
sess.close()
