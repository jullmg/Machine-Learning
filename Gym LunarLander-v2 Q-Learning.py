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

'''

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
from collections import deque
#import matplotlib.pyplot as plt


# Pre-flight parameters
logfile_name = './LunarLander_Logs/LunarLander_Qlearn_23.log'
debug_name = './LunarLander_Logs/LunarLander_Qlearn_debug_01.log'
modelsave_name = './LunarLander_Models/LunarLander_Q_Learning_17-'
modelload_name = './LunarLander_Models/LunarLander_Q_Learning_09-'

try:
    logfile = open(logfile_name, 'w')
    debugfile = open(debug_name, 'w')
except FileNotFoundError:
    os.mknod(FileNotFoundError.filename)
    logfile = open(FileNotFoundError.filename, 'w')


logfile.write('\n')

redef_init_pop = False
init_pop_games = 10000
init_pop_goal = 0

pre_train = False
load_model = False
save_model = False
render = False

optimizer = 'Adam'
loss_function = 'mean_square'

nn_layer_1_activation = 'relu'
nn_layer_2_activation = 'tanh'
nn_output_activation = 'linear'
nn_dropout = False
nn_dropout_factor = 0.95
epochs = 2
batch = 64
train_step = 5
game_timeout = 3000

lr = 1e-3
N = 100000
eps = 1
eps_min = 0.01
eps_factor = 1 # only if using formula from original script
gamma = 0.99

batch_size = 4
memory = deque(maxlen=5000)

env = gym.make('LunarLander-v2')
t0 = time.time()

def init_pop(games):
    data = []
    total_kept_games = 0

    for n in range(games):
        env.reset()
        done = False
        total_score = 0
        game_memory = []
        while not done:
            action = np.random.randint(0, env.action_space.n)
            observation, reward, done, _ = env.step(action)
            game_memory.append([observation, action, reward])
            total_score += reward

        if total_score >= init_pop_goal:
            #print(total_score)
            for step in game_memory:
                data.append(step)
            total_kept_games += 1

    data = np.array(data)
    print('saving initpop_01')
    np.save('lunarlander_qlearn_initpop_01', data)


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


def create_nn(input_size):

    # Input layer
    network = input_data(shape=[None, input_size, 1], name='input')

    # Hidden layers
    network = fully_connected(network, 256, activation=nn_layer_1_activation)
    if nn_dropout:
        network = dropout(network, nn_dropout_factor)
    network = fully_connected(network, 512, activation=nn_layer_1_activation)
    if nn_dropout:
        network = dropout(network, nn_dropout_factor)

    # Output layer
    network = fully_connected(network, 1, activation=nn_output_activation)

    network = regression(network, optimizer=optimizer, learning_rate=lr, loss=loss_function, name='targets')

    # On definie le model
    model = tflearn.DNN(network)

    return model


class Model:
    def __init__(self, env):
        self.env = env

        # Creating as much models as there are actions
        self.graphlist = []
        self.modellist = []

        for graph in range(env.action_space.n):
            graph = tf.Graph()
            with graph.as_default():
                model = create_nn(env.observation_space.shape[0])
            self.graphlist.append(graph)
            self.modellist.append(model)

        title = './LunarLander_Models/LunarLander_Q_Learning_08-0'


    def prediction(self, s):
        s = s.reshape(-1, 8, 1)
        predictions = []

        # One prediction for each model
        for i in range(env.action_space.n):
            prediction = self.modellist[i].predict(s)
            predictions.append(prediction)

        return predictions

    def update(self, observation, action, G):
        X = observation.reshape(-1, len(observation), 1)
        G = [[G]]

        self.modellist[action].fit(X, G, n_epoch=epochs)

    def train(self, data):
        x = [[], [], [], []]
        y = [[], [], [], []]

        for step in data:
            x[step[1]].append(step[0])
            y[step[1]].append([step[2]])

        for n in range(len(x)):
            if len(x[n]) > 0

                x[n] = np.array(x[n])
                x[n] = x[n].reshape(-1, 8, 1)
                self.modellist[n].fit(x[n], y[n], n_epoch=epochs, batch_size=batch)

    def nn_save(self):
        print('Saving model')
        for i in range(env.action_space.n):
            graph = tf.Graph()
            with graph.as_default():
                title = modelsave_name + str(i)
                self.modellist[i].save(title)

    def nn_load(self):
        '''print('Loading model')
        for i in range(env.action_space.n):
            title = modelload_name + str(i)
            self.modellist[i].load(title)'''
        for i in range(env.action_space.n):
            graph = tf.Graph()
            with graph.as_default():
                title = modelload_name + str(i)
                self.modellist[i].load(title)


    def sample_action(self, s, eps):
        # np.random (0.01-0.99)
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return  np.argmax(model.prediction(s))


def play_one(env, model, eps, gamma):
    state = env.reset()
    done = False
    totalreward = 0

    while not done:
        '''
        if observation[6] == 1 and observation[7] == 1 and abs(craft_angle) < 0.35:
            action = 0
        else:
            action = model.sample_action(observation, eps)
        '''

        action = model.sample_action(state, eps)

        next_state, reward, done, info = env.step(action)
        totalreward += reward

        memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(memory) > batch_size:

            minibatch = random.sample(memory, batch_size)

            for state, action, reward, next_state, done_memory in minibatch:

                target = reward

                if not done_memory:
                    target = reward + gamma * np.max(model.prediction(next_state)) # use np.amax?

                model.update(state, action, target)

        #debugfile.write('Act: {} Pred : {} rew : {} eps : {}\n'.format(np.argmax(prediction), round(prediction_max, 2), round(reward, 2), round(eps, 2)))
        debugfile.flush()

        if render == True:
            env.render()

    logfile.write('Last game total reward: {}\n'.format(round(totalreward, 2)))
    return totalreward

if redef_init_pop == True:
    logfile.write('Redefining init pop for {} games\n'.format(init_pop_games))
    logfile.flush()
    init_pop(init_pop_games)

model = Model(env)

if load_model:
    model.nn_load()

if pre_train and not load_model:
    training_data = np.load('lunarlander_qlearn_initpop_01.npy')
    logfile.write('Training model with {} steps from random games with min {} score.'.format(len(training_data), init_pop_goal))
    logfile.flush()
    model.train(training_data)
    tx = time.time() - t0
    logfile.write('Training Done, elapsed time: {}s\n\n'.format(round(tx,2)))

totalrewards = np.empty(N)
costs = np.empty(N)

def log_parameters():
    logfile.write(str(model.modellist[0].get_train_vars()))
    logfile.write('\nEpochs: {}\nGamma: {}\nLearning Rate: {}\n'.format(epochs, gamma, lr))
    logfile.write('Epsilon: {}\nOptimizer: {}\nLoss Function: {}\n'.format(eps_factor, optimizer, loss_function))
    logfile.write(
        'Layer 1 activation: {}\nLayer 2 activation: {}\nOutput activation: {}\n'.format(nn_layer_1_activation,
                                                                                         nn_layer_2_activation,
                                                                                         nn_output_activation))
    if nn_dropout:
        logfile.write('Dropout factor: {}\n\n'.format(nn_dropout_factor))
    else:
        logfile.write('No Dropout\n\n')

    logfile.flush()

log_parameters()

for n in range(N):
    logfile.flush()

    if eps > eps_min:
        eps *= 0.995
        #eps = eps_factor / np.sqrt(n + 1)

    totalreward = play_one(env, model, eps, gamma)
    totalrewards[n] = totalreward

    if n > 1 and n % 10 == 0:
        if save_model:
            model.nn_save()


        if n > 1 and n % 100 == 0 and train_step < 3 :
            train_step += 1

        tx = time.time() - t0
        output = 'Episode: ' + str(n) + "\navg reward (last 100): " + str(totalrewards[max(0, n - 100):(n + 1)].mean())
        logfile.write('{}\nElapsed time : {}s\n\n'.format(output, round(tx, 2)))

    # If average totalreward of last 100 games is >=200 stop
    #if totalrewards[max(0, n - 100):(n + 1)].mean() >= 200:
    #    break


#plt.plot(totalrewards)
#plt.title("Rewards")
#plt.show()
#plot_running_avg(totalrewards)

logfile.close()
debugfile.close()
env.close()
