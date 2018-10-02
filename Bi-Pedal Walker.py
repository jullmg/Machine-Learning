'''


To do:

essayer avec moins d'inputs
essayer avec plus d'actions possibles (ex : -0.5 ou 0.5)


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
#import matplotlib.pyplot as plt

env = gym.make('BipedalWalker-v2')
# Pre-flight parameters
logfile_name = './BipedalWalker_Logs/BipedalWalker_01.log'
modelsave_name = './LunarLander_Models/LunarLander_Q_Learning_11-'
modelload_name = './LunarLander_Models/LunarLander_Q_Learning_09-'

try:
    logfile = open(logfile_name, 'w')
except FileNotFoundError:
    os.mknod(logfile_name)
    logfile = open(logfile_name, 'w')


logfile.write('\n')

redef_init_pop = True
init_pop_games = 25

pre_train = True
save_model = False
load_model = False
render = True

optimizer = 'Adam'
loss_function = 'mean_square'

nn_layer_1_activation = 'relu'
nn_layer_2_activation = 'tanh'
nn_output_activation = 'linear'
nn_dropout = False
nn_dropout_factor = 0.95
epochs = 2

lr = 1e-3
N = 10000
eps_factor = 0.3
# Importance given to predicted action
gamma = 0.99

num_observations = 24
num_actions = 8

t0 = time.time()


def init_pop(games):
    data = []
    logfile.write('Redifining init pop')
    for n in range(games):
        env.reset()
        done = False

        while not done:
            action = [0, 0, 0, 0, 0, 0, 0, 0]
            action_index = random.randint(0, 7)

            if action_index % 2 == 0:
                action[action_index] = 1
            else:
                action[action_index] = -1

            observation, reward, done, _ = env.step(action)
            data.append([observation, action_index, reward])

    data = np.array(data)

    print('saving initpop_01')
    np.save('BipedalWalker_initpop_01', data)


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

        for graph in range(num_actions):
            graph = tf.Graph()
            with graph.as_default():
                model = create_nn(num_observations)
            self.graphlist.append(graph)
            self.modellist.append(model)


    def prediction(self, s):
        s = s.reshape(-1, len(s), 1)
        predictions = []

        # One prediction for each model
        for i in range(num_actions):
            prediction = self.modellist[i].predict(s)
            predictions.append(prediction)

        return predictions

    def update(self, observation, action, G):
        X = observation.reshape(-1, len(observation), 1)
        G = [[G]]

        self.modellist[action].fit(X, G, n_epoch=epochs)

    def train(self, init_pop_data):
        print('Training model with init. pop.')

        for s in init_pop_data:
            x = s[0]
            x = x.reshape(-1, len(x), 1)
            y = s[2]
            y = [[y]]

            self.modellist[s[1]].fit(x, y, n_epoch=epochs)

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
            return int(np.argmax(model.prediction(s)))


def play_one(env, model, eps, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0


    while not done:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        action_converted = []
        print(action)

        if type(action) == int:

            if action == 0:
                action_converted = [1, 0, 0, 0]

            elif action == 1:
                action_converted = [-1, 0, 0, 0]

            elif action == 2:
                action_converted = [0, 1, 0, 0]

            elif action == 3:
                action_converted = [0, -1, 0, 0]

            elif action == 4:
                action_converted = [0, 0, 1, 0]

            elif action == 5:
                action_converted = [0, 0, -1, 0]

            elif action == 6:
                action_converted = [0, 0, 0, 1]

            elif action == 7:
                action_converted = [0, 0, 0, -1]

            observation, reward, done, info = env.step(action_converted)
            prev_observation = observation
            next = model.prediction(observation)
            # G est eleve si le reward l'est aussi si la prediction etait avec confiance
            G = reward + gamma * np.max(next)
            model.update(prev_observation, action, G)

        else:
            observation, reward, done, info = env.step(action)



        if render:
            env.render()



        totalreward += reward
        iters += 1

    return totalreward, iters


if redef_init_pop == True:
    logfile.write('Redefining init pop for {} games\n'.format(init_pop_games))
    logfile.flush()
    init_pop(init_pop_games)

model = Model(env)


if load_model:
    model.nn_load()

if pre_train and not load_model:
    logfile.write('Pre Training with {} random games\n'.format(init_pop_games))
    logfile.flush()
    training_data = np.load('BipedalWalker_initpop_01.npy')
    model.train(training_data)
    tx = time.time() - t0
    logfile.write('Training Done, elapsed time: {}s\n\n'.format(round(tx,2)))


totalrewards = np.empty(N)
costs = np.empty(N)

logfile.write(str(model.modellist[0].get_train_vars()))
logfile.write('\nEpochs: {}\nGamma: {}\nLearning Rate: {}\n'.format(epochs, gamma, lr))
logfile.write('Epsilon: {}\nOptimizer: {}\nLoss Function: {}\n'.format(eps_factor, optimizer, loss_function))
logfile.write('Layer 1 activation: {}\nLayer 2 activation: {}\nOutput activation: {}\n'.format(nn_layer_1_activation, nn_layer_2_activation, nn_output_activation))
if nn_dropout:
    logfile.write('Dropout factor: {}\n\n'.format(nn_dropout_factor))
else:
    logfile.write('No Dropout\n\n')

for n in range(N):
    # Emptying buffer in log file
    logfile.flush()

    # Le eps diminue a chaque partie (max 1 min 0), c'est la probabilite de choisir une action au hasard

    eps = eps_factor / np.sqrt(n + 1)
    totalreward, iters = play_one(env, model, eps, gamma)
    totalrewards[n] = totalreward

    if n > 1 and n % 10 == 0:
        #logfile.write('Saving model' + '\n')
        if save_model:
            model.nn_save()
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
env.close()
