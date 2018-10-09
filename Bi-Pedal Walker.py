'''


To do:
double neurones
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
logfile_name = './BipedalWalker_Logs/BipedalWalker_04.log'
modelsave_name = './LunarLander_Models/LunarLander_Q_Learning_11-'
modelload_name = './LunarLander_Models/LunarLander_Q_Learning_09-'

try:
    logfile = open(logfile_name, 'w')
except FileNotFoundError:
    os.mknod(logfile_name)
    logfile = open(logfile_name, 'w')


logfile.write('tripled neurones\n')

redef_init_pop = False
init_pop_games = 50
init_pop_target = -999

pre_train = False
save_model = False
load_model = False
render = False

optimizer = 'Adam'
loss_function = 'mean_square'

nn_layer_1_activation = 'relu'
nn_layer_2_activation = 'tanh'
nn_output_activation = 'linear'
nn_dropout = False
nn_dropout_factor = 0.95
epochs = 1

lr = 1e-3
N = 10000
eps_factor = 1
# Importance given to predicted action
gamma = 0.99

observations_retained_indices = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
num_observations = len(observations_retained_indices)
num_actions = 8
game_timeout = 1650
init_timeout = 300

t0 = time.time()

'''
            self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*self.hull.angularVelocity/FPS,
            0.3*vel.x*(VIEWPORT_W/SCALE)/FPS,  # Normalized to get -1..1 range
            0.3*vel.y*(VIEWPORT_H/SCALE)/FPS,
            self.joints[0].angle,   # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0
'''


def init_pop(games):
    kept_data = []

    for n in range(games):
        observation = env.reset()
        done = False
        game_data = []
        game_score = 0

        for i in range(init_timeout):
            previous_observation = observation
            action = [0, 0, 0, 0, 0, 0, 0, 0]
            action_index = random.randint(0, 7)

            if action_index % 2 == 0:
                action[action_index] = 1
            else:
                action[action_index] = -1

            observation, reward, done, _ = env.step(action)
            game_data.append([previous_observation, action_index, reward])
            game_score += reward

            if done:
                break

        if game_score >= init_pop_target:
            for s in game_data:
                kept_data.append(s)

    kept_data = np.array(kept_data)
    np.save('BipedalWalker_initpop_01', kept_data)


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
    network = fully_connected(network, 256, activation=nn_layer_2_activation)
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
        s = list(s[observations_retained_indices])
        s = np.array(s)
        s = s.reshape(-1, len(s), 1)
        predictions = []

        # One prediction for each model
        for i in range(num_actions):

            prediction = self.modellist[i].predict(s)
            predictions.append(prediction)

        return predictions

    def update(self, observation, action, G):
        X = list(observation[observations_retained_indices])
        X = np.array(X)
        X = X.reshape(-1, len(X), 1)
        G = [[G]]

        self.modellist[action].fit(X, G, n_epoch=epochs)

    def train(self, init_pop_data):
        logfile.write('Training model with {} steps from init. pop min value {}.'.format(len(init_pop_data), init_pop_target))
        logfile.flush()

        for s in init_pop_data:
            x = s[0]
            x = list(x[observations_retained_indices])
            x = np.array(x)
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
            return random.randint(0, 7)
        else:
            return int(np.argmax(model.prediction(s)))


def play_one(env, model, eps, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    for g in range(game_timeout):
        action = model.sample_action(observation, eps)
        print(action)
        prev_observation = observation
        action_converted = []

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
        next = model.prediction(observation)
        # G est eleve si le reward l'est aussi si la prediction etait avec confiance
        G = reward + gamma * np.max(next)
        model.update(prev_observation, action, G)

        if render:
            env.render()

        totalreward += reward
        iters += 1

        if done:
            break

    logfile.write('Last game total reward: {}\n'.format(totalreward))
    return totalreward, iters


def log_parameters():
    logfile.write(str(model.modellist[0].get_train_vars()))
    logfile.write('\nEpochs: {}\nGamma: {}\nLearning Rate: {}\n'.format(epochs, gamma, lr))
    logfile.write('Epsilon: {}\nOptimizer: {}\nLoss Function: {}\n'.format(eps_factor, optimizer, loss_function))
    logfile.write(
        'Layer 1 activation: {}\nLayer 2 activation: {}\nOutput activation: {}\n'.format(nn_layer_1_activation,
                                                                                         nn_layer_2_activation,
                                                                                         nn_output_activation))
    logfile.flush()


if redef_init_pop:
    logfile.write('Redefining init pop for {} games\n'.format(init_pop_games))
    logfile.flush()
    init_pop(init_pop_games)

model = Model(env)


if load_model:
    model.nn_load()

if pre_train and not load_model:
    training_data = np.load('BipedalWalker_initpop_01.npy')
    model.train(training_data)
    tx = time.time() - t0
    logfile.write('Training Done, elapsed time: {}s\n\n'.format(round(tx,2)))


totalrewards = np.empty(N)

log_parameters()

for n in range(N):
    logfile.flush()

    eps = 0.5

    if n > 200:
        eps = 0.4
    elif n > 400:
        eps = 0.3
    elif n > 700:
        eps = 0.2
    elif n > 1000:
        eps = 0.1


    totalreward, iters = play_one(env, model, eps, gamma)

    totalrewards[n] = totalreward

    if n > 1 and n % 10 == 0:
        if save_model:
            model.nn_save()
        tx = time.time() - t0
        output = 'Episode: ' + str(n) + "\navg reward (last 100): " + str(totalrewards[max(0, n - 100):(n + 1)].mean())
        logfile.write('{}\nElapsed time : {}s\n\n'.format(output, round(tx, 2)))

logfile.close()
env.close()
