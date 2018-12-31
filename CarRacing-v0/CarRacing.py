'''
STATE :
RGB screen
(96, 96, 3)

Actions :
        steer -1 left, 1 right, gas (0/1), brake (0/1)
        self.action_space = spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]), dtype=np.float32) # steer, gas, brake

To do :

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
from conv_net import ConvDQNet

# Pre-flight parameters
script_num = '01'
logfile_name = './CarRacing_Logs/CarRacing_{}.log'.format(script_num)
modelsave_name = './CarRacing_Models/CarRacing_Qlearn_{}'.format(script_num)
modelload_name = './CarRacing_Models/CarRacing_Qlearn_{}-110'.format(script_num)

debug_name = './CarRacing_Logs/CarRacing_debug_01.log'

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
debug = True

# 1 to use gpu 0 to use CPU
use_gpu = 1
config = tf.ConfigProto(device_count={'GPU': use_gpu})

CER = True
tf.initializers.random_normal

nn_dropout = False
nn_dropout_factor = 0.95

tau = 0
tau_max = 5000

epochs = 1

break_reward = 999

lr = 0.001
N = 1500

eps = 0.9
eps_decay = 0.975
eps_min = 0.1

# 20 semble optimal
minibatch_size = 12
memory = deque(maxlen=500000)
minibatch_trigger = 300

env = gym.make('CarRacing-v0')


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

def play_one(env, model, eps, gamma):
    state = env.reset()

    done = False
    totalreward = 0
    global tau

    for step in range(env.spec.timestep_limit):
        action, train_data = dqnetwork.sample_action(state, eps)
        next_state, reward, done, info = env.step(action)
        last_sequence = (state, train_data, reward, next_state, done)

        totalreward += reward

        memory.append(last_sequence)

        state = next_state

        if len(memory) > minibatch_trigger:
            minibatch = random.sample(memory, minibatch_size)


            # Combined Experience Replay
            if CER:
                minibatch.append(last_sequence)

            dqnetwork.train(minibatch, dqnetwork_target)

        if debug:
            target_qvalue, step_reward, network_output, custom_value_1 = dqnetwork.debug()
            step_reward = round(step_reward, 2)

            debugfile.write('Network output: {}\nReward: {}\n\n'.format(train_data, reward))
            # debugfile.write('Target Q Value: {}\nReward: {}\nNetwork Output: {}\nCustom Value 1: {}\n\n'.format(target_qvalue, step_reward, network_output, custom_value_1))
            debugfile.flush()

        tau += 1


        if tau > tau_max:
            update_target = update_target_graph()
            sess.run(update_target)
            tau = 0
            print("Model updated")

        if render == True:
            env.render()

        if done:
            break

    logfile.write('Last game total reward: {}\n'.format(round(totalreward, 2)))

    return totalreward


def replay(model, num):
    totalrewards = np.empty(replay_count)

    for game in range(num):
        observation = env.reset()
        game_score = 0
        done = False

        while not done:
            observation = observation.reshape(-1, 3, 96, 96, 1)
            action = model.predict(observation)
            print(action)
            observation, reward, done, info = env.step(action)

            game_score += reward

            env.render()

        print('total game score: {}'.format(game_score))
        totalrewards[game] = game_score
        if game % 10 == 0 and game > 0:
            output = 'Episode: ' + str(game) + "\navg reward (last 100): " + str(totalrewards[max(0, game - 100):(game + 1)].mean())


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

###################FUNCTIONS&CLASSES############################################

# Reset the graph


tf.reset_default_graph()

totalrewards = np.empty(N)

with tf.Session(config=config) as sess:
    # Instantiate DQNetwork
    dqnetwork = ConvDQNet(name='dqnetwork', env=env, sess=sess)

    # Instantiate Target DQNetwork
    dqnetwork_target = ConvDQNet(name='dqnetwork_target', env=env, sess=sess, target=True)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    if load_model:
        # Not worth using GPU for replaying model
        # config_replay = tf.ConfigProto(device_count={'GPU': 0})

        sess.run(tf.global_variables_initializer())

        saver.restore(sess, modelload_name)

        replay(dqnetwork, replay_count)

        exit()

    else:
        nn_layer_1_units, nn_layer_1_activation, gamma = dqnetwork.parameters()
        log_parameters()

    for n in range(N):
        logfile.flush()

        if eps > eps_min and len(memory) > minibatch_trigger:
            eps *= eps_decay

        # Play one game
        totalreward = play_one(env, dqnetwork, eps, gamma)
        totalrewards[n] = totalreward

        reward_avg_last50 = totalrewards[max(0, n - 50):(n + 1)].mean()
        reward_avg_last100 = totalrewards[max(0, n - 100):(n + 1)].mean()

        if n > 1 and n % 10 == 0:
            if save_model:
                saver.save(sess, modelsave_name, global_step=n)

            tx = time.time() - t0
            output = 'Episode: ' + str(n) + "\navg reward (last 100): " + str(reward_avg_last100)
            logfile.write('{}\nElapsed time : {}s\n\n'.format(output, round(tx, 2)))

        if reward_avg_last50 >= break_reward:
            break

logfile.close()
env.close()
sess.close()
