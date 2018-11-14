'''
state = [
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
]


self.joints[0].motorSpeed     = float(SPEED_HIP     * np.sign(action[0]))

self.joints[1].motorSpeed     = float(SPEED_KNEE    * np.sign(action[1]))

self.joints[2].motorSpeed     = float(SPEED_HIP     * np.sign(action[2]))

self.joints[3].motorSpeed     = float(SPEED_KNEE    * np.sign(action[3]))



'''

import gym
import numpy as np
import tensorflow as tf
import time
import os
import random
from collections import deque

logfile_name = './testdelete.log'
modelsave_name = './LunarLander_Models/LunarLander_Qlearn_sanspattes-01'
modelload_name = './LunarLander_Models/LunarLander_Q_Learning_10-10'
debug_name = './LunarLander_Logs/LunarLander_Qlearn_debug_01.log'

try:
    logfile = open(logfile_name, 'w')
    debugfile = open(debug_name, 'w')
except FileNotFoundError:
    os.mknod(FileNotFoundError.filename)
    logfile = open(FileNotFoundError.filename, 'w')

logfile.write('\n')


save_model = False
load_model = False
replay_count = 1000
render = True
max_game_step = 650
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

eps = 0.95
eps_decay = 0.995
eps_min = 0.1

minibatch_size = 32
memory = deque(maxlen=500000)

env = gym.make('BipedalWalker-v2')
input_size = env.observation_space.shape[0] #24
output_size = 4
t0 = time.time()

# Ornstein-Uhlenbeck (Random noise) process for action exploration
mu=0
theta=0.15
sigma=0.5
noise = np.ones(output_size) * mu

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

def play_one(env, model, gamma):
    state = env.reset()
    done = False
    totalreward = 0
    global tau

    # while not done:
    for t in range(max_game_step):

        state = np.array(state).reshape(-1, input_size)

        action = sess.run(nn_actor.actor_outputs, feed_dict={nn_actor.state_inputs: state})
        noise = ounoise()
        print('noise', noise)

        # print('action', action)
        action += noise
        action = np.clip(action, -1, 1)
        action = action[0]

        next_state, reward, done, info = env.step(action)
        #print('reward:', reward)

        totalreward += reward
        last_sequence = (state, action, reward, next_state, done)
        memory.append(last_sequence)

        state = next_state

        if len(memory) > 10000:
            minibatch = random.sample(memory, minibatch_size)

            # Combined Experience Replay
            if CER:
                minibatch.append(last_sequence)

            nn_critic.train(minibatch)
            nn_actor.train(minibatch)
            exit()

        tau += 1

        if tau > tau_max:
            update_target = update_target_graph()
            sess.run(update_target)
            tau = 0
            print("Model updated")

        if render == True:
            env.render()

        if done:
            # Re-iniitialize the random process when an episode ends
            noise = np.ones(output_size) * mu
            break


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
    # Get the parameters of our Network
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "network")

    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "network_target")

    op_holder = []

    # Update our target_network parameters with Network parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))

    return op_holder

def ounoise(mu=0, theta=0.15, sigma=0.5):
    global noise
    x = noise
    print('x', x)
    dx = theta * (mu - x) + sigma * np.random.randn(len(x))
    noise = x + dx
    return noise


class Net:
    def __init__(self, name, env=None, target=False, actor=False, critic=False):
        self.name = name
        self.env = env
        self.actor = actor

        # Actor build
        if actor:
            with tf.variable_scope(self.name):
                self.state_inputs = tf.placeholder(tf.float32, [None, input_size], name="state_inputs")

                self.actor_l1 = tf.layers.dense(self.state_inputs, 256, activation=tf.nn.relu)

                self.actor_l2 = tf.layers.dense(self.actor_l1, 512, activation=tf.nn.relu)

                self.actor_outputs = tf.layers.dense(self.actor_l2, 4, activation=tf.nn.tanh)

                if not target:
                    self.q_gradient_inputs = tf.placeholder(tf.float32, [None, input_size])

                    self.parameters_gradients = tf.gradients(self.actor_outputs, self.state_inputs,
                                                             -self.q_gradient_inputs)
                    # self.optimizer = tf.train.AdamOptimizer(lr).apply_gradients(
                    #     zip(self.parameters_gradients, self.net))


        # Critic build
        if critic:
            with tf.variable_scope(self.name):
                self.critic_state_inputs = tf.placeholder(tf.float32, [None, 24], name="critic_state_inputs")

                self.critic_action_inputs = tf.placeholder(tf.float32, [None, 4], name="critic_action_inputs")

                self.critic_state_l1 = tf.layers.dense(self.critic_state_inputs, 1024, activation=tf.nn.relu)

                self.critic_action_l1 = tf.layers.dense(self.critic_action_inputs, 1024, activation=tf.nn.relu)

                self.mergedlayer = tf.concat([self.critic_state_l1, self.critic_action_l1], 1)

                self.mergedlayer_l1 =  tf.layers.dense(self.mergedlayer, 2048, activation=tf.nn.relu)

                self.critic_output = tf.layers.dense(self.mergedlayer_l1, 1)

                if not target:
                    # For training critic
                    self.target_Q = tf.placeholder(tf.float32, [None, 1], name="target_Q")

                    #self.loss = tf.losses.mean_squared_error(self.target_Q, self.outputs)
                    self.critic_loss = tf.losses.huber_loss(self.target_Q, self.critic_output)

                    self.critic_train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.critic_loss)

                    # Gradient Clipping -5,5
                    # self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                    # self.gvs = self.optimizer.compute_gradients(self.loss)
                    # self.capped_gvs = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in self.gvs]
                    # self.critic_train_op = self.optimizer.apply_gradients(self.capped_gvs)

    def train(self, data):
        if self.actor:

            # Actor Training
            _, actor_outputs, actor_corrected_action = sess.run([nn_actor.actor_train_op, nn_actor.actor_outputs],
                                                                feed_dict={nn_actor.actor_inputs: x,
                                                                           nn_actor.actor_qvalue_input: TDerrors})


        else:

            x = []
            actions = []
            y = []

            for state, action, reward, next_state, done in data:
                x.append(state)
                actions.append(action)
                target_Qvalue = reward


                #print('reward', reward)

                next_state = np.array(next_state).reshape(-1, input_size)

                state = np.array(state).reshape(-1, input_size)
                action = np.array(action).reshape(-1, output_size)


                if not done:
                    target_action = sess.run(nn_actor_target.actor_outputs, feed_dict={nn_actor_target.state_inputs: state})

                    target_prediction = sess.run(nn_critic_target.critic_output, feed_dict={nn_critic_target.critic_state_inputs: next_state, nn_critic_target.critic_action_inputs: target_action})
                    target_Qvalue = reward + gamma * target_prediction
                    print('targetQ: ', target_Qvalue)

                Qvalue = sess.run(self.critic_output, feed_dict={self.critic_state_inputs: state, self.critic_action_inputs: action})

                y.append(target_Qvalue)

            x = np.array(x).reshape(-1, input_size)
            y = np.array(y).reshape(-1, 1)


            # Critic Training
            _, loss, outputs = sess.run([self.critic_train_op, self.critic_loss, self.critic_output], feed_dict={self.critic_state_inputs: x, self.critic_action_inputs: actions, self.target_Q: y})



###################FUNCTIONS&CLASSES############################################

tf.reset_default_graph()

# Instantiate Actor Network
nn_actor = Net(name='nn_actor', env=env, actor=True)

# Instantiate Actor's Target Network
nn_actor_target = Net(name='nn_actor_target', env=env, target=True, actor=True)

# Instantiate Critic Network
nn_critic = Net(name='nn_critic', env=env, critic=True)

# Instantiate Critic's Target Network
nn_critic_target = Net(name='nn_critic_target', env=env, target=True, critic=True)

#saver = tf.train.Saver()

totalrewards = np.empty(N)

#log_parameters()

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    for n in range(N):
        logfile.flush()

        # Play one game
        totalreward = play_one(env, nn_actor, gamma)
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
