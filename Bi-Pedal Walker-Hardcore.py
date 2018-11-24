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

suffix = '01'
logfile_name = './Bipedal-Hardcore_Logs/Bipedal-{}.log'.format(suffix)
modelsave_name = './Bipedal-Hardcore_Models/Bipedal-{}'.format(suffix)
modelload_name = './Bipedal-Hardcore_Models/Bipedal-{}-11660'.format(suffix)
scoressave_name = './Bipedal-Hardcore_Logs/Score-{}.npy'.format(suffix)

debug_name = './Bipedal-Hardcore_Logs/Bipedal_Debug.log'
debugfile = open(debug_name, 'w')

try:
    logfile = open(logfile_name, 'w')

except FileNotFoundError:
    os.mknod(FileNotFoundError.filename)
    logfile = open(FileNotFoundError.filename, 'w')

load_model = True

if not load_model:
    logfile.write('With Dropout\n')

replay_count = 1000
render = False

# 1 to use gpu 0 to use CPU
use_gpu = 0
config = tf.ConfigProto(device_count={'GPU': use_gpu})

CER = True

tau = 0
tau_max = 5000

lr_actor = 1e-4
lr_critic = 1e-3
N = 100000
test_num = 10
gamma = 0.99
epochs = 1

# Dropout
nn_dropout = True
keep_prob = 0.85

minibatch_size = 64
memory = deque(maxlen=750000)
pre_train_steps = 5000

env = gym.make('BipedalWalkerHardcore-v2')
input_size = env.observation_space.shape[0] #24
output_size = 4
t0 = time.time()

# Ornstein-Uhlenbeck (Random noise) process for action exploration
mu=0
theta=0.15
sigma=0.2
noise = np.ones(output_size) * mu

###################FUNCTIONS&CLASSES############################################


def play_one(env, model, gamma):
    state = env.reset()
    done = False
    totalreward = 0
    global tau

    while not done:
        state = np.array(state).reshape(-1, input_size)

        action = sess.run(nn_actor.outputs, feed_dict={nn_actor.state_inputs: state})
        noise = ounoise()
        # print('noise: ', noise)

        action += noise
        action = np.clip(action, -1, 1)
        action = action[0]

        next_state, reward, done, info = env.step(action)
        #print('reward:', reward)

        totalreward += reward
        last_sequence = (state, action, reward, next_state, done)
        memory.append(last_sequence)

        state = next_state

        if len(memory) > pre_train_steps:
            minibatch = random.sample(memory, minibatch_size)

            # Combined Experience Replay
            if CER:
                minibatch.append(last_sequence)

            train(minibatch)

        tau += 1

        if tau > tau_max:
            update_target_actor, update_target_critic = update_target_graphs()
            sess.run([update_target_actor, update_target_critic])
            tau = 0

        if render == True:
            env.render()

        if done:
            # Re-iniitialize the random process when an episode ends
            noise = np.ones(output_size) * mu
            break

    return totalreward


def replay(model, num, test=False):
    totalrewards = np.empty(replay_count)

    for game in range(num):
        state = env.reset()
        game_score = 0
        done = False

        while not done:
            state = np.array(state).reshape(-1, input_size)

            action = sess.run(nn_actor.outputs, feed_dict={nn_actor.state_inputs: state})[0]
            # print(action)

            state, reward, done, infow = env.step(action)
            game_score += reward

            if not test:
                env.render()

        if test:
            logfile.write('Score: {}\n'.format(game_score))
            logfile.flush()
        else:
            print('Game {} score: {}'.format(game, game_score))

        totalrewards[game] = game_score
        average_10 = totalrewards[max(0, game - 10):(game + 1)].mean()
        output = 'Average score last 10: {}\n'.format(average_10)

        if game % 10 == 0 and game > 0:
           print(output)

    if test:
        logfile.write(output)
        logfile.flush()
        return average_10


def log_parameters():
    logfile.write('\nEpochs: {}\nGamma: {}\nActor_Learning Rate: {}, Critic_Learning Rate: {}\nMiniBatch_Size: {} \n'.format(epochs, gamma, lr_actor, lr_critic, minibatch_size))
    logfile.write('OU noise theta: {}, sigma: {}\n'.format(theta, sigma))
    logfile.write('With CER\n') if CER else logfile.write('No CER\n')
    # logfile.write(
    #     'Layer 1 : units: {} activation: {}\n'.format(nn_l1_units,nn_layer_1_activation))
    if nn_dropout:
        logfile.write('Dropout on, keep prob: {}\n\n'.format(keep_prob))
    else:
        logfile.write('No Dropout\n\n')

    logfile.flush()


def update_target_graphs():
    # Get the parameters of our Network
    from_vars_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "nn_actor")
    from_vars_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "nn_critic")

    # Get the parameters of our Target_network
    to_vars_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "nn_actor_target")
    to_vars_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "nn_critic_target")

    op_holder_actor = []
    op_holder_critic = []

    # Update our Actor target_network parameters with Network parameters
    for from_vars_actor, to_vars_actor in zip(from_vars_actor, to_vars_actor):
        op_holder_actor.append(to_vars_actor.assign(from_vars_actor))

    # Update our Critic target_network parameters with Network parameters
    for from_vars_critic, to_vars_critic in zip(from_vars_critic, to_vars_critic):
        op_holder_critic.append(to_vars_critic.assign(from_vars_critic))

    return op_holder_actor, op_holder_critic


def ounoise(mu=0, theta=0.15, sigma=sigma):
    global noise
    x = noise
    dx = theta * (mu - x) + sigma * np.random.randn(len(x))
    noise = x + dx
    return noise


def train(minibatch):
    state_batch = np.asarray([data[0] for data in minibatch])
    action_batch = np.asarray([data[1] for data in minibatch])
    reward_batch = np.asarray([data[2] for data in minibatch])
    next_state_batch = np.asarray([data[3] for data in minibatch])
    done_batch = np.asarray([data[4] for data in minibatch])

    next_action_batch = sess.run(nn_actor_target.outputs, feed_dict={nn_actor_target.state_inputs: next_state_batch})

    q_value_batch = sess.run(nn_critic_target.output, feed_dict={nn_critic_target.state_inputs: next_state_batch, nn_critic_target.action_inputs: next_action_batch})
    # print('q_value[0]', q_value_batch[0])

    # Discounted QValue (reward + gamma*Qvalue)
    y_batch = []

    # If done append reward only else append Discounted Qvalue
    for i in range(len(minibatch)):
        if done_batch[i]:
            y_batch.append(reward_batch[i])
        else:
            y_batch.append(reward_batch[i] + gamma * q_value_batch[i])

    state_batch = state_batch.reshape(-1, input_size)
    action_batch = action_batch.reshape(-1, output_size)

    y_batch = np.asarray(y_batch)
    y_batch = y_batch.reshape(-1, 1)

    # Train op
    _, loss, outputs = sess.run([nn_critic.train_op, nn_critic.loss, nn_critic.output], feed_dict={nn_critic.state_inputs: state_batch, nn_critic.action_inputs: action_batch, nn_critic.target_Q: y_batch, nn_critic.keep_prob: keep_prob})

    action_batch_for_grads = sess.run(nn_actor.outputs, feed_dict={nn_actor.state_inputs: state_batch})
    q_gradient_batch = sess.run(nn_critic.action_gradients, feed_dict={
            nn_critic.state_inputs: state_batch,
            nn_critic.action_inputs: action_batch_for_grads
        })[0]

    _, par_grads, train_vars = sess.run([nn_actor.optimizer, nn_actor.parameters_gradients, nn_actor.train_vars], feed_dict={nn_actor.state_inputs: state_batch, nn_actor.q_gradient_inputs: q_gradient_batch, nn_actor.keep_prob: keep_prob})
    # print('par_grads', len(par_grads[3]))
    # print(len(train_vars[3]))


class ActorNet:
    def __init__(self, name, env=None, target=False):
        self.name = name
        self.env = env
        self.target = target

        # Actor build
        with tf.variable_scope(self.name):
            self.state_inputs = tf.placeholder(tf.float32, [None, 24], name="state_inputs")
            self.keep_prob = tf.placeholder(tf.float32)


            self.actor_l1 = tf.layers.dense(self.state_inputs, 256, activation=tf.nn.relu)

            self.dropout_l1 = tf.nn.dropout(self.actor_l1, keep_prob=self.keep_prob)

            self.actor_l2 = tf.layers.dense(self.actor_l1, 512, activation=tf.nn.relu)

            self.dropout_l2 = tf.nn.dropout(self.actor_l2, keep_prob=self.keep_prob)

            self.outputs = tf.layers.dense(self.actor_l2, output_size, activation=tf.nn.tanh)

            # Training stage
            if not target:
                self.q_gradient_inputs = tf.placeholder(tf.float32, [None, output_size])
                self.train_vars = tf.trainable_variables()

                self.parameters_gradients = tf.gradients(self.outputs, self.train_vars,
                                                         -self.q_gradient_inputs)

                self.optimizer = tf.train.AdamOptimizer(lr_actor).apply_gradients(
                     zip(self.parameters_gradients, self.train_vars))


class CriticNet:
    def __init__(self, name, env=None, target=False):
        self.name = name
        self.env = env

        # Critic build

        with tf.variable_scope(self.name):
            self.state_inputs = tf.placeholder(tf.float32, [None, 24], name="critic_state_inputs")
            self.action_inputs = tf.placeholder(tf.float32, [None, 4], name="critic_action_inputs")
            self.keep_prob = tf.placeholder(tf.float32)

            self.state_l1 = tf.layers.dense(self.state_inputs, 256, activation=tf.nn.relu)
            self.dropout_state_l1 = tf.nn.dropout(self.state_l1, keep_prob=self.keep_prob)

            self.action_l1 = tf.layers.dense(self.action_inputs, 256, activation=tf.nn.relu)
            self.dropout_action_l1 = tf.nn.dropout(self.action_l1, keep_prob=self.keep_prob)

            self.mergedlayer = tf.concat([self.state_l1, self.action_l1], 1)

            self.mergedlayer_l1 =  tf.layers.dense(self.mergedlayer, 512, activation=tf.nn.relu)
            self.dropout_mergedlayer = tf.nn.dropout(self.mergedlayer_l1, keep_prob=self.keep_prob)

            self.output = tf.layers.dense(self.mergedlayer_l1, 1)

            if not target:
                # Training stage
                self.target_Q = tf.placeholder(tf.float32, [None, 1], name="target_Q")

                #self.loss = tf.losses.mean_squared_error(self.target_Q, self.outputs)
                self.loss = tf.losses.huber_loss(self.target_Q, self.output)

                self.train_op = tf.train.AdamOptimizer(learning_rate=lr_critic).minimize(self.loss)

                self.action_gradients = tf.gradients(self.output, self.action_inputs)
                # self.action_gradients = tf.gradients(self.output, self.state_inputs)

###################FUNCTIONS&CLASSES############################################


tf.reset_default_graph()

# Instantiate Actor Network
nn_actor = ActorNet(name='nn_actor', env=env)

# Instantiate Actor's Target Network
nn_actor_target = ActorNet(name='nn_actor_target', env=env, target=True)

# Instantiate Critic Network
nn_critic = CriticNet(name='nn_critic', env=env)

# Instantiate Critic's Target Network
nn_critic_target = CriticNet(name='nn_critic_target', env=env, target=True)

saver = tf.train.Saver()

if load_model:
    # Not worth using GPU for replaying model
    config_replay = tf.ConfigProto(device_count={'GPU': 0})

    with tf.Session(config=config_replay) as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, modelload_name)

        replay(nn_actor, replay_count)

        exit()

totalrewards = np.empty(N)

log_parameters()

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    scores_for_graph = []

    for n in range(N):
        logfile.flush()

        # Play one game
        play_one(env, nn_actor, gamma)

        if n > 1 and n % 10 == 0:
            # Testing model
            logfile.write('Testing from game {}\n'.format(n))
            avg_score = replay(nn_actor, test_num, test=True)

            scores_for_graph.append(avg_score)
            saved_scores = np.array(scores_for_graph)
            np.save(scoressave_name, saved_scores)

            tx = time.time() - t0
            logfile.write('Elapsed time : {}s\n\n'.format(round(tx, 2)))

            if avg_score > 0:
                saver.save(sess, modelsave_name, global_step=n)

        # if n > 1 and n % 1000 == 0:
        #     saver.save(sess, modelsave_name, global_step=n)


logfile.close()
debugfile.close()
env.close()
