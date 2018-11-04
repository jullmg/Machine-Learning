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
import random
from collections import deque
#import matplotlib.pyplot as plt

# Pre-flight parameters
logfile_name = './LunarLander_Logs/LunarLander_Qlearn_test01.log'
modelsave_name = './LunarLander_Models/LunarLander_Q_Learning_test01'
modelload_name = './LunarLander_Models/LunarLander_Q_Learning_10-10'
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
# 1 to use gpu, 0 to use CPU
use_gpu = 0
config = tf.ConfigProto(device_count={'GPU': use_gpu})

CER = True
PER = True

nn_layer_1_activation = 'relu'
nn_layer_1_units = 512
nn_output_activation = 'linear'
nn_dropout = False
nn_dropout_factor = 0.95

tau = 0
tau_max = 5000

epochs = 1
break_reward = 215

lr = 0.001
N = 10000

eps = 1
eps_decay = 0.995
eps_min = 0.1

gamma = 0.99

# 20 semble optimal
minibatch_size = 20
memory = deque(maxlen=500000)

memory_size = 500000


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

        if state[6] == 1 and state[7] == 1 and state[4] < 0.1:
            action = 0
        else:
            action = dqnetwork.sample_action(state, eps)

        # state = np.array(state).reshape(-1, 8)
        # print('Network pred:', sess.run(dqnetwork.outputs, feed_dict={dqnetwork.inputs: state}))
        # print('Target  pred:', sess.run(dqnetwork_target.outputs, feed_dict={dqnetwork_target.inputs: state}))

        next_state, reward, done, info = env.step(action)
        totalreward += reward
        last_sequence = (state, action, reward, next_state, done)
        memory.append(last_sequence)
        per_memory.store(last_sequence)

        state = next_state

        if len(memory) > 500:
            #minibatch = random.sample(memory, minibatch_size)
            # Obtain random mini-batch from memory
            tree_idx, minibatch, ISWeights_mb = per_memory.sample(minibatch_size)

            # Combined Experience Replay
            if CER and PER:
                minibatch.append([last_sequence])

            elif CER and not PER:
                minibatch.append(last_sequence)


            dqnetwork.train(tree_idx, minibatch)

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
                # Single Qvalue predicted for prefered action
                self.target_Qvalue = tf.placeholder(tf.float32, [None], name="target_Qvalue")

                # The loss is modified because of PER
                self.absolute_errors = tf.abs(self.target_Qvalue - self.outputs)  # for updating Sumtree

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

    def train(self, tree_idx, batch):

        y = []

        x = [each[0][0] for each in batch]
        actions = [each[0][1] for each in batch]
        rewards = [each[0][2] for each in batch]
        next_states = [each[0][3] for each in batch]
        dones = [each[0][4] for each in batch]

        for n in range(len(batch)):
            target_Qvalue = rewards[n]

            next_state = np.array(next_states[n]).reshape(-1, 8)

            state = np.array(x[n]).reshape(-1, 8)

            if not dones[n]:
                #target = reward + gamma * np.max(self.predict(next_state))  # use np.amax?
                target_Qvalue = rewards[n] + gamma * np.max(sess.run(dqnetwork_target.outputs, feed_dict={dqnetwork_target.inputs:next_state})) # use np.amax?
                #print('target', target)

            #target_f = self.predict(state)
            target_f = sess.run(self.outputs, feed_dict={self.inputs:state})

            target_f[0][actions[n]] = target_Qvalue

            y.append(target_f)

        y = np.array(y).reshape(-1, 4)
        x = np.array(x).reshape(-1, 8)

        _, loss, target_Q, outputs, absolute_errors = sess.run([self.train_op, self.loss, self.target_Q, self.outputs, self.absolute_errors], feed_dict={self.inputs: x, self.target_Q: y, self.target_Qvalue: target_Qvalue})

        #print('absolute errors:', absolute_errors)

        # Update priority
        per_memory.batch_update(tree_idx, absolute_errors)

    def sample_action(self, s, eps):
        s = np.array(s).reshape(-1, 8)
        # np.random (0.01-0.99)
        #print(np.max(self.predict(s)))

        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            prediction = np.argmax(sess.run(self.outputs, feed_dict={self.inputs: s}))

            return prediction

class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0

    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """
    # Capacity est le nombre de feuilles
    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)



    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0

    """
    Update the leaf priority score and propagate the change through tree
    """

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]

        # Ici on ajoute la priorite sur la feuille
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:  # this method is faster than the recursive loop in the reference code

            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 

            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node

class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)

    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = [] # OK


        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment # OK

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1 # OK

        # Calculating the max_weight
        # La priorite minimale parmis toutes les feuilles divise par la priosite totale
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities on the tree
    """

    def batch_update(self, tree_idx, abs_errors):

        print(abs_errors)
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        exit()

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

###################FUNCTIONS&CLASSES############################################

# Reset the graph
tf.reset_default_graph()

# Instantiate DQNetwork
dqnetwork = DQNet(name='dqnetwork', env=env)

# Instantiate Target DQNetwork
dqnetwork_target = DQNet(name='dqnetwork_target', env=env, target=True)

# Instantiate PER memory
per_memory = Memory(memory_size)

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

with tf.Session(config=config) as sess:
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

logfile.close()
debugfile.close()
env.close()
