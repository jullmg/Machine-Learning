import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from dense_network import Dense_NN

mnist = input_data.read_data_sets("mnist_data", one_hot = True)

network = Dense_NN()

def train(network):
    network.layer_1






