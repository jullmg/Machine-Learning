import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from dense_network import Dense_NN

batch_size = 10
epoch = 1

mnist = input_data.read_data_sets("mnist_data", one_hot = True)

# Return a tuple [0] = image [1] = label (answer)
data = mnist.train.next_batch(batch_size)

network = Dense_NN()

for i in range(epoch):
    network.train(data)


print(y_batch)



