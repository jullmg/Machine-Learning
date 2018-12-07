import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from dense_network import Dense_NN

batch_size = 50
epoch = 5
iteration = round(60000 / batch_size)

sess = tf.Session()

# training set of 60,000 examples, and a test set of 10,000 examples
mnist = input_data.read_data_sets("mnist_data", one_hot=True)

network = Dense_NN(sess)

sess.run(tf.global_variables_initializer())

for _ in range(epoch):
    for _ in range(iteration):
        # Return a tuple [0] = image [1] = label (answer)
        data = mnist.train.next_batch(batch_size)
        loss, prediction = network.train(data)










