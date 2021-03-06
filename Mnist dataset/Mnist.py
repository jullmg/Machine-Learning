import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from dense_network import Dense_NN
from conv_network import Conv_NN

use_conv = True

batch_size = 350
epoch_num = 15
dropout = 0.9

# 1 to use gpu 0 to use CPU
use_gpu = 1
config = tf.ConfigProto(device_count={'GPU': use_gpu})

# training set of 55,000 examples, and a test set of 10,000 examples
mnist = input_data.read_data_sets("./mnist_data/", one_hot=True)

# mnist_dataset = tf.data.Dataset()
# batch = mnist_dataset.batch(10)

iteration_num = round(mnist.train.num_examples / batch_size)

sess = tf.Session(config=config)

if use_conv:
    network = Conv_NN(sess)
else:
    network = Dense_NN(sess)


sess.run(tf.global_variables_initializer())

# Training phase
print('Training phase')
for epoch in range(epoch_num):
    for iteration in range(iteration_num):
        # Return a tuple [0] = image [1] = label (answer)
        data = mnist.train.next_batch(batch_size)
        x_data, y_data = data

        if use_conv:
            x_data = np.reshape(x_data, [-1, 28, 28, 1])

        loss, output = network.train(x_data, y_data, dropout)

        labels = []
        predictions = []
        good_guesses = 0

        for i in output:
            prediction = np.argmax(i)
            predictions.append(prediction)

        for i in y_data:
            for index in range(10):
                if i[index] == 1:
                    label = index
                    labels.append(label)

        for i in range(batch_size):
            if predictions[i] == labels[i]:
                good_guesses += 1

        accuracy = round(((good_guesses / batch_size)*100), 2)

        # print('Predictions', predictions)
        # print('Labels     ', labels)
        if iteration % 100 == 0:
            print('Epoch: {}, Iteration: {}, Accuracy : {}'.format(epoch + 1, iteration, accuracy))

# Testing phase
num_tests = mnist.test.images.shape[0]
good_guesses_test = 0

print('Testing phase')
print('Total Test Examples in Dataset = ' + str(num_tests))

x_test = mnist.test.images
y_test = mnist.test.labels

for index in range(num_tests):
    x = x_test[index]
    if use_conv :
        x = np.reshape(x, [-1, 28, 28, 1])
    else:
        x = np.reshape(x, [-1, 784])

    test_prediction = network.test(x)
    test_answer = np.argmax(y_test[index])

    # print('p: {} a: {}'.format(test_prediction, test_answer))
    if test_prediction == test_answer:
        good_guesses_test += 1

print('Test right answers', good_guesses_test)

test_accuracy = round(good_guesses_test / num_tests * 100, 2)
error_rate = round(100 - test_accuracy, 2)

print('Test Accuracy = {}%\nError rate = {}%'.format(test_accuracy, error_rate))







