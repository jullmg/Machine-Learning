import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from dense_network import Dense_NN

batch_size = 50
epoch_num = 2

# training set of 55,000 examples, and a test set of 10,000 examples
mnist = input_data.read_data_sets("./mnist_data/", one_hot=True)

# mnist_dataset = tf.data.Dataset()
# batch = mnist_dataset.batch(10)

iteration_num = round(mnist.train.num_examples / batch_size)

sess = tf.Session()

network = Dense_NN(sess)

sess.run(tf.global_variables_initializer())

# Training phase
print('Training phase')
for epoch in range(epoch_num):
    for iteration in range(iteration_num):
        # Return a tuple [0] = image [1] = label (answer)
        data = mnist.train.next_batch(batch_size)
        x_data, y_data = data
        loss, output = network.train(x_data, y_data)
        # print(prediction[0])
        # print(np.argmax(prediction[0]))
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

        accuracy = (good_guesses / batch_size)*100

        # print('Predictions', predictions)
        # print('Labels     ', labels)
        if iteration % 100 == 0:
            print('Epoch: {}, Iteration: {}, Accuracy : {}'.format(epoch, iteration, accuracy))

# Testing phase
num_tests = mnist.test.images.shape[0]

print('Testing phase')
print('Total Test Examples in Dataset = ' + str(num_tests))

x_test = mnist.test.images
y_test = mnist.test.labels

for index in range(num_tests):
    x = x_test[index]
    x = np.reshape(x, [-1, 784])

    test_prediction = network.test(x)
    test_answer = np.argmax(y_test[index])
    # print('p: {} a: {}'.format(test_prediction, test_answer))








