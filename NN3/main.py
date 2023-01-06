import numpy
import scipy.special
import csv
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import math

# define neural network class
class neuralNetwork:
    # define input values, in_nodes, hidden_nodes and out_nodes. Also learning rate = 0.2
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate=0.2):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate
        # random weights
        self.w1 = (numpy.random.rand(self.hnodes, self.inodes) - 1)
        self.w2 = (numpy.random.rand(self.onodes, self.hnodes) - 1)

    # train function
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # feedforward
        hidden_inputs = numpy.dot(self.w1, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        final_inputs = numpy.dot(self.w2, hidden_outputs)
        final_outputs = self.activation(final_inputs)
        # backpropogation
        output_errors = targets - final_outputs
        output_errors_delta = output_errors * final_outputs * (1 - final_outputs)

        hidden_errors = numpy.dot(self.w2.T, output_errors)
        hidden_errors_delta = hidden_errors * hidden_outputs * (1 - hidden_outputs)

        newW2 = numpy.dot(output_errors_delta, numpy.transpose(hidden_outputs))
        newW1 = numpy.dot(hidden_errors_delta, numpy.transpose(inputs))

        self.w1 += self.lr * newW1
        self.w2 += self.lr * newW2

    # test function
    def test(self, inputs_list):
        hidden_inputs = numpy.dot(self.w1, inputs_list)
        hidden_outputs = self.activation(hidden_inputs)
        final_inputs = numpy.dot(self.w2, hidden_outputs)
        final_outputs = self.activation(final_inputs)
        return final_outputs

    # activation function (sigmoid)
    def activation(self, num):
        return 1 / (1 + numpy.exp(-num))
    # activation function for backpropagation
    def deriv_activation(self, num):
        return num * (1 - num)

def main():
    # create new neural network
    input_nodes = 4
    hidden_nodes = 3
    output_nodes = 3

    learning_rate = 0.2

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    maxval = 0
    input = []
    output = []
    # open dataset
    with open('Iris.data') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            temp = []
            temp.append(float(row[0]))
            temp.append(float(row[1]))
            temp.append(float(row[2]))
            temp.append(float(row[3]))
            input.append(temp)
            output.append(row[4])

    # re-evaluate dataset numbers 0 < data < 1
    maxval = numpy.amax(input)
    input = input / maxval

    lb = preprocessing.LabelBinarizer()
    output = lb.fit_transform(output)

    # split data into test and train
    X_train, X_test, Y_train, Y_test = train_test_split(input, output, test_size=0.3, random_state=1)

    # train 1000 times
    for _ in range(1000):
        for i in range(len(X_train)):
            n.train(X_train[i], Y_train[i])
    # just normal round system
    def normal_round(final_outputs):
        if final_outputs - math.floor(final_outputs) < 0.5:
            return math.floor(final_outputs)
        return math.ceil(final_outputs)

    sumOfGoodTest = 0
    # test accuracy of neural network
    for i in range(len(X_test)):
        print(X_test[i], "Expected: ", Y_test[i])
        final_outputs = n.test(X_test[i])
        for j in range(0, 3):
            final_outputs[j] = normal_round(final_outputs[j])
        if (final_outputs == Y_test[i]).all():
            sumOfGoodTest += 1
        else:
            pass
        print("Result:  ", final_outputs)
    acccc = sumOfGoodTest / len(Y_test)
    acccc = acccc * 100
    print("Iris-setosa = 1. 0. 0.")
    print("Iris-versicolor  = 0. 1. 0.")
    print("Iris-verginica = 0. 0. 1.")
    print(f"Accuracy = {acccc}%")
    return 0


if __name__ == "__main__":
    main()
