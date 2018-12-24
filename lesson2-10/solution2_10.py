import os
import numpy as np
from numpy import genfromtxt, fliplr, flipud
import matplotlib.pyplot as plt

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)
my_data = genfromtxt(os.getcwd() + '/lesson2-10/data.csv', delimiter=',')

X = np.array(my_data[:, [0, 1]])
y = np.array(my_data[:, 2])
num_epochs = 25


def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return stepFunction((np.matmul(X, W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.


def perceptronStep(X, y, W, b, learn_rate=0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i], W, b)
        # if y[i]-y_hat == 0 then this means that prediction classified the (p,q) point correctly
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.


def trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=num_epochs):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0] / W[1]+1.5, -b / W[1]))
    return boundary_lines


boundary_lines = np.fliplr(trainPerceptronAlgorithm(X, y))
for i in range(num_epochs):
    if i == num_epochs-1:
        plt.plot(boundary_lines[i], 'k-', zorder=1)
        break
    print(i)
    print(boundary_lines[i])
    plt.plot(boundary_lines[i], 'g--',  zorder=1)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)

color = 'blue'
for i, y_i in enumerate(y):
    if y_i != 0:
        color = 'blue'
    else:
        color = 'red'
    plt.scatter(X[i][0], X[i][1], c=color, edgecolors='k',  zorder=2)


plt.show()
