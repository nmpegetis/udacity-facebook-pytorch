import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.


def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -1 * np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))


Y = [1, 1, 0]
P = [0.8, 0.7, 0.1]
result = cross_entropy(Y, P)

print(
    "The result of the above Y[1,1,0] and P[0.8, 0.7, 0.1] should be 0.69, and the calculated was "+str(result))

Y = [0, 0, 1]
P = [0.8, 0.7, 0.1]
result = cross_entropy(Y, P)

print(
    "The result of the above Y[0, 0, 1] and P[0.8, 0.7, 0.1] should be 5.12, and the calculated was "+str(result))
