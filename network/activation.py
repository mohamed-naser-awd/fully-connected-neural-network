import math


def relu(x: float):
    return max(0, x)


def sigmoid(x: float):
    return 1 / (1 + math.exp(-x))
