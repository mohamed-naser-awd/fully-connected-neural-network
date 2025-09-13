from math import exp


def relu(x: float) -> float:
    return max(0, x)


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def tanh(x: float) -> float:
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))


def softmax(x: float) -> float:
    return exp(x) / sum(exp(x))
