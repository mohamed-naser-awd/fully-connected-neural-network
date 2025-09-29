import math


def relu(x: float):
    return max(0, x)


def sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def softmax(z: list[float]) -> list[float]:
    exp_z = [math.exp(v - max(z)) for v in z]
    sum_exp = sum(exp_z)
    return [v / sum_exp for v in exp_z]
