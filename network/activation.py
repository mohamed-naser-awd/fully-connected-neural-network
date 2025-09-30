import math
import numpy as np


def sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0, dtype=x.dtype)


def softmax(z: np.ndarray) -> np.ndarray:
    # z.shape = (classes, 1)
    z_shift = z - np.max(z, axis=0, keepdims=True)  # subtract max per column
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
