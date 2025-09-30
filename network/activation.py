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


def softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=0, keepdims=True)
    e = np.exp(x - m)
    return e / (np.sum(e, axis=0, keepdims=True) + 1e-12)