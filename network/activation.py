from math import exp
from typing import List


def relu(x: float) -> float:
    return max(0, x)


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def tanh(x: float) -> float:
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))


def softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [exp(x - m) for x in xs]
    denom = sum(exps)
    return [e / denom for e in exps]
