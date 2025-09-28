from uuid import uuid4
from random import uniform
from typing import Self


class Node:
    id: str
    bias: float
    weights: dict[str, float]
    activation_input: float

    def __init__(self, id=None, bias=None, weights=None) -> None:
        self.id = id or uuid4().hex
        self.bias = bias or uniform(-0.01, 0.01)
        self.weights = weights or {}

    def __str__(self):
        return f"Node(id={self.id}, bias={self.bias})"

    def get_weight(self, node: Self):
        weight = self.weights.get(node.id)

        if weight is None:
            weight = self.set_weight(node)

        return weight

    def set_weight(self, node: Self):
        weight = uniform(-0.01, 0.01)
        self.weights[node.id] = weight
        return weight

    def set_activation_input(self, activation_input):
        self.activation_input = activation_input
