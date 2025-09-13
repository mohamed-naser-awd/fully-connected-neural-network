from enums import StrEnum
from uuid import uuid4
from random import uniform
from typing import Callable, Self


class NodeLayer:
    activation_function: Callable
    nodes: list


class Node:
    class NodeType(StrEnum):
        INPUT = "input"
        HIDDEN = "hidden"
        OUTPUT = "output"

    node_type: NodeType
    input = 0
    output = 0

    def __init__(self, node_layer: NodeLayer) -> None:
        self.pk = uuid4().hex
        self.node_layer = node_layer


class BiasNode(Node):
    bias: float

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.bias = 0


class WeightedNode(BiasNode):
    weights: dict[str, float]

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.weights = {}

    def get_node_weight(self, node: Self):
        try:
            return self.weights[node.pk]
        except KeyError:
            return self.set_node_weight(node)

    def set_node_weight(self, node):
        weight = uniform(-0.1, 0.1)
        self.weights[node.pk] = weight
        return weight

    def set_info(self, prev_layer: NodeLayer) -> None:
        for node in prev_layer.nodes:
            self.set_node_weight(node)


class InputNode(Node):
    node_type = Node.NodeType.INPUT


class HiddenNode(WeightedNode):
    node_type = Node.NodeType.HIDDEN


class OutputNode(WeightedNode):
    node_type = Node.NodeType.OUTPUT
