from .node import Node
from enum import Enum
from typing import Self
from .activation import relu, sigmoid


class Layer:
    nodes: list[Node]

    class LayerType(Enum):
        INPUT = "input"
        HIDDEN = "hidden"
        OUTPUT = "output"

    def __init__(self, layer_type: LayerType):
        self.layer_type = layer_type
        self.nodes = []

    def set_nodes(self, nodes: list[Node]):
        self.nodes = nodes

    def init_nodes(self, size: int):
        self.nodes = [Node(bias=None) for _ in range(size)]

    def forward(self, previous_layer: Self):
        for node in self.nodes:
            self.forward_node(node, previous_layer)

    def get_output(self):
        node_tuple_set = [
            (idx, self.activate_output(node.activation_input))
            for idx, node in enumerate(self.nodes)
        ]

        sorted_tuple_set = sorted(
            node_tuple_set, key=lambda prediction_obj: prediction_obj[1]
        )
        prediction = sorted_tuple_set[-1]
        return prediction

    def forward_node(self, node: Node, previous_layer: Self):
        node_output = 0
        node_output += node.bias

        for previous_node in previous_layer.nodes:
            connection_weight = node.get_weight(previous_node)
            node_output += connection_weight * previous_node.activation_input

        node.activation_input = self.activate_output(node_output)

    def activate_output(self, output: float):
        activation_function_map: dict[Enum, callable] = {
            self.LayerType.HIDDEN: relu,
            self.LayerType.INPUT: lambda x: x,
            self.LayerType.OUTPUT: sigmoid,
        }
        activation_function = activation_function_map[self.layer_type]
        return activation_function(output)

    def set_input(self, list: list[int]):
        for input, node in zip(list, self.nodes):
            node.activation_input = input
