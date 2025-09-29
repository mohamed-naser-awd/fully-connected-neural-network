from .node import Node
from enum import Enum
from typing import Self
from .activation import relu, softmax


class Layer:
    nodes: list[Node]

    class LayerType(Enum):
        INPUT = "input"
        HIDDEN = "hidden"
        OUTPUT = "output"

    default_layer_type = LayerType.HIDDEN

    def __init__(self, layer_type: LayerType = None):
        self.layer_type = layer_type or self.default_layer_type
        self.nodes = []

    def set_nodes(self, nodes: list[Node]):
        self.nodes = nodes

    def init_nodes(self, size: int):
        self.nodes = [Node(bias=None, layer=self) for _ in range(size)]

    def forward(self, previous_layer: Self):
        for node in self.nodes:
            self.forward_node(node, previous_layer)

    def forward_node(self, node: Node, previous_layer: Self):
        node_activation_input = 0

        for previous_node in previous_layer.nodes:
            weight = node.get_weight(previous_node)
            node_activation_input += previous_node.activated_output * weight

        node.activation_input = node_activation_input
        node.raw_output = node.activation_input + node.bias

        if self.layer_type != self.LayerType.OUTPUT:
            node.activated_output = self.activate_output(node.raw_output)

    def activate_output(self, output: float):
        activation_function_map: dict[Enum, callable] = {
            self.LayerType.HIDDEN: relu,
            self.LayerType.INPUT: lambda x: x,
        }
        activation_function = activation_function_map[self.layer_type]
        return activation_function(output)


class InputLayer(Layer):
    default_layer_type = Layer.LayerType.INPUT

    def set_input(self, input: list[int]):
        for i, node in zip(input, self.nodes):
            node.activation_input = i
            node.activated_output = i
            node.raw_output = i


class OutputLayer(Layer):
    default_layer_type = Layer.LayerType.OUTPUT

    def get_output(self):
        node_tuple_set = []

        results = softmax([node.raw_output for node in self.nodes])

        for node, result in zip(self.nodes, results):
            node.activated_output = result

        for idx, result in enumerate(results):
            node_tuple_set.append((idx, result))

        sorted_tuple_set = sorted(
            node_tuple_set, key=lambda prediction_obj: prediction_obj[1]
        )
        prediction = sorted_tuple_set[-1]
        return prediction
