from network.node import Node
from .layer import Layer, InputLayer
from functools import cached_property


class Network:
    name: str
    layers: list[Layer]

    def __init__(self, name="MnistNetwork"):
        self.name = name
        self.layers = []

    def __str__(self):
        return f"Network(name={self.name})"

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def predict(self, input: list[int]):
        self.input_layer.set_input(input)

        for idx, layer in enumerate(self.layers):
            try:
                next_layer = self.layers[idx + 1]
                next_layer.forward(layer)
            except IndexError:
                output_layer = layer
                return output_layer.get_output()

    @property
    def input_layer(self) -> InputLayer:
        return self.layers[0]

    @property
    def output_layer(self):
        return self.layers[-1]

    @property
    def hidden_layers(self):
        return self.layers[1:-1]

    @cached_property
    def nodes(self) -> list[Node]:
        nodes: list[Node] = []

        for layer in self.layers:
            nodes.extend(layer.nodes)

        return nodes

    @cached_property
    def node_id_map(self):
        node_id_map = {}

        for node in self.nodes:
            node_id_map[node.id] = node

        return node_id_map
