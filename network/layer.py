from enums import StrEnum
from .activation import relu, softmax
from .node import Node, InputNode, HiddenNode, OutputNode
from uuid import uuid4
import typing


class Layer:
    class LayerType(StrEnum):
        INPUT = "input"
        HIDDEN = "hidden"
        OUTPUT = "output"

    activation_function: typing.Callable
    layer_type: LayerType
    nodes: list[Node]

    def __init__(self, pk=None, size=None) -> None:
        self.pk = pk if pk is not None else uuid4().hex
        self.nodes = [] if size is None else self.generate_nodes(size)

    def generate_nodes(self, size: int) -> list[Node]:
        klass = self.get_node_cls()
        return [klass(self) for _ in range(size)]

    @classmethod
    def get_node_cls(cls) -> type[Node]:
        origin = typing.get_args(cls.__annotations__.get("nodes"))[0]
        return origin


class InputLayer(Layer):
    activation_function = lambda x: x
    layer_type = Layer.LayerType.INPUT
    nodes: list[InputNode]

    def set_nodes_output(self, input: list[float]) -> None:
        assert len(self.nodes) == len(
            input
        ), f"Input length {len(input)} must match number of nodes {len(self.nodes)}"

        for node, input_value in zip(self.nodes, input):
            node.output = input_value


class StaticLayer(Layer):
    def set_info(self, prev_layer: Layer) -> None:
        for node in self.nodes:
            node.set_info(prev_layer)


class HiddenLayer(StaticLayer):
    activation_function = lambda self, x: relu(x)
    layer_type = Layer.LayerType.HIDDEN
    nodes: list[HiddenNode]

    def compute(self, prev_layer: Layer) -> None:
        for node in self.nodes:
            node_output = node.bias
            for input_node in prev_layer.nodes:
                node_output += input_node.output * node.get_node_weight(input_node)
            node.output = self.activation_function(node_output)


class OutputLayer(StaticLayer):
    activation_function = lambda self, x: softmax(x)
    layer_type = Layer.LayerType.OUTPUT
    nodes: list[OutputNode]

    def compute(self, prev_layer: HiddenLayer):
        logits: typing.List[float] = []

        for node in self.nodes:
            z = node.bias
            for prev_node in prev_layer.nodes:
                z += prev_node.output * node.get_node_weight(prev_node)
            logits.append(z)

        probs = self.activation_function(logits)

        for node, p in zip(self.nodes, probs):
            node.output = p

        return self.get_prediction()

    def get_prediction(self) -> int:
        _ = [(idx, node.output) for idx, node in enumerate(self.nodes)]
        return sorted(_, key=lambda x: x[1], reverse=True)[0][0]
